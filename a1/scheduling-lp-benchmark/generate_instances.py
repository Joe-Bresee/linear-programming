"""Generate LP instances for GPU/CPU scheduling benchmarking.

Creates instances saved as compressed numpy files in `instances/`.
"""
import os
import json
import numpy as np


def generate_instance(n_jobs, n_resources=10, seed=42):
    rng = np.random.default_rng(seed)
    # Variables: x_{i,r} for i in jobs, r in resources -> n = n_jobs * n_resources
    n_vars = n_jobs * n_resources

    # Objective: random positive weights per job expanded across resources
    w = rng.uniform(0.1, 10.0, size=(n_jobs,))
    c = np.repeat(w, n_resources)  # simple objective: cost per allocation

    # Capacity constraints: for each resource, sum over jobs <= capacity_r
    A_ub = np.zeros((n_resources, n_vars))
    for r in range(n_resources):
        for i in range(n_jobs):
            A_ub[r, i * n_resources + r] = 1.0
    capacity = rng.uniform(0.5 * n_jobs, 1.5 * n_jobs, size=(n_resources,))

    # Demand constraints: for each job, sum over resources >= demand_i
    A_eq = np.zeros((n_jobs, n_vars))
    for i in range(n_jobs):
        A_eq[i, i * n_resources:(i + 1) * n_resources] = 1.0
    demand = rng.uniform(0.2 * n_resources, 0.8 * n_resources, size=(n_jobs,))

    # Bounds: 0 <= x_ij <= 1 (fractional allocation)
    bounds = [(0.0, 1.0)] * n_vars

    instance = {
        "n_jobs": int(n_jobs),
        "n_resources": int(n_resources),
        "c": c.astype(float),
        "A_ub": A_ub.astype(float),
        "b_ub": capacity.astype(float),
        "A_eq": A_eq.astype(float),
        "b_eq": demand.astype(float),
        "bounds": bounds,
    }
    return instance


def save_instance(instance, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Save arrays to npz; bounds serialized as two 1D arrays
    np.savez_compressed(
        path,
        n_jobs=instance["n_jobs"],
        n_resources=instance["n_resources"],
        c=instance["c"],
        A_ub=instance["A_ub"],
        b_ub=instance["b_ub"],
        A_eq=instance["A_eq"],
        b_eq=instance["b_eq"],
    )


def generate_batch(sizes=(10, 50, 100, 500, 1000), n_resources=10, out_dir="instances", seeds=None):
    os.makedirs(out_dir, exist_ok=True)
    if seeds is None:
        seeds = [None] * len(sizes)
    for n, seed in zip(sizes, seeds):
        inst = generate_instance(n, n_resources=n_resources, seed=seed)
        fname = f"instance_n{n}_r{inst['n_resources']}.npz"
        path = os.path.join(out_dir, fname)
        save_instance(inst, path)
        print("Saved", path)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", type=int, default=[10, 50, 100, 500, 1000])
    p.add_argument("--resources", type=int, default=10)
    p.add_argument("--out", default="instances")
    args = p.parse_args()
    generate_batch(sizes=tuple(args.sizes), n_resources=args.resources, out_dir=args.out)
