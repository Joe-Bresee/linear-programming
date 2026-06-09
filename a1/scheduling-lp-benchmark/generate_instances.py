import os
import numpy as np


def generate_instance(n_jobs, n_resources=10, seed=42):
    rng = np.random.default_rng(seed)
    n_vars = n_jobs * n_resources

    w = rng.uniform(0.1, 10.0, size=(n_jobs,))
    c = np.repeat(w, n_resources)

    A_ub = np.zeros((n_resources, n_vars))
    for r in range(n_resources):
        for i in range(n_jobs):
            A_ub[r, i * n_resources + r] = 1.0
    capacity = rng.uniform(0.5 * n_jobs, 1.5 * n_jobs, size=(n_resources,))

    A_eq = np.zeros((n_jobs, n_vars))
    for i in range(n_jobs):
        A_eq[i, i * n_resources:(i + 1) * n_resources] = 1.0
    demand = rng.uniform(0.2 * n_resources, 0.8 * n_resources, size=(n_jobs,))

    return {
        "n_jobs": int(n_jobs),
        "n_resources": int(n_resources),
        "c": c.astype(float),
        "A_ub": A_ub.astype(float),
        "b_ub": capacity.astype(float),
        "A_eq": A_eq.astype(float),
        "b_eq": demand.astype(float),
    }


def save_instance(instance, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
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


def generate_batch(sizes=(10, 50, 100, 500, 1000), n_resources=10,
                   out_dir="instances", n_instances=5, base_seed=0):
    os.makedirs(out_dir, exist_ok=True)
    for n in sizes:
        for i in range(n_instances):
            seed = base_seed + n * 1000 + i   # deterministic, unique per (n, i)
            inst = generate_instance(n, n_resources=n_resources, seed=seed)
            fname = f"instance_n{n}_r{n_resources}_s{i}.npz"
            path = os.path.join(out_dir, fname)
            save_instance(inst, path)
            print(f"Saved {path}  (seed={seed})")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", type=int, default=[10, 50, 100, 500, 1000])
    p.add_argument("--resources", type=int, default=10)
    p.add_argument("--out", default="instances")
    p.add_argument("--n-instances", type=int, default=5,
                   help="Number of independently seeded instances per problem size")
    p.add_argument("--base-seed", type=int, default=0,
                   help="Base seed; actual seed = base_seed + n*1000 + instance_index")
    args = p.parse_args()

    generate_batch(
        sizes=tuple(args.sizes),
        n_resources=args.resources,
        out_dir=args.out,
        n_instances=args.n_instances,
        base_seed=args.base_seed,
    )