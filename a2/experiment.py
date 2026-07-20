"""
CSC 445 Assignment 2, Part 1 - Updated for Numerical Stability
"""

import argparse
import json
import time
import os
import re
import numpy as np
import pulp as pl
from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
OUT_DIR = PROJECT_DIR
RNG_MASTER_SEED = 42

@dataclass
class Instance:
    m: int
    n: int
    ratio: float
    orientation: str
    seed: int
    A: np.ndarray
    b: np.ndarray
    c: np.ndarray
    sense: np.ndarray
    nnz: int

def build_ratio_grid(cfg):
    return [round(float(r), 6) for r in np.geomspace(1.0, cfg["RATIO_MAX"], cfg["N_RATIO_POINTS"])]

def dims_for_ratio(ratio: float, orientation: str, cfg: dict):
    if orientation == "wide":
        m = cfg["BASE_DIM"]
        n = max(cfg["BASE_DIM"], round(cfg["BASE_DIM"] * ratio))
    elif orientation == "tall":
        n = cfg["BASE_DIM"]
        m = max(cfg["BASE_DIM"], round(cfg["BASE_DIM"] * ratio))
    return m, n

def generate_instance(m, n, ratio, orientation, seed, cfg: dict, max_attempts=50) -> Instance:
    rng = np.random.default_rng(seed)
    for attempt in range(max_attempts):
        mask = rng.random((m, n)) < cfg["DENSITY"]
        vals = rng.lognormal(mean=0.0, sigma=0.5, size=(m, n))
        A = np.where(mask, vals, 0.0)

        min_dim = min(m, n)
        A[:min_dim, :min_dim] += np.eye(min_dim) * 0.1
        if np.linalg.matrix_rank(A, tol=1e-5) < min_dim:
            continue

        c = rng.lognormal(mean=0.0, sigma=0.3, size=n)
        x0 = rng.lognormal(mean=0.0, sigma=0.25, size=n)
        base = A @ x0

        # Randomly assign each row as a minimum (>=) or maximum (<=)
        # requirement -- roughly half and half. x=0 satisfies the <=
        # rows and violates the >= rows, so neither the trivial primal
        # basis nor the trivial dual basis is uniformly feasible.
        is_min_row = rng.random(m) < 0.5
        b = np.empty(m)
        sense = np.empty(m, dtype=object)
        b[is_min_row] = base[is_min_row] * rng.uniform(0.3, 0.7, size=is_min_row.sum())
        sense[is_min_row] = "GE"
        b[~is_min_row] = base[~is_min_row] * rng.uniform(1.3, 2.0, size=(~is_min_row).sum())
        sense[~is_min_row] = "LE"

        return Instance(m=m, n=n, ratio=ratio, orientation=orientation, seed=seed,
                         A=A, b=b, c=c, sense=sense, nnz=int(np.count_nonzero(A)))
    raise RuntimeError("Failed to generate a stable, full-rank instance.")

def build_pulp_problem(inst: Instance):
    prob = pl.LpProblem("Primal_Formulation", pl.LpMinimize)
    x_vars = [pl.LpVariable(f"x_{j}", lowBound=0, cat=pl.LpContinuous) for j in range(inst.n)]
    prob += pl.lpSum(inst.c[j] * x_vars[j] for j in range(inst.n))
    for i in range(inst.m):
        lhs = pl.lpSum(inst.A[i, j] * x_vars[j] for j in range(inst.n))
        if inst.sense[i] == "GE":
            prob += lhs >= inst.b[i]
        else:
            prob += lhs <= inst.b[i]
    return prob, x_vars

def get_glpk_iters(log_path):
    if not os.path.exists(log_path):
        return 0

    last_iter = 0

    with open(log_path) as f:
        for line in f:
            m = re.match(r'^\s*[*#-]?\s*(\d+):', line)
            if m:
                last_iter = int(m.group(1))

    return last_iter

def solve_with_algorithm(prob: pl.LpProblem, x_vars: list, inst: Instance, presolve: bool, algo: str, cfg: dict, timed: bool = True):
    log_file = f"glpk_{algo}_temp.log"
    options = ["--primal" if algo == "primal_simplex" else "--dual"]
    if not presolve: options.append("--nopresol")
    options.extend(["--log", log_file])
    solver = pl.GLPK_CMD(msg=False, options=options, keepFiles=False)
    
    def _solve_once():
        t0 = time.perf_counter()
        status = prob.solve(solver)
        return status, time.perf_counter() - t0, get_glpk_iters(log_file)

    if not timed:
        status, t, iters = _solve_once()
        res = {"status": status, "x": np.array([v.varValue if v.varValue is not None else 0 for v in x_vars])}
        # if os.path.exists(log_file): os.remove(log_file)
        return res, t, iters
        
    times = []
    status, t, iters = _solve_once()
    res = {"status": status, "x": np.array([v.varValue if v.varValue is not None else 0 for v in x_vars])}
    # if os.path.exists(log_file): os.remove(log_file)
    return res, t, iters

def correctness_check(prob: pl.LpProblem, x_vars: list, inst: Instance, cfg: dict):
    res_ps, _, _ = solve_with_algorithm(prob, x_vars, inst, True, "primal_simplex", cfg, timed=False)
    res_ds, _, _ = solve_with_algorithm(prob, x_vars, inst, True, "dual_simplex", cfg, timed=False)
    
    if res_ps["status"] != pl.LpStatusOptimal or res_ds["status"] != pl.LpStatusOptimal:
        return False, float('inf')

    # Relative error check
    norm = np.max(np.abs(res_ps["x"])) + 1e-9
    max_rel_diff = float(np.max(np.abs(res_ps["x"] - res_ds["x"])) / norm)
    return max_rel_diff <= 1e-4, max_rel_diff

def run_experiment(cfg: dict):
    ratio_grid = build_ratio_grid(cfg)
    rows, correctness_rows = [], []
    for orientation in ["wide", "tall"]:
        for ratio in ratio_grid:
            m, n = dims_for_ratio(ratio, orientation, cfg)
            for seed_idx in range(cfg["N_SEEDS"]):
                inst = generate_instance(m, n, ratio, orientation, seed_idx, cfg)
                prob, x_vars = build_pulp_problem(inst)
                ok, diff = correctness_check(prob, x_vars, inst, cfg)
                correctness_rows.append({"m": m, "n": n, "ratio": ratio, "within_tol": ok})
                for presolve in cfg["PRESOLVE_SETTINGS"]:
                    for algo in ["primal_simplex", "dual_simplex"]:
                        res, t_s, iters = solve_with_algorithm(prob, x_vars, inst, presolve, algo, cfg)
                        rows.append({"orientation": orientation, "m": m, "n": n, "ratio": ratio, "presolve": presolve, "algorithm": algo, "time_s": t_s, "iters": iters})
                print(f"Done: {orientation} ratio={ratio:.2f}")
    return rows, correctness_rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, 'r') as f: config = json.load(f)
    rows, corr = run_experiment(config)
    prefix = Path(args.config).stem
    with open(f"{prefix}_raw_timing_data.json", "w") as f: json.dump(rows, f, indent=2)