import os
import time
import argparse
import numpy as np
import csv

from scipy.optimize import linprog


def load_instance(path):
    data = np.load(path)
    inst = {
        "n_jobs": int(data["n_jobs"].item()) if data["n_jobs"].shape == () else int(data["n_jobs"]),
        "n_resources": int(data["n_resources"].item()) if data["n_resources"].shape == () else int(data["n_resources"]),
        "c": data["c"],
        "A_ub": data["A_ub"],
        "b_ub": data["b_ub"],
        "A_eq": data["A_eq"],
        "b_eq": data["b_eq"],
    }
    return inst


def run_solver(inst, solve_method):
    c = inst["c"]
    A_ub = inst["A_ub"]
    b_ub = inst["b_ub"]
    A_eq = inst["A_eq"]
    b_eq = inst["b_eq"]
    bounds = [(0, 1)] * c.size

    t0 = time.perf_counter()
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=solve_method)
    t1 = time.perf_counter()
    runtime = t1 - t0
    nit = res.get("nit", None)
    return {
        "success": bool(res.success),
        "status": int(res.status) if hasattr(res, "status") else None,
        "message": str(res.message),
        "time_s": float(runtime),
        "nit": int(nit) if nit is not None else None,
    }


def benchmark_dir(inst_dir, out_csv, methods=("revised-simplex", "highs-ds", "highs-ipm")):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    files = sorted([f for f in os.listdir(inst_dir) if f.endswith('.npz')])
    rows = []
    for f in files:
        path = os.path.join(inst_dir, f)
        inst = load_instance(path)
        for m in methods:
            print(f"Running {m} on {f}...")
            try:
                r = run_solver(inst, solve_method=m)
            except Exception as e:
                r = {"success": False, "status": None, "message": str(e), "time_s": None, "nit": None}
            row = {
                "instance": f,
                "n_jobs": inst["n_jobs"],
                "n_resources": inst["n_resources"],
                "method": m,
                "success": r["success"],
                "status": r["status"],
                "message": r["message"],
                "time_s": r["time_s"],
                "nit": r["nit"],
            }
            rows.append(row)

    # write CSV
    keys = ["instance", "n_jobs", "n_resources", "method", "success", "status", "message", "time_s", "nit"]
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print("Wrote results to", out_csv)

    # Automatically generate plots from results
    try:
        import subprocess
        plots_dir = os.path.join(os.path.dirname(out_csv), '..', 'plots')
        plots_dir = os.path.normpath(plots_dir)
        cmd = [
            "python3",
            os.path.join(os.path.dirname(__file__), "plot_results.py"),
            "--results",
            out_csv,
            "--out-dir",
            plots_dir,
        ]
        print("Generating plots...")
        subprocess.run(cmd, check=False)
        print("Plot generation finished (errors, if any, printed above).")
    except Exception as e:
        print("Failed to run plot generator:", e)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--instances", default="instances")
    p.add_argument("--out", default="results/results.csv")
    p.add_argument("--methods", nargs="+", default=["revised-simplex", "highs-ds", "highs-ipm"],
                   help="Solver methods to run. Valid values: revised-simplex, highs-ds, highs-ipm")
    args = p.parse_args()
    benchmark_dir(args.instances, args.out, methods=tuple(args.methods))
