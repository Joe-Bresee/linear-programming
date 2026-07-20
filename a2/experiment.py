"""
CSC 445 Assignment 2, Part 1 - Multi-instance row/column ratio experiment.

Benchmarks HiGHS dual-simplex solve time (in-process, presolve on/off)
across a spread of real Netlib LP instances -- from near-square through
moderately wide to extremely wide -- plus the two genuine primal/dual
pairs (fit1d/fit1p, fit2d/fit2p). All instances stored in standard form
(Ax = b, lo <= x <= hi) must have cols >= rows for full row rank, so
"tall" (rows > cols) instances don't exist here; this set instead spans
the full range from near-square out to extreme.

Requires: numpy, scipy
    pip install numpy scipy --break-system-packages
"""

import os
import time
import json
import urllib.request
import numpy as np
from scipy.io import loadmat
from scipy.optimize import linprog

BASE_URL = "https://sparse.tamu.edu/mat/LPnetlib/{}.mat"

# Ordered roughly by cols:rows ratio, near-square to extreme wide.
INSTANCES = [
    "lp_brandy",    # 220 x 303      ~1.38:1
    "lp_agg",       # 488 x 615      ~1.26:1
    "lp_afiro",     # 27 x 51        ~1.9:1
    "lp_25fv47",    # 821 x 1876     ~2.3:1
    "lp_cre_a",     # 3516 x 7248    ~2.1:1
    "lp_fit1p",     # 627 x 1677     ~2.7:1  (real dual pair w/ fit1d)
    "lp_80bau3b",   # 2262 x 12061   ~5.3:1
    "lp_cre_d",     # 8926 x 73948   ~8.3:1
    "lp_fit1d",     # 24 x 1049      ~44:1   (real dual pair w/ fit1p)
    "lp_fit2p",     # 3000 x 13525   ~4.5:1  (real dual pair w/ fit2d)
    "lp_fit2d",     # 25 x 10524     ~421:1  (real dual pair w/ fit2p)
]

DATA_DIR = "netlib_lp_data"
N_TRIALS = 5          # repeat each solve, report the median (timing is noisy)
PRESOLVE_SETTINGS = [True, False]


def download_instance(name, out_dir=DATA_DIR):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.mat")
    if os.path.exists(path):
        print(f"{name}: already downloaded, skipping.")
        return path
    url = BASE_URL.format(name)
    print(f"Downloading {name} from {url} ...")
    urllib.request.urlretrieve(url, path)
    return path


def load_instance(path):
    """
    SuiteSparse .mat files store everything inside a top-level 'Problem'
    struct. loadmat with squeeze_me=True, struct_as_record=False lets us
    access fields by name: mat['Problem'].A, .b, .c, .lo, .hi, .z0

    Fields are split inconsistently across instances: b sits directly on
    Problem, while c/lo/hi/z0 sit under Problem.aux. Check both locations
    per field rather than assuming one source holds everything.
    """
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    prob = mat["Problem"]
    A = prob.A
    if hasattr(A, "tocsc"):
        A = A.tocsc()

    aux = getattr(prob, "aux", None)

    def field(name, required=True):
        if hasattr(prob, name):
            return getattr(prob, name)
        if aux is not None and hasattr(aux, name):
            return getattr(aux, name)
        if required:
            prob_fields = getattr(prob, "_fieldnames", dir(prob))
            aux_fields = getattr(aux, "_fieldnames", dir(aux)) if aux is not None else None
            raise AttributeError(
                f"Field '{name}' not found on Problem or Problem.aux. "
                f"Problem fields: {prob_fields}. Problem.aux fields: {aux_fields}"
            )
        return None

    b = np.asarray(field("b"), dtype=float).ravel()
    c = np.asarray(field("c"), dtype=float).ravel()
    lo = np.asarray(field("lo"), dtype=float).ravel()
    hi = np.asarray(field("hi"), dtype=float).ravel()
    z0 = field("z0", required=False)
    return A, b, c, lo, hi, z0


def solve_once(A, b, c, lo, hi, presolve):
    bounds = list(zip(lo, hi))
    t0 = time.perf_counter()
    res = linprog(
        c, A_eq=A, b_eq=b, bounds=bounds,
        method="highs-ds",  # dual-simplex backend -- consistent with A1
        options={"presolve": presolve},
    )
    t_s = time.perf_counter() - t0
    return res, t_s


def benchmark_instance(name, path):
    A, b, c, lo, hi, z0 = load_instance(path)
    m, n = A.shape
    print(f"\n{name}: {m} rows x {n} cols "
          f"(ratio {n/m:.2f}:1)  nnz={getattr(A, 'nnz', 'n/a')}  known_optimal={z0}")

    rows = []
    for presolve in PRESOLVE_SETTINGS:
        times, iters, status, obj = [], None, None, None
        for _ in range(N_TRIALS):
            res, t_s = solve_once(A, b, c, lo, hi, presolve)
            times.append(t_s)
            iters = getattr(res, "nit", None)
            status = res.status
            obj = res.fun
        row = {
            "instance": name, "m": int(m), "n": int(n), "ratio": n / m,
            "presolve": presolve,
            "median_time_s": float(np.median(times)),
            "iterations": iters, "status": status,
            "objective": obj, "known_optimal": float(z0) if z0 is not None else None,
        }
        rows.append(row)
        print(f"  presolve={presolve!s:5} median_time={row['median_time_s']:.5f}s "
              f"iters={iters} status={status} obj={obj}")
    return rows


def main():
    all_rows = []
    for name in INSTANCES:
        try:
            path = download_instance(name)
            all_rows.extend(benchmark_instance(name, path))
        except Exception as e:
            print(f"!! Skipping {name}: {e}")

    with open("fit_primal_dual_results.json", "w") as f:
        json.dump(all_rows, f, indent=2)
    print("\nSaved results to fit_primal_dual_results.json")


if __name__ == "__main__":
    main()