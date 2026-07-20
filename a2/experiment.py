"""
CSC 445 Assignment 2, Part 1 - Real primal/dual pair experiment.

Downloads the Netlib fit1/fit2 primal-dual LP pairs from the SuiteSparse
Matrix Collection and times HiGHS dual-simplex solves on each, with
presolve on and off, in-process (no subprocess overhead like the
GLPK_CMD/PuLP approach had).

Pairs (documented by Bob Fourer / Netlib as genuine primal-dual pairs
of the same underlying piecewise-linear data-fitting model):
    fit1d ( 24 x   1049)  <->  fit1p ( 627 x  1677)
    fit2d ( 25 x  10524)  <->  fit2p (3000 x 13525)

Each .mat file bundles fields A, b, c, lo, hi (and z0, the known
optimal objective value, useful as a correctness sanity check) such
that the LP is:

    minimize c^T x   s.t.   A x = b,   lo <= x <= hi

Requires: numpy, scipy
    pip install numpy scipy --break-system-packages

NOTE: I could not run this myself (no network access in my sandbox) --
this is built directly against the field layout documented on the
SuiteSparse Matrix Collection pages for these instances. If the field
names come back different when you actually load a file, print
`mat["Problem"].__dict__` or `dir(mat["Problem"])` to see what's
really there and adjust load_instance() accordingly.
"""

import os
import time
import json
import urllib.request
import numpy as np
from scipy.io import loadmat
from scipy.optimize import linprog

BASE_URL = "https://sparse.tamu.edu/mat/LPnetlib/{}.mat"
INSTANCES = ["lp_fit1d", "lp_fit1p", "lp_fit2d", "lp_fit2p", "lp_agg", "lp_cre_d"]
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
    """
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    prob = mat["Problem"]
    A = prob.A
    if hasattr(A, "tocsc"):
        A = A.tocsc()

    def field(name):
        # Standard fields like 'b' are usually on the top-level prob.
        if hasattr(prob, name):
            return getattr(prob, name)
        # LP-specific fields like 'c', 'lo', 'hi' are usually in prob.aux.
        if hasattr(prob, "aux") and hasattr(prob.aux, name):
            return getattr(prob.aux, name)
            
        raise AttributeError(f"Field '{name}' not found on Problem or Problem.aux.")

    b = np.asarray(field("b"), dtype=float).ravel()
    c = np.asarray(field("c"), dtype=float).ravel()
    lo = np.asarray(field("lo"), dtype=float).ravel()
    hi = np.asarray(field("hi"), dtype=float).ravel()
    
    # z0 is optional, so we handle it gracefully if it's entirely missing
    try:
        z0 = field("z0")
    except AttributeError:
        z0 = None
        
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
    print(f"\n{name}: {A.shape[0]} rows x {A.shape[1]} cols "
          f"nnz={getattr(A, 'nnz', 'n/a')}  known_optimal={z0}")

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
            "instance": name, "m": int(A.shape[0]), "n": int(A.shape[1]),
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
        path = download_instance(name)
        all_rows.extend(benchmark_instance(name, path))

    with open("fit_primal_dual_results.json", "w") as f:
        json.dump(all_rows, f, indent=2)
    print("\nSaved results to fit_primal_dual_results.json")


if __name__ == "__main__":
    main()