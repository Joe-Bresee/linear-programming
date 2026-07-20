import os
import time
import json
import urllib.request
import numpy as np
from scipy.io import loadmat
from scipy.optimize import linprog

BASE_URL = "https://sparse.tamu.edu/mat/LPnetlib/{}.mat"
INSTANCES = ["lp_fit1d", "lp_fit1p", "lp_fit2d", "lp_fit2p"]
DATA_DIR = "netlib_lp_data"
N_TRIALS = 5
PRESOLVE_SETTINGS = [True, False]
METHODS = ["highs-ds", "highs-ipm"]

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
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    prob = mat["Problem"]
    A = prob.A
    if hasattr(A, "tocsc"):
        A = A.tocsc()

    def field(name):
        if hasattr(prob, name):
            return getattr(prob, name)
        if hasattr(prob, "aux") and hasattr(prob.aux, name):
            return getattr(prob.aux, name)
        raise AttributeError(f"Field '{name}' not found.")

    b = np.asarray(field("b"), dtype=float).ravel()
    c = np.asarray(field("c"), dtype=float).ravel()
    lo = np.asarray(field("lo"), dtype=float).ravel()
    hi = np.asarray(field("hi"), dtype=float).ravel()
    
    try:
        z0 = field("z0")
    except AttributeError:
        z0 = None
        
    return A, b, c, lo, hi, z0

def solve_once(A, b, c, lo, hi, presolve, method):
    bounds = list(zip(lo, hi))
    t0 = time.perf_counter()
    res = linprog(
        c, A_eq=A, b_eq=b, bounds=bounds,
        method=method,
        options={"presolve": presolve},
    )
    t_s = time.perf_counter() - t0
    return res, t_s

def benchmark_instance(name, path, method):
    A, b, c, lo, hi, z0 = load_instance(path)
    print(f"\n{name} ({method}): {A.shape[0]} rows x {A.shape[1]} cols")

    rows = []
    for presolve in PRESOLVE_SETTINGS:
        times, iters, status, obj = [], None, None, None
        for _ in range(N_TRIALS):
            res, t_s = solve_once(A, b, c, lo, hi, presolve, method)
            times.append(t_s)
            iters = getattr(res, "nit", None)
            status = res.status
            obj = res.fun
        row = {
            "instance": name, "m": int(A.shape[0]), "n": int(A.shape[1]),
            "method": method, "presolve": presolve,
            "median_time_s": float(np.median(times)),
            "iterations": iters, "status": status,
            "objective": obj, "known_optimal": float(z0) if z0 is not None else None,
        }
        rows.append(row)
        print(f"  presolve={presolve!s:5} median_time={row['median_time_s']:.5f}s iters={iters}")
    return rows

def main():
    for method in METHODS:
        print(f"\n{'='*40}\nRunning Method: {method}\n{'='*40}")
        all_rows = []
        for name in INSTANCES:
            path = download_instance(name)
            all_rows.extend(benchmark_instance(name, path, method))
        
        # Save a separate file for each method
        filename = f"results_{method.replace('-', '_')}.json"
        with open(filename, "w") as f:
            json.dump(all_rows, f, indent=2)
        print(f"\nSaved results to {filename}")

if __name__ == "__main__":
    main()