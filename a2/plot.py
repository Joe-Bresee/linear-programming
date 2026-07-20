"""
Plots the results from experiment.py (fit_primal_dual_results.json).

Primary figure: median solve time vs. number of rows (both log scale),
across every tested instance -- this is the direct test of "solve time
tracks constraint count", using real applications spanning near-square
to extremely wide. Presolve on/off shown as separate series. Points are
labeled with instance name and its cols:rows ratio.

Usage:
    python plot_results.py [path/to/fit_primal_dual_results.json]

Requires: matplotlib, numpy
    pip install matplotlib numpy --break-system-packages
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt

RESULTS_PATH = sys.argv[1] if len(sys.argv) > 1 else "fit_primal_dual_results.json"


def load_results(path):
    with open(path) as f:
        data = json.load(f)
    by_instance = {}
    for row in data:
        by_instance.setdefault(row["instance"], {"m": row["m"], "n": row["n"],
                                                   "ratio": row["ratio"]})
        by_instance[row["instance"]][row["presolve"]] = row["median_time_s"]
    return by_instance


def main():
    by_instance = load_results(RESULTS_PATH)
    names = sorted(by_instance, key=lambda k: by_instance[k]["m"])

    rows_m = np.array([by_instance[n]["m"] for n in names])
    on_times = np.array([by_instance[n].get(True, np.nan) for n in names])
    off_times = np.array([by_instance[n].get(False, np.nan) for n in names])

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.scatter(rows_m, on_times, s=70, color="#4C72B0", label="presolve on", zorder=3)
    ax.scatter(rows_m, off_times, s=70, color="#DD8452", marker="^", label="presolve off", zorder=3)

    for n, m_val, t_on in zip(names, rows_m, on_times):
        ratio = by_instance[n]["ratio"]
        label = f"{n.replace('lp_', '')}\n({ratio:.1f}:1)"
        ax.annotate(label, (m_val, t_on), textcoords="offset points",
                    xytext=(6, 6), fontsize=8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of rows / constraints (log scale)")
    ax.set_ylabel("Median solve time (s, log scale)")
    ax.set_title("Solve time vs. row count across real Netlib LP instances\n"
                  "(HiGHS dual simplex; labels show cols:rows ratio)")
    ax.grid(which="both", linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()
    fig.savefig("solve_time_vs_rows.png", dpi=200)
    fig.savefig("solve_time_vs_rows.pdf")
    print("Saved solve_time_vs_rows.png and solve_time_vs_rows.pdf")


if __name__ == "__main__":
    main()