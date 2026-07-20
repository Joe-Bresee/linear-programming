"""
Plots the results from fit_primal_dual_benchmark.py (fit_primal_dual_results.json).

Produces a grouped bar chart: median solve time (log scale) for each of
the four instances (fit1d, fit1p, fit2d, fit2p), with presolve on/off
shown side by side, so the row-count effect and the presolve effect can
both be read off the same figure.

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
ORDER = ["lp_fit1d", "lp_fit1p", "lp_fit2d", "lp_fit2p"]


def load_results(path):
    with open(path) as f:
        data = json.load(f)
    by_instance = {name: {} for name in ORDER}
    rows = {}
    for row in data:
        name = row["instance"]
        by_instance.setdefault(name, {})[row["presolve"]] = row["median_time_s"]
        rows[name] = row["m"]
    return by_instance, rows


def main():
    by_instance, rows = load_results(RESULTS_PATH)
    present = [name for name in ORDER if name in by_instance]
    if not present:
        present = list(by_instance.keys())

    labels = [f"{name.replace('lp_', '')}\n({rows[name]} rows)" for name in present]
    on_times = [by_instance[name].get(True) for name in present]
    off_times = [by_instance[name].get(False) for name in present]

    x = np.arange(len(present))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars_on = ax.bar(x - width / 2, on_times, width, label="presolve on", color="#4C72B0")
    bars_off = ax.bar(x + width / 2, off_times, width, label="presolve off", color="#DD8452")

    ax.set_yscale("log")
    ax.set_ylabel("Median solve time (s, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Solve time: real Netlib primal/dual pairs (HiGHS dual simplex)")
    ax.legend()
    ax.grid(axis="y", which="both", linestyle="--", alpha=0.4)

    for bars in (bars_on, bars_off):
        for b in bars:
            h = b.get_height()
            if h is None:
                continue
            ax.annotate(f"{h:.4f}", (b.get_x() + b.get_width() / 2, h),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig("fit_primal_dual_timing.png", dpi=200)
    fig.savefig("fit_primal_dual_timing.pdf")
    print("Saved fit_primal_dual_timing.png and fit_primal_dual_timing.pdf")


if __name__ == "__main__":
    main()