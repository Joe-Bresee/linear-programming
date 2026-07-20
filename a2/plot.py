"""
plot.py
Produces a grouped bar chart (median solve time, log scale).
Usage: python plot.py <input.json> <output.png> <"Plot Title">
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt

def load_results(path):
    with open(path) as f:
        data = json.load(f)
    
    # Extract method name for the title if not provided
    method_name = data[0].get("method", "Solver")
    
    ORDER = ["lp_fit1d", "lp_fit1p", "lp_fit2d", "lp_fit2p"]
    by_instance = {name: {} for name in ORDER}
    rows = {}
    
    for row in data:
        name = row["instance"]
        by_instance.setdefault(name, {})[row["presolve"]] = row["median_time_s"]
        rows[name] = row["m"]
        
    return by_instance, rows, method_name

def main():
    if len(sys.argv) < 3:
        print("Usage: python plot.py <input.json> <output.png> ['Title']")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    by_instance, rows, method_name = load_results(input_path)
    
    title = sys.argv[3] if len(sys.argv) > 3 else f"Solve time: real Netlib primal/dual pairs ({method_name})"

    present = [name for name in by_instance.keys() if by_instance[name]]
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
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", which="both", linestyle="--", alpha=0.4)

    for bars in (bars_on, bars_off):
        for b in bars:
            h = b.get_height()
            if h is None or h == 0:
                continue
            ax.annotate(f"{h:.4f}", (b.get_x() + b.get_width() / 2, h),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    main()