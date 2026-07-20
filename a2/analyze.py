"""
Analysis + output generation for CSC 445 A2 Part 1.
Reads {prefix}_raw_timing_data.json and produces outputs prefixed with the config name.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent
OUT_DIR = PROJECT_DIR

parser = argparse.ArgumentParser(description="Analyze simplex experiments.")
parser.add_argument("--config", type=str, required=True, help="Path to the JSON config file used.")
args = parser.parse_args()

prefix = Path(args.config).stem

data_file = OUT_DIR / f"{prefix}_raw_timing_data.json"
if not data_file.exists():
    raise FileNotFoundError(f"Data file {data_file} not found. Run experiment.py first.")

with open(data_file) as f:
    raw = json.load(f)

df = pd.DataFrame(raw)
orientation_order = ["wide", "tall"]
orientation_labels = {
    "wide": "wide primal sweep (n/m >= 1)",
    "tall": "tall primal sweep (m/n >= 1)",
}

# --------------------------------------------------------------------------
# 1. Table
# --------------------------------------------------------------------------
grouped = (
    df.groupby(["orientation", "m", "n", "ratio", "presolve", "algorithm"])
      .agg(median_time_s=("time_s", "median"), median_iters=("iters", "median"))
      .reset_index()
)

table = grouped.pivot_table(
    index=["orientation", "m", "n", "ratio", "presolve"],
    columns="algorithm",
    values=["median_time_s", "median_iters"],
).reset_index()
table.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in table.columns.values]
table = table.sort_values(["orientation", "presolve", "ratio"], ascending=[True, False, True])
table_out = OUT_DIR / f"{prefix}_table.csv"
table.to_csv(table_out, index=False)
print(f"Wrote {table_out} ({len(table)} rows)")

# --------------------------------------------------------------------------
# Dynamic Plot Generation (Loops over Presolve Settings)
# --------------------------------------------------------------------------
presolve_settings = df["presolve"].unique()

for presolve_val in presolve_settings:
    
    # 2. Figure: solve time vs ratio
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, orientation in zip(axes, orientation_order):
        sub = grouped[(grouped["orientation"] == orientation) & (grouped["presolve"] == presolve_val)]
        for algo, marker in [("primal_simplex", "o"), ("dual_simplex", "s")]:
            s = sub[sub["algorithm"] == algo].sort_values("ratio")
            ax.plot(s["ratio"], s["median_time_s"], marker=marker, label=algo.replace('_', ' ').title(), linewidth=1.5)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{orientation_labels[orientation]} | presolve={presolve_val}")
        ax.set_xlabel("stretch ratio (>= 1)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("median solve time (s, log scale)")
    fig.suptitle(f"[{prefix}] Solve Time (Identical Formulation)")
    fig.tight_layout()
    fig_out = OUT_DIR / f"{prefix}_fig1_time_vs_ratio_presolve_{presolve_val}.png"
    fig.savefig(fig_out, dpi=150)
    plt.close(fig)

    # 3. Figure: simplex iterations vs ratio
    fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, orientation in zip(axes3, orientation_order):
        sub = grouped[(grouped["orientation"] == orientation) & (grouped["presolve"] == presolve_val)]
        for algo, marker in [("primal_simplex", "o"), ("dual_simplex", "s")]:
            s = sub[sub["algorithm"] == algo].sort_values("ratio")
            ax.plot(s["ratio"], s["median_iters"], marker=marker, label=algo.replace('_', ' ').title(), linewidth=1.5)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{orientation_labels[orientation]} | presolve={presolve_val}")
        ax.set_xlabel("stretch ratio (>= 1)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    axes3[0].set_ylabel("median simplex iterations (log scale)")
    fig3.suptitle(f"[{prefix}] Iteration Count")
    fig3.tight_layout()
    fig3_out = OUT_DIR / f"{prefix}_fig3_iters_vs_ratio_presolve_{presolve_val}.png"
    fig3.savefig(fig3_out, dpi=150)
    plt.close(fig3)

    # 4. Time-per-iteration
    df["time_per_iter"] = df["time_s"] / df["iters"].replace(0, np.nan)
    grouped_tpi = (
        df.groupby(["orientation", "ratio", "presolve", "algorithm"])
          .agg(median_time_per_iter=("time_per_iter", "median"))
          .reset_index()
    )
    fig4, axes4 = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, orientation in zip(axes4, orientation_order):
        sub = grouped_tpi[(grouped_tpi["orientation"] == orientation) & (grouped_tpi["presolve"] == presolve_val)]
        for algo, marker in [("primal_simplex", "o"), ("dual_simplex", "s")]:
            s = sub[sub["algorithm"] == algo].sort_values("ratio")
            ax.plot(s["ratio"], s["median_time_per_iter"], marker=marker, label=algo.replace('_', ' ').title(), linewidth=1.5)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{orientation_labels[orientation]} | presolve={presolve_val}")
        ax.set_xlabel("stretch ratio (>= 1)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    axes4[0].set_ylabel("median time per iteration (s, log scale)")
    fig4.suptitle(f"[{prefix}] Per-iteration cost")
    fig4.tight_layout()
    fig4_out = OUT_DIR / f"{prefix}_fig4_time_per_iter_vs_ratio_presolve_{presolve_val}.png"
    fig4.savefig(fig4_out, dpi=150)
    plt.close(fig4)

print(f"Finished generating plots for {prefix}.")