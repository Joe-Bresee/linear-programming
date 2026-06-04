"""Plot benchmark results from CSV and save figures into `plots/`.

Generates:
- `time_vs_njobs.png`: median solve time vs n_jobs per solver (log y-scale)
- `iters_vs_njobs.png`: median iteration count vs n_jobs per solver

Usage:
  python plot_results.py --results results/results.csv --out-dir plots
"""
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_time(df, out_path):
    df_time = df.dropna(subset=["time_s"]).copy()
    if df_time.empty:
        print("No timing data to plot.")
        return
    grouped = df_time.groupby(["n_jobs", "method"])['time_s'].median().unstack('method')
    ax = grouped.plot(marker='o', logy=True, figsize=(8, 5))
    ax.set_xlabel('n_jobs')
    ax.set_ylabel('Median solve time (s)')
    ax.set_title('Median solve time vs n_jobs')
    ax.grid(True, which='both', ls='--', lw=0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print('Wrote', out_path)


def plot_iters(df, out_path):
    if 'nit' not in df.columns:
        print('No iteration data present.')
        return
    df_it = df.dropna(subset=['nit']).copy()
    if df_it.empty:
        print('No iteration data to plot.')
        return
    grouped = df_it.groupby(['n_jobs', 'method'])['nit'].median().unstack('method')
    ax = grouped.plot(marker='o', figsize=(8, 5))
    ax.set_xlabel('n_jobs')
    ax.set_ylabel('Median iterations (nit)')
    ax.set_title('Median iteration count vs n_jobs')
    ax.grid(True, ls='--', lw=0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print('Wrote', out_path)


def main(results_csv, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(results_csv)
    # Ensure n_jobs numeric
    if 'n_jobs' in df.columns:
        df['n_jobs'] = pd.to_numeric(df['n_jobs'], errors='coerce')
    else:
        print('results CSV missing n_jobs column')
        return

    # Normalize method labels if present
    if 'method' in df.columns:
        df['method'] = df['method'].astype(str)

    plot_time(df, os.path.join(out_dir, 'time_vs_njobs.png'))
    plot_iters(df, os.path.join(out_dir, 'iters_vs_njobs.png'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--results', default='results/results.csv')
    p.add_argument('--out-dir', default='plots')
    args = p.parse_args()
    main(args.results, args.out_dir)
