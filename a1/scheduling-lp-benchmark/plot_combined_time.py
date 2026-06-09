import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'n_jobs' not in df.columns or 'time_s' not in df.columns or 'method' not in df.columns:
        raise ValueError(f"Missing required columns in {path}")
    df = df.copy()
    df['n_jobs'] = pd.to_numeric(df['n_jobs'], errors='coerce')
    df['time_s'] = pd.to_numeric(df['time_s'], errors='coerce')
    df['method'] = df['method'].astype(str)
    return df.dropna(subset=['n_jobs', 'time_s', 'method'])


def main(revised_csv: str, highs_csv: str, out_path: str) -> None:
    revised = load_results(revised_csv)
    highs = load_results(highs_csv)

    df = pd.concat([revised, highs], ignore_index=True)
    wanted = ['revised-simplex', 'highs-ds', 'highs-ipm']
    df = df[df['method'].isin(wanted)].copy()

    grouped = df.groupby(['n_jobs', 'method'])['time_s'].median().unstack('method')
    grouped = grouped.reindex(columns=wanted)

    ax = grouped.plot(marker='o', logy=True, figsize=(8, 5))
    ax.set_xlabel('n_jobs')
    ax.set_ylabel('Median solve time (s)')
    ax.set_title('Median solve time vs n_jobs')
    ax.grid(True, which='both', ls='--', lw=0.5)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print(f'Wrote {out_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--revised', default='scheduling-lp-benchmark/results/revised_simplex_n100.csv')
    parser.add_argument('--highs', default='scheduling-lp-benchmark/results/highs_all.csv')
    parser.add_argument('--out', default='scheduling-lp-benchmark/plots/time_vs_njobs_combined.png')
    args = parser.parse_args()
    main(args.revised, args.highs, args.out)
