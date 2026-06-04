"""Delete all files and subdirectories under the plots directory.

Usage:
  python clean_plots.py --plots-dir plots
"""
import os
import shutil
import argparse


def clean_plots(out_dir='plots'):
    if not os.path.exists(out_dir):
        print(f"Plots directory '{out_dir}' does not exist; nothing to clean.")
        return
    for name in os.listdir(out_dir):
        path = os.path.join(out_dir, name)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        except Exception as e:
            print(f"Failed to remove {path}: {e}")
    print(f"Cleaned plots in {out_dir}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--plots-dir', default='plots')
    args = p.parse_args()
    clean_plots(args.plots_dir)
