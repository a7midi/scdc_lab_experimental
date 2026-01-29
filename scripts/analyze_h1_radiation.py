#!/usr/bin/env python3
"""
Analyze H1 "radiation" runs (detector flux vs time).

Input CSV columns (emitted by scdc_lab.experiments.h1_radiation_search):
  t, active_size, active_depth_max, detector_ge_count, detector_total, ...

Outputs:
  - a time-series plot of detector_ge_count(t) for each run
  - printed summary metrics (first hit time, peak, integrated count)
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def first_hit_tick(df: pd.DataFrame) -> int | None:
    hits = df[df["detector_ge_count"] > 0]
    if hits.empty:
        return None
    return int(hits["t"].iloc[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True, help="One or more radiation CSV files.")
    ap.add_argument("--label", nargs="*", default=None, help="Optional labels (same count as --csv).")
    ap.add_argument("--out_fig", required=True, help="Path to save the plot (png).")
    ap.add_argument("--title", default="Detector flux vs time (H1 radiation)", help="Figure title.")
    args = ap.parse_args()

    csv_paths: List[str] = args.csv
    labels: List[str] = args.label if args.label else [os.path.basename(p) for p in csv_paths]
    if len(labels) != len(csv_paths):
        raise SystemExit("If provided, --label must match the number of --csv inputs.")

    plt.figure(figsize=(8, 5))

    for path, label in zip(csv_paths, labels):
        df = pd.read_csv(path)
        if "t" not in df.columns or "detector_ge_count" not in df.columns:
            raise SystemExit(f"CSV missing required columns: {path}")

        t = df["t"].values
        y = df["detector_ge_count"].values

        first = first_hit_tick(df)
        peak = int(np.max(y)) if len(y) else 0
        total = int(np.sum(y)) if len(y) else 0
        active_peak = int(np.max(df["active_size"].values)) if "active_size" in df.columns else None

        print(f"== {label} ==")
        print(f"  first detector hit tick: {first}")
        print(f"  detector peak count:     {peak}")
        print(f"  detector integrated:     {total}")
        if active_peak is not None:
            print(f"  peak active size:        {active_peak}")

        plt.plot(t, y, label=label, linewidth=1.8)

    plt.xlabel("tick")
    plt.ylabel("detector_ge_count")
    plt.title(args.title)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out_fig), exist_ok=True)
    plt.savefig(args.out_fig, dpi=200)
    plt.close()
    print("Saved:", args.out_fig)


if __name__ == "__main__":
    main()
