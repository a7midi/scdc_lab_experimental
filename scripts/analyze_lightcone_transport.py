#!/usr/bin/env python3
"""
Analyze horizon-limited light-cone transport across a density scan.

Input:
  - a directory containing *_summary.json files emitted by
    scdc_lab.experiments.unified_consistency_universe

Outputs:
  - lightcone-level CSV (one row per injected source)
  - run-level CSV (one row per run/seed)
  - pf-level CSV (aggregated over runs)
  - Figure: regime fractions vs p_forward
  - Figure: volume spectrum (pf=0.12 vs pf=0.14)

This script is used by scripts/run_prl_transport_figures.sh
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PF_SEED_RE = re.compile(r"pf(?P<pf>[0-9.]+)_seed(?P<seed>[0-9]+)_summary\.json$")


def parse_pf_seed(path: str) -> Tuple[float, int]:
    m = PF_SEED_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Could not parse pf/seed from filename: {path}")
    return float(m.group("pf")), int(m.group("seed"))


def classify(v_last: float, dead_lt: float, loc_le: float) -> str:
    if v_last < dead_lt:
        return "dead"
    if v_last > loc_le:
        return "shockwave"
    return "localized"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, required=True,
                    help="Directory containing pf*_seed*_summary.json files.")
    ap.add_argument("--dead_lt", type=float, default=5.0,
                    help="Dead threshold: v_last < dead_lt")
    ap.add_argument("--loc_le", type=float, default=20.0,
                    help="Shockwave threshold: v_last > loc_le; otherwise localized")
    ap.add_argument("--out_lightcone_csv", type=str, required=True)
    ap.add_argument("--out_run_csv", type=str, required=True)
    ap.add_argument("--out_pf_csv", type=str, required=True)
    ap.add_argument("--out_fig_regimes", type=str, required=True)
    ap.add_argument("--out_fig_spectrum", type=str, required=True)
    ap.add_argument("--spectrum_pf_a", type=float, default=0.12)
    ap.add_argument("--spectrum_pf_b", type=float, default=0.14)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.runs_dir, "pf*_seed*_summary.json")))
    if not paths:
        raise SystemExit(f"No summary json files found in: {args.runs_dir}")

    lightcone_rows: List[Dict] = []
    run_rows: List[Dict] = []

    for path in paths:
        pf, seed = parse_pf_seed(path)
        with open(path, "r") as f:
            d = json.load(f)

        light_cones = d.get("light_cones", [])
        if not isinstance(light_cones, list) or len(light_cones) == 0:
            continue

        statuses = []
        v_lasts = []
        for i, lc in enumerate(light_cones):
            vols = lc.get("volumes", [])
            v_last = float(vols[-1]) if vols else float("nan")
            status = classify(v_last, args.dead_lt, args.loc_le)
            statuses.append(status)
            v_lasts.append(v_last)

            lightcone_rows.append({
                "pf": pf,
                "seed": seed,
                "run": os.path.basename(path).replace("_summary.json", ""),
                "source_i": i,
                "v_last": v_last,
                "growth_model": lc.get("growth_model", ""),
                "eff_dim": lc.get("eff_dim", np.nan),
                "status": status,
            })

        # run-level aggregation (per seed)
        statuses = np.array(statuses)
        n = len(statuses)
        dead_ct = int(np.sum(statuses == "dead"))
        loc_ct = int(np.sum(statuses == "localized"))
        shock_ct = int(np.sum(statuses == "shockwave"))
        run_rows.append({
            "pf": pf,
            "seed": seed,
            "n_sources": n,
            "dead_ct": dead_ct,
            "localized_ct": loc_ct,
            "shockwave_ct": shock_ct,
            "dead_frac": dead_ct / n if n else np.nan,
            "localized_frac": loc_ct / n if n else np.nan,
            "shockwave_frac": shock_ct / n if n else np.nan,
            "v_last_mean": float(np.mean(v_lasts)) if v_lasts else np.nan,
            "v_last_max": float(np.max(v_lasts)) if v_lasts else np.nan,
        })

    lc_df = pd.DataFrame(lightcone_rows)
    run_df = pd.DataFrame(run_rows)

    os.makedirs(os.path.dirname(args.out_lightcone_csv), exist_ok=True)
    lc_df.to_csv(args.out_lightcone_csv, index=False)
    run_df.to_csv(args.out_run_csv, index=False)

    # pf-level aggregation: pool all sources (good for binomial counting)
    pf_rows = []
    for pf, g in lc_df.groupby("pf"):
        n = len(g)
        dead = int(np.sum(g["status"] == "dead"))
        loc = int(np.sum(g["status"] == "localized"))
        shock = int(np.sum(g["status"] == "shockwave"))
        pf_rows.append({
            "pf": float(pf),
            "n_sources": n,
            "dead_frac": dead / n if n else np.nan,
            "localized_frac": loc / n if n else np.nan,
            "shockwave_frac": shock / n if n else np.nan,
            "dead_ct": dead,
            "localized_ct": loc,
            "shockwave_ct": shock,
        })
    pf_df = pd.DataFrame(pf_rows).sort_values("pf")
    pf_df.to_csv(args.out_pf_csv, index=False)

    # Figure 1: regime fractions vs density
    x = pf_df["pf"].values
    plt.figure(figsize=(8, 5))
    plt.plot(x, pf_df["dead_frac"].values, marker="o", label=f"dead (v<{args.dead_lt:g})")
    plt.plot(x, pf_df["localized_frac"].values, marker="o", label="localized")
    plt.plot(x, pf_df["shockwave_frac"].values, marker="o", label=f"shockwave (v>{args.loc_le:g})")
    plt.xlabel("p_forward")
    plt.ylabel("fraction of sources")
    plt.title("Transport regimes vs density (horizon-limited light-cone volume)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_fig_regimes, dpi=200)
    plt.close()

    # Figure 2: spectrum scatter for two pf values
    a = float(args.spectrum_pf_a)
    b = float(args.spectrum_pf_b)
    sel = lc_df[(lc_df["pf"].isin([a, b]))].copy()
    if len(sel) > 0:
        # Small x-jitter for visual separation
        jitter = (np.random.RandomState(0).rand(len(sel)) - 0.5) * 0.01
        sel["pf_jitter"] = sel["pf"].values + jitter

        plt.figure(figsize=(8, 5))
        for pf_val in [a, b]:
            s = sel[sel["pf"] == pf_val]
            plt.scatter(s["pf_jitter"], s["v_last"], label=f"pf={pf_val:g}", alpha=0.85)
        plt.yscale("log")
        plt.xlabel("p_forward")
        plt.ylabel("v_last at horizon (log scale)")
        plt.title(f"Horizon-limited volume spectrum (pf={a:g} vs pf={b:g})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.out_fig_spectrum, dpi=200)
        plt.close()

    print("Wrote:")
    print(" ", args.out_run_csv)
    print(" ", args.out_lightcone_csv)
    print(" ", args.out_pf_csv)
    print(" ", args.out_fig_regimes)
    print(" ", args.out_fig_spectrum)


if __name__ == "__main__":
    main()
