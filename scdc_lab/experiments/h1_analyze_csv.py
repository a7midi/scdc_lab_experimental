
"""
H1 CSV analysis utility.

This reads the CSV emitted by h1_glider_search (or similar) and reports:
- pocket/active size statistics
- centroid drift slopes after a burn-in
- Jaccard stability statistics

IMPORTANT FIX:
Earlier versions incorrectly called "percolated" whenever the pocket rapidly stabilized
relative to its own maximum. That is not "near-global". This version only calls
percolation/localization if you provide --n_total (the total number of vertices).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def _robust_median(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.median(x))


def _first_tick_at_or_above(t: np.ndarray, y: np.ndarray, thr: float) -> Optional[int]:
    mask = np.isfinite(y) & (y >= thr)
    if not np.any(mask):
        return None
    return int(t[np.where(mask)[0][0]])


def _linreg_slope(t: np.ndarray, y: np.ndarray) -> float:
    """Slope of y ~ a + b t using least squares (b returned)."""
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]
    if t.size < 2:
        return float("nan")
    t0 = t - t.mean()
    denom = float(np.dot(t0, t0))
    if denom <= 0:
        return float("nan")
    b = float(np.dot(t0, y - y.mean()) / denom)
    return b


@dataclass
class Verdict:
    label: str
    details: str


def classify_pocket(
    pocket_median: float,
    pocket_max: float,
    n_total: Optional[int],
    percolation_frac: float,
    localized_frac: float,
) -> Verdict:
    """
    Classification by *fraction of the total vertex set*.

    - Percolated: median pocket fraction >= percolation_frac
    - Localized:  max pocket fraction <= localized_frac
    - Mesoscopic: otherwise
    """
    if n_total is None or n_total <= 0:
        return Verdict(
            "Unknown",
            "Pass --n_total to classify percolation/localization by fraction of total vertices.",
        )
    frac_med = pocket_median / float(n_total)
    frac_max = pocket_max / float(n_total)

    if frac_med >= percolation_frac:
        return Verdict(
            "Percolated",
            f"median pocket fraction={frac_med:.3f} >= {percolation_frac:.3f}",
        )
    if frac_max <= localized_frac:
        return Verdict(
            "Localized",
            f"max pocket fraction={frac_max:.3f} <= {localized_frac:.3f}",
        )
    return Verdict(
        "Mesoscopic",
        f"median fraction={frac_med:.3f}, max fraction={frac_max:.3f} (between thresholds)",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="CSV file from h1_glider_search (columns: t, active_size, ...)")
    ap.add_argument("--burn", type=int, default=20, help="Burn-in ticks to drop before drift slope fit.")
    ap.add_argument("--n_total", type=int, default=None, help="Total number of vertices (n). Enables percolation/localization verdict.")
    ap.add_argument("--percolation_frac", type=float, default=0.60, help="Median pocket fraction threshold for percolation.")
    ap.add_argument("--localized_frac", type=float, default=0.30, help="Max pocket fraction threshold for localization.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    required = ["t", "active_size", "active_centroid_depth", "pocket_size", "pocket_centroid_depth", "jaccard_prev", "active_jaccard_prev"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}. Got columns: {list(df.columns)}")

    t = df["t"].to_numpy(dtype=float)
    pocket = df["pocket_size"].to_numpy(dtype=float)
    active = df["active_size"].to_numpy(dtype=float)

    pocket_cent = df["pocket_centroid_depth"].to_numpy(dtype=float)
    active_cent = df["active_centroid_depth"].to_numpy(dtype=float)

    j_pocket = df["jaccard_prev"].to_numpy(dtype=float)
    j_active = df["active_jaccard_prev"].to_numpy(dtype=float)

    pocket_nonzero = int(np.sum(np.isfinite(pocket) & (pocket > 0)))
    rows = int(df.shape[0])

    pocket_min = float(np.nanmin(pocket))
    pocket_med = float(np.nanmedian(pocket))
    pocket_max = float(np.nanmax(pocket))

    stab_tick = _first_tick_at_or_above(t, pocket, 0.95 * pocket_max)

    # Drift slopes after burn
    burn = int(max(0, args.burn))
    mask_burn = t >= burn
    pocket_slope = _linreg_slope(t[mask_burn], pocket_cent[mask_burn])
    active_slope = _linreg_slope(t[mask_burn], active_cent[mask_burn])

    # Jaccards
    j_pocket_med = _robust_median(j_pocket)
    j_active_med = _robust_median(j_active)
    j_pocket_min = float(np.nanmin(j_pocket[np.isfinite(j_pocket)])) if np.any(np.isfinite(j_pocket)) else float("nan")
    j_active_min = float(np.nanmin(j_active[np.isfinite(j_active)])) if np.any(np.isfinite(j_active)) else float("nan")
    j_pocket_last = float(j_pocket[np.where(np.isfinite(j_pocket))[0][-1]]) if np.any(np.isfinite(j_pocket)) else float("nan")
    j_active_last = float(j_active[np.where(np.isfinite(j_active))[0][-1]]) if np.any(np.isfinite(j_active)) else float("nan")

    # Jaccards after burn
    j_pocket_med_burn = _robust_median(j_pocket[mask_burn])
    j_active_med_burn = _robust_median(j_active[mask_burn])
    j_pocket_min_burn = float(np.nanmin(j_pocket[mask_burn][np.isfinite(j_pocket[mask_burn])])) if np.any(np.isfinite(j_pocket[mask_burn])) else float("nan")
    j_active_min_burn = float(np.nanmin(j_active[mask_burn][np.isfinite(j_active[mask_burn])])) if np.any(np.isfinite(j_active[mask_burn])) else float("nan")

    verdict = classify_pocket(
        pocket_median=pocket_med,
        pocket_max=pocket_max,
        n_total=args.n_total,
        percolation_frac=float(args.percolation_frac),
        localized_frac=float(args.localized_frac),
    )

    # Pretty print
    print("=== H1 CSV Analysis ===")
    print(f"Rows: {rows}  (ticks {int(np.nanmin(t))}..{int(np.nanmax(t))})")
    if args.n_total is not None:
        print(f"n_total: {int(args.n_total)}")
        print(f"Pocket fraction: median={pocket_med/args.n_total:.3f} max={pocket_max/args.n_total:.3f}")
        print(f"Active fraction: median={float(np.nanmedian(active))/args.n_total:.3f} max={float(np.nanmax(active))/args.n_total:.3f}")

    print(f"Pocket size: min={pocket_min:.0f}  median={pocket_med:.1f}  max={pocket_max:.0f}")
    print(f"Stabilization tick (>= 0.95*max): {stab_tick}")
    print(f"Pocket centroid depth: start={float(pocket_cent[0]):.3f}  end={float(pocket_cent[-1]):.3f}")
    print(f"Drift slope after burn={burn}: pocket_slope={pocket_slope:.6g} depth/tick  active_slope={active_slope:.6g} depth/tick")
    print(f"Jaccard(prev): pocket median={j_pocket_med:.3f} min={j_pocket_min:.3f} last={j_pocket_last:.3f}")
    print(f"Jaccard(prev): active median={j_active_med:.3f} min={j_active_min:.3f} last={j_active_last:.3f}")
    print(f"Jaccard(prev) after burn: pocket median={j_pocket_med_burn:.3f} min={j_pocket_min_burn:.3f}")
    print(f"Jaccard(prev) after burn: active median={j_active_med_burn:.3f} min={j_active_min_burn:.3f}")

    print(f"Verdict: {verdict.label}. {verdict.details}")


if __name__ == "__main__":
    main()
