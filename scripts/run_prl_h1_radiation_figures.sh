#!/usr/bin/env bash
set -euo pipefail

# H1: "Radiation / waveguide" experiment
# Compares detector flux with a dense knot vs the same knot nodes with knot density set to 0.

OUTDIR="${1:-results}"
mkdir -p "${OUTDIR}"

CSV_DENSE="${OUTDIR}/h1_rad_knot_dense.csv"
CSV_NONE="${OUTDIR}/h1_rad_knot_none.csv"
FIG="${OUTDIR}/fig_h1_detector_flux.png"

python -m scdc_lab.experiments.h1_radiation_search \
  --n 800 --layers 40 \
  --p_forward 0.10 --p_skip 0.01 \
  --knot_k 8 --knot_layer 2 --knot_density 0.9 \
  --rule xor --excite ones \
  --detector_layer 30 --steps 200 --seed 1 \
  --out_csv "${CSV_DENSE}"

python -m scdc_lab.experiments.h1_radiation_search \
  --n 800 --layers 40 \
  --p_forward 0.10 --p_skip 0.01 \
  --knot_k 8 --knot_layer 2 --knot_density 0.0 \
  --rule xor --excite ones \
  --detector_layer 30 --steps 200 --seed 1 \
  --out_csv "${CSV_NONE}"

python scripts/analyze_h1_radiation.py \
  --csv "${CSV_DENSE}" "${CSV_NONE}" \
  --label "dense knot (p=0.9)" "no internal knot (p=0.0)" \
  --title "H1 radiation: dense knot amplifies detector flux" \
  --out_fig "${FIG}"

echo "Done. Wrote:"
echo "  ${CSV_DENSE}"
echo "  ${CSV_NONE}"
echo "  ${FIG}"
