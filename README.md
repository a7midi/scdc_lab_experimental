# SCDC Lab — PRL Reproducibility Suite (v2.1)

This repository is a **self-contained, reproducible suite** used to generate the figures and CSVs for a short
PRL-style letter on **transport, localization, and waveguiding** in *schedule-consistent* causal DAGs.

The core idea is simple:

- A random directed acyclic graph (DAG) is treated as a **disordered causal medium**.
- A local update rule evolves binary activity forward in causal time.
- A **Schedule-Consistent Diamond Closure (SCDC)** enforces *local confluence* (commutation) on genuinely
  schedule-free diamonds, yielding an update rule that is robust to admissible update-order freedom.

The experiments in `scripts/` quantify how injected excitations propagate (die, localize, or blow up)
as a function of the forward-edge density, and how **dense local motifs ("knots")** can act as waveguides
that amplify radiation at a detector without changing reachability.

---

## Quick start

### 1) Create an environment and install
```bash
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -e .
```

### 2) Reproduce the PRL transport figures (Figure 1–2)
```bash
bash scripts/run_prl_transport_figures.sh
```

Outputs (written to `results/`):
- `fig_transport_regimes.png` — regime fractions vs density
- `fig_volume_spectrum_pf012_pf014.png` — volume spectrum at pf=0.12 vs pf=0.14
- `run_level_transport.csv` — per-run (seed) aggregation
- `lightcone_level_transport.csv` — per-source measurements
- `pf_level_transport.csv` — pooled fractions per density

### 3) Reproduce the PRL H1 radiation / waveguide figure (Figure 3)
```bash
bash scripts/run_prl_h1_radiation_figures.sh
```

Outputs (written to `results/`):
- `h1_rad_knot_dense.csv`
- `h1_rad_knot_none.csv`
- `fig_h1_detector_flux.png`

---

## What is measured?

### H0 transport metric: horizon-limited light-cone volume

Each run injects activity at `n_sources=8` randomly chosen sources and evolves for `tmax=8` ticks.
For each source we record the **horizon-limited volume**
\[
V_h \equiv |A(t_\mathrm{max})|
\]
(the number of active nodes at the final tick, confined to the future cone).

We then classify each source into a transport regime:
- **dead** if \(V_h < 5\)
- **localized** if \(5 \le V_h \le 20\)
- **shockwave** if \(V_h > 20\)

These thresholds are intentionally coarse: they separate obvious extinction from confined propagation and runaway growth.

> Important: Figure 1 reports **fractions of sources**, pooled across seeds at each density.  
> With 5 seeds and 8 sources per run, each density point typically has \(N=40\) samples.

### H1 radiation metric: detector flux

The radiation experiment evolves XOR dynamics on a layered DAG and records the **detector-layer flux**
(`detector_ge_count`) as a function of time. Two cases are compared with identical global parameters:

- a **dense knot** (internal knot density `--knot_density 0.9`)
- the **same knot nodes** but with **no internal knot wiring** (`--knot_density 0.0`)

This isolates whether local motif density can act as a **waveguide/lens**, amplifying detector flux.

---

## Notes on SCDC diamond confluence (audit fix)

`scdc.py` enforces diamond confluence on patterns \(x\to y\), \(x\to z\), \(y\to w\), \(z\to w\).

**Audit remark (correct):** If \(y\) reaches \(z\) (or vice versa), then one of the two update orders is anti-causal.
Forcing equality under the forbidden order can artificially "bleach" causal structure.

**Fix in this suite:** Diamonds are only enforced when \(y\) and \(z\) are *topologically independent*
(neither reaches the other) in the condensation DAG.

---

## Single-run reproduction (example)

To reproduce a specific run (e.g. pf=0.12, seed=1):

```bash
python -m scdc_lab.experiments.unified_consistency_universe \
  --graph_type layered \
  --n 500 --layers 100 \
  --p_forward 0.12 --p_skip2 0.02 --p_skip3 0.005 \
  --energy_mode motif --genesis_steps 1500 \
  --inject_knot_k 20 --knot_p 0.9 \
  --rule threshold --threshold 2 \
  --steps 120 \
  --seed 1 \
  --out_prefix runs/pf0.12_seed1
```

This produces:
- `runs/pf0.12_seed1_summary.json`
- `runs/pf0.12_seed1_spacetime.png`
- `runs/pf0.12_seed1_geodesics.png`

---

## Reproducibility checklist

For PRL-quality rigor, we recommend reporting:
- sample sizes (sources and seeds) per density
- uncertainty bands or confidence intervals (bootstrap or binomial)
- ablations: toggle SCDC closure, toggle knot injection, vary thresholds \(V_h\) used for classification
- parameter stability: verify conclusions persist under modest changes to `layers`, `n`, and `tmax`

