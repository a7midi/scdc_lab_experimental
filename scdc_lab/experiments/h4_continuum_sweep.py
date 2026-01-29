"""H4 continuum-limit sweep for knot renormalization.

Goal
----
Turn the "knot renormalization" idea into a *clean, reproducible* experiment module
that supports a continuum-limit style sweep:

  - micro-knot generation (small directed multigraph between N input wires and N output wires)
  - SCDC closure -> quotient world
  - renormalized pocket transfer map f_eff : Inputs -> Outputs
  - (i) discrete symmetry skeleton: which wire permutations commute with f_eff
  - (ii) continuum proxy: fit a complex linear map M on a roots-of-unity embedding of Z_k

This script is exploratory: it does NOT claim SU(2)/SU(3) emergence.
It produces auditable CSVs so you can test whether increasing alphabet size k and/or
knot depth drives transfer maps toward approximately unitary mixing.

Usage examples
--------------

Small exact (enumerate all input states):
  python -m scdc_lab.experiments.h4_continuum_sweep \
    --n_wires_list 2 3 --k_list 2 3 4 --depth_list 2 3 4 \
    --samples_per_cell 50 --density 0.4 --out_dir results/h4_sweep_small

Continuum proxy (sampled IO; safe for k=128):
  python -m scdc_lab.experiments.h4_continuum_sweep \
    --n_wires_list 3 --k_list 8 16 32 64 128 --depth_list 3 4 6 \
    --samples_per_cell 50 --io_samples 2048 --rule random_hash \
    --scdc_samples 1024 --diamond_samples 64 --out_dir results/h4_sweep_k128
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import networkx as nx
from sympy.combinatorics import Permutation, PermutationGroup

from scdc_lab.world import WorldInstance, make_rule_factory
from scdc_lab.scdc import SCDCConfig, compute_lambda_star, quotient_world


# -------------------------
# Graph / knot generation
# -------------------------


def _product(xs: Sequence[int]) -> int:
    out = 1
    for x in xs:
        out *= int(x)
    return int(out)


def generate_random_knot(
    *,
    n_wires: int,
    depth: int,
    density: float,
    rng: np.random.Generator,
    tangle_factor: float = 2.0,
    force_dag: bool = True,
) -> Tuple[nx.MultiDiGraph, List[int], List[int]]:
    """Random "knot" connecting N inputs to N outputs.

    The structure is a feed-forward backbone with additional skip/tangle edges.
    For transfer-map extraction we typically force a DAG (u < v constraint).
    """

    n_wires = int(n_wires)
    depth = int(depth)
    density = float(density)

    if n_wires <= 0:
        raise ValueError("n_wires must be positive")
    if depth < 0:
        raise ValueError("depth must be >= 0")
    if not (0.0 <= density <= 1.0):
        raise ValueError("density must be in [0,1]")

    G = nx.MultiDiGraph()
    total_nodes = 2 * n_wires + depth * n_wires
    G.add_nodes_from(range(total_nodes))

    input_nodes = list(range(n_wires))
    output_nodes = list(range(n_wires + depth * n_wires, 2 * n_wires + depth * n_wires))

    layers: List[List[int]] = []
    layers.append(input_nodes)

    cursor = n_wires
    for _ in range(depth):
        layer = list(range(cursor, cursor + n_wires))
        layers.append(layer)
        cursor += n_wires
    layers.append(output_nodes)

    # Feed-forward backbone edges
    for i in range(len(layers) - 1):
        srcs = layers[i]
        dsts = layers[i + 1]
        for u in srcs:
            for v in dsts:
                if rng.random() < density:
                    if (not force_dag) or (u < v):
                        G.add_edge(int(u), int(v))

    # Tangle edges (skip connections and cross-wiring)
    hidden_nodes = [n for layer in layers[1:-1] for n in layer]
    all_nodes = input_nodes + hidden_nodes + output_nodes

    n_tangle = int(max(0, round(total_nodes * density * float(tangle_factor))))
    for _ in range(n_tangle):
        u = int(rng.choice(all_nodes[:-n_wires]))  # don't start at outputs
        v = int(rng.choice(all_nodes[n_wires:]))   # don't end at inputs
        if u == v:
            continue
        if force_dag and u >= v:
            continue
        G.add_edge(u, v)

    return G, input_nodes, output_nodes


# -------------------------
# Transfer map extraction
# -------------------------


def _topo_order(G: nx.MultiDiGraph) -> List[int]:
    D = nx.DiGraph()
    D.add_nodes_from(G.nodes())
    D.add_edges_from((u, v) for (u, v, _k) in G.edges(keys=True))
    return list(nx.topological_sort(D))


def eval_transfer(
    qworld: WorldInstance,
    *,
    topo: List[int],
    input_nodes: Sequence[int],
    output_nodes: Sequence[int],
    input_tuple: Sequence[int],
    internal_init: str = "vacuum",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[int, ...]:
    """Evaluate the quotient world's boundary-to-boundary transfer for one input."""

    if internal_init not in {"vacuum", "random"}:
        raise ValueError("internal_init must be 'vacuum' or 'random'")

    # Initialize internal state
    if internal_init == "vacuum":
        state = {v: 0 for v in qworld.G.nodes()}
    else:
        if rng is None:
            rng = np.random.default_rng(0)
        state = {v: int(rng.integers(0, qworld.alphabet_size[v])) for v in qworld.G.nodes()}

    for i, u in enumerate(input_nodes):
        state[int(u)] = int(input_tuple[i])

    for v in topo:
        if v in input_nodes:
            continue
        inputs = [state[u] for (u, _k) in qworld.in_edges[v]]
        state[v] = int(qworld.local_rule[v](tuple(inputs)))

    return tuple(int(state[u]) for u in output_nodes)


def sample_io_pairs(
    qworld: WorldInstance,
    *,
    topo: List[int],
    input_nodes: Sequence[int],
    output_nodes: Sequence[int],
    max_enumerate: int,
    io_samples: int,
    rng: np.random.Generator,
    internal_init: str = "vacuum",
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Return (X_int, Y_int, enumerated_all_inputs?).

    X_int and Y_int are integer arrays of shape (m, n_wires).
    """

    in_sizes = [int(qworld.alphabet_size[int(u)]) for u in input_nodes]
    total = _product(in_sizes)
    n_wires = len(input_nodes)

    if total <= int(max_enumerate):
        # exact enumeration
        X_list = list(itertools.product(*[range(s) for s in in_sizes]))
        m = len(X_list)
        X = np.asarray(X_list, dtype=np.int64).reshape(m, n_wires)
        Y = np.zeros_like(X)
        for j in range(m):
            Y[j, :] = np.asarray(
                eval_transfer(
                    qworld,
                    topo=topo,
                    input_nodes=input_nodes,
                    output_nodes=output_nodes,
                    input_tuple=X[j, :],
                    internal_init=internal_init,
                    rng=rng,
                ),
                dtype=np.int64,
            )
        return X, Y, True

    # sampled
    m = int(io_samples)
    X = np.zeros((m, n_wires), dtype=np.int64)
    Y = np.zeros((m, n_wires), dtype=np.int64)
    for j in range(m):
        X[j, :] = np.asarray([int(rng.integers(0, s)) for s in in_sizes], dtype=np.int64)
        Y[j, :] = np.asarray(
            eval_transfer(
                qworld,
                topo=topo,
                input_nodes=input_nodes,
                output_nodes=output_nodes,
                input_tuple=X[j, :],
                internal_init=internal_init,
                rng=rng,
            ),
            dtype=np.int64,
        )
    return X, Y, False


# -------------------------
# Symmetry / continuum metrics
# -------------------------


def _embed_roots(x_int: np.ndarray, sizes: Sequence[int]) -> np.ndarray:
    """Roots-of-unity embedding of Z_k: s -> exp(2π i s/k)."""
    x_int = np.asarray(x_int, dtype=np.int64)
    out = np.zeros(x_int.shape, dtype=np.complex128)
    for j, k in enumerate(sizes):
        k = int(k)
        if k <= 0:
            raise ValueError("alphabet size must be positive")
        out[:, j] = np.exp(2j * np.pi * (x_int[:, j] % k) / k)
    return out


@dataclass
class FitMetrics:
    rmse: float
    rel_rmse: float
    unitarity_defect: float
    det_mag: float
    det_phase: float
    min_sv: float
    max_sv: float
    cond_sv: float
    offdiag_ratio: float


def fit_linear_map_roots(
    X_int: np.ndarray,
    Y_int: np.ndarray,
    *,
    in_sizes: Sequence[int],
    out_sizes: Sequence[int],
    rcond: float = 1e-10,
) -> Tuple[np.ndarray, FitMetrics]:
    """Fit complex linear map M s.t. y ≈ M x (column-vector convention)."""

    Xc = _embed_roots(X_int, in_sizes)
    Yc = _embed_roots(Y_int, out_sizes)

    # columns are samples
    Xcol = Xc.T  # (n, m)
    Ycol = Yc.T  # (n, m)

    pinv = np.linalg.pinv(Xcol, rcond=float(rcond))
    M = Ycol @ pinv

    Yhat = M @ Xcol
    resid = Ycol - Yhat
    rmse = float(np.sqrt(np.mean(np.abs(resid) ** 2)))
    yn = float(np.sqrt(np.mean(np.abs(Ycol) ** 2)))
    rel = float(rmse / yn) if yn > 0 else float("nan")

    n = M.shape[0]
    I = np.eye(n, dtype=np.complex128)
    unitarity_defect = float(np.linalg.norm(M.conj().T @ M - I, ord="fro"))
    det = np.linalg.det(M)
    det_mag = float(np.abs(det))
    det_phase = float(np.angle(det))

    svals = np.linalg.svd(M, compute_uv=False)
    min_sv = float(np.min(svals))
    max_sv = float(np.max(svals))
    cond = float(max_sv / min_sv) if min_sv > 0 else float("inf")

    off = M - np.diag(np.diag(M))
    nM = float(np.linalg.norm(M, ord="fro"))
    offdiag_ratio = float(np.linalg.norm(off, ord="fro") / nM) if nM > 0 else float("nan")

    metrics = FitMetrics(
        rmse=rmse,
        rel_rmse=rel,
        unitarity_defect=unitarity_defect,
        det_mag=det_mag,
        det_phase=det_phase,
        min_sv=min_sv,
        max_sv=max_sv,
        cond_sv=cond,
        offdiag_ratio=offdiag_ratio,
    )
    return M, metrics


@dataclass
class PermutationSymmetry:
    group_order: int
    n_generators: int
    mean_error: float
    max_error: float


def symmetry_under_wire_permutations(
    qworld: WorldInstance,
    *,
    topo: List[int],
    input_nodes: Sequence[int],
    output_nodes: Sequence[int],
    X_int: np.ndarray,
    Y_int: np.ndarray,
    enumerated_all_inputs: bool,
    tol: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    internal_init: str = "vacuum",
) -> PermutationSymmetry:
    """Test which wire permutations commute with f_eff.

    If all inputs were enumerated, this is exact.
    Otherwise it is a Monte Carlo estimate using the sampled inputs.
    """
    n_wires = len(input_nodes)
    perms = list(itertools.permutations(range(n_wires)))

    # Optional exact lookup if we enumerated the full input space.
    table: Optional[Dict[Tuple[int, ...], Tuple[int, ...]]] = None
    if enumerated_all_inputs:
        table = {tuple(map(int, X_int[j, :])): tuple(map(int, Y_int[j, :])) for j in range(X_int.shape[0])}

    errors: List[float] = []
    generators: List[Permutation] = []

    for p in perms:
        mism = 0
        m = X_int.shape[0]
        for j in range(m):
            x = tuple(int(v) for v in X_int[j, :])
            y = tuple(int(v) for v in Y_int[j, :])
            x_p = tuple(x[i] for i in p)
            y_p = tuple(y[i] for i in p)

            if table is not None:
                y2 = table.get(x_p)
                if y2 is None or y2 != y_p:
                    mism += 1
            else:
                y2 = eval_transfer(
                    qworld,
                    topo=topo,
                    input_nodes=input_nodes,
                    output_nodes=output_nodes,
                    input_tuple=x_p,
                    internal_init=internal_init,
                    rng=rng,
                )
                if tuple(map(int, y2)) != y_p:
                    mism += 1

        err = float(mism / max(1, m))
        errors.append(err)
        if err <= float(tol):
            generators.append(Permutation(p))

    if not generators:
        return PermutationSymmetry(group_order=1, n_generators=0, mean_error=float(np.mean(errors)), max_error=float(np.max(errors)))

    G = PermutationGroup(generators)
    return PermutationSymmetry(
        group_order=int(G.order()),
        n_generators=len(generators),
        mean_error=float(np.mean(errors)),
        max_error=float(np.max(errors)),
    )


# -------------------------
# Experiment driver
# -------------------------


@dataclass
class SampleResult:
    # configuration
    n_wires: int
    k: int
    depth: int
    density: float
    rule: str
    sample_idx: int
    seed: int

    # graph sizes
    nodes: int
    edges: int

    # quotient summary
    scdc_ok: bool
    in_sizes: str
    out_sizes: str
    enumerated_all_inputs: bool
    io_pairs: int

    # discrete symmetry
    perm_group_order: int
    perm_generators: int
    perm_mean_error: float
    perm_max_error: float

    # continuum proxy
    fit_rmse: float
    fit_rel_rmse: float
    unitarity_defect: float
    det_mag: float
    det_phase: float
    min_sv: float
    max_sv: float
    cond_sv: float
    offdiag_ratio: float

    # misc
    note: str = ""


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="H4 continuum sweep: renormalized knot transfer + symmetry/continuum proxies")

    p.add_argument("--n_wires_list", type=int, nargs="+", default=[2, 3])
    p.add_argument("--k_list", type=int, nargs="+", default=[2, 3, 4, 8, 16])
    p.add_argument("--depth_list", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--density", type=float, default=0.4)
    p.add_argument("--tangle_factor", type=float, default=2.0)
    p.add_argument("--allow_cycles", action="store_true", help="Allow cycles in the micro-knot (not recommended for transfer maps)")

    p.add_argument("--samples_per_cell", type=int, default=50)
    p.add_argument("--io_samples", type=int, default=2048, help="IO samples when enumeration is too large")
    p.add_argument("--max_enumerate", type=int, default=4096, help="Max input-state count for exact enumeration")
    p.add_argument("--perm_tol", type=float, default=0.0, help="Permutation commutation tolerance (0 = exact)")

    p.add_argument("--internal_init", type=str, default="vacuum", choices=["vacuum", "random"], help="Initialize internal pocket state")

    p.add_argument("--rule", type=str, default="random_hash", choices=["random", "random_hash", "threshold", "xor"], help="Local rule family")
    p.add_argument("--vacuum", type=int, default=0)
    p.add_argument("--threshold", type=int, default=2)

    # SCDC controls
    p.add_argument("--scdc_iters", type=int, default=20)
    p.add_argument("--scdc_samples", type=int, default=1024)
    p.add_argument("--scdc_max_enum", type=int, default=1024)
    p.add_argument("--diamond_samples", type=int, default=64)
    p.add_argument("--q_cache_limit", type=int, default=200000)

    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out_dir", type=str, default="results/h4_continuum_sweep")
    p.add_argument("--save_matrices", action="store_true", help="Save fitted matrices M for each successful sample")

    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "n_wires_list": args.n_wires_list,
        "k_list": args.k_list,
        "depth_list": args.depth_list,
        "density": args.density,
        "tangle_factor": args.tangle_factor,
        "force_dag": (not bool(args.allow_cycles)),
        "samples_per_cell": args.samples_per_cell,
        "io_samples": args.io_samples,
        "max_enumerate": args.max_enumerate,
        "perm_tol": args.perm_tol,
        "internal_init": args.internal_init,
        "rule": args.rule,
        "vacuum": args.vacuum,
        "threshold": args.threshold,
        "scdc": {
            "iters": args.scdc_iters,
            "samples": args.scdc_samples,
            "max_enum": args.scdc_max_enum,
            "diamond_samples": args.diamond_samples,
            "q_cache_limit": args.q_cache_limit,
        },
        "seed": args.seed,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    results: List[SampleResult] = []
    matrices_dir = out_dir / "matrices"
    if args.save_matrices:
        matrices_dir.mkdir(parents=True, exist_ok=True)

    base_seed = int(args.seed)

    # Outer sweep
    for n_wires in args.n_wires_list:
        n_wires = int(n_wires)
        for k in args.k_list:
            k = int(k)
            for depth in args.depth_list:
                depth = int(depth)

                print(f"\n=== Cell: n_wires={n_wires} k={k} depth={depth} density={args.density} rule={args.rule} ===")

                for si in range(int(args.samples_per_cell)):
                    seed = base_seed + 10_000 * n_wires + 100 * k + 7 * depth + si
                    rng = np.random.default_rng(seed)

                    # 1) generate knot
                    G, inputs, outputs = generate_random_knot(
                        n_wires=n_wires,
                        depth=depth,
                        density=float(args.density),
                        rng=rng,
                        tangle_factor=float(args.tangle_factor),
                        force_dag=(not bool(args.allow_cycles)),
                    )

                    # 2) build world + SCDC quotient
                    rule_factory = make_rule_factory(
                        str(args.rule),
                        threshold=int(args.threshold),
                        alphabet_k=int(k),
                        vacuum_fixed=int(args.vacuum),
                    )
                    world = WorldInstance.homogeneous(G=G, k=int(k), rule_factory=rule_factory, seed=seed + 123)

                    scdc_cfg = SCDCConfig(
                        seed=int(seed),
                        max_iterations=int(args.scdc_iters),
                        sample_tuples_per_vertex=int(args.scdc_samples),
                        max_enumerate_tuples_per_vertex=int(args.scdc_max_enum),
                        max_diamond_state_samples=int(args.diamond_samples),
                    )

                    note = ""
                    try:
                        profile = compute_lambda_star(world, cfg=scdc_cfg)
                        qworld = quotient_world(world, profile, cache_limit=int(args.q_cache_limit))
                        scdc_ok = True
                    except Exception as e:
                        qworld = world
                        scdc_ok = False
                        note = f"scdc_failed:{type(e).__name__}"

                    topo = _topo_order(qworld.G)

                    # 3) sampled transfer map
                    X_int, Y_int, enumerated = sample_io_pairs(
                        qworld,
                        topo=topo,
                        input_nodes=inputs,
                        output_nodes=outputs,
                        max_enumerate=int(args.max_enumerate),
                        io_samples=int(args.io_samples),
                        rng=rng,
                        internal_init=str(args.internal_init),
                    )

                    in_sizes = [int(qworld.alphabet_size[int(u)]) for u in inputs]
                    out_sizes = [int(qworld.alphabet_size[int(u)]) for u in outputs]

                    # 4) discrete symmetry skeleton
                    perm = symmetry_under_wire_permutations(
                        qworld,
                        topo=topo,
                        input_nodes=inputs,
                        output_nodes=outputs,
                        X_int=X_int,
                        Y_int=Y_int,
                        enumerated_all_inputs=bool(enumerated),
                        tol=float(args.perm_tol),
                        rng=rng,
                        internal_init=str(args.internal_init),
                    )

                    # 5) continuum proxy: complex linear fit
                    try:
                        M, fm = fit_linear_map_roots(
                            X_int,
                            Y_int,
                            in_sizes=in_sizes,
                            out_sizes=out_sizes,
                        )
                        if args.save_matrices:
                            np.save(matrices_dir / f"M_n{n_wires}_k{k}_d{depth}_s{si}_seed{seed}.npy", M)
                    except Exception as e:
                        # Fit can fail if pinv blows up (pathological rank issues)
                        M = np.full((n_wires, n_wires), np.nan, dtype=np.complex128)
                        fm = FitMetrics(
                            rmse=float("nan"),
                            rel_rmse=float("nan"),
                            unitarity_defect=float("nan"),
                            det_mag=float("nan"),
                            det_phase=float("nan"),
                            min_sv=float("nan"),
                            max_sv=float("nan"),
                            cond_sv=float("nan"),
                            offdiag_ratio=float("nan"),
                        )
                        note = (note + ";" if note else "") + f"fit_failed:{type(e).__name__}"

                    results.append(
                        SampleResult(
                            n_wires=n_wires,
                            k=k,
                            depth=depth,
                            density=float(args.density),
                            rule=str(args.rule),
                            sample_idx=int(si),
                            seed=int(seed),
                            nodes=int(G.number_of_nodes()),
                            edges=int(G.number_of_edges()),
                            scdc_ok=bool(scdc_ok),
                            in_sizes=str(in_sizes),
                            out_sizes=str(out_sizes),
                            enumerated_all_inputs=bool(enumerated),
                            io_pairs=int(X_int.shape[0]),
                            perm_group_order=int(perm.group_order),
                            perm_generators=int(perm.n_generators),
                            perm_mean_error=float(perm.mean_error),
                            perm_max_error=float(perm.max_error),
                            fit_rmse=float(fm.rmse),
                            fit_rel_rmse=float(fm.rel_rmse),
                            unitarity_defect=float(fm.unitarity_defect),
                            det_mag=float(fm.det_mag),
                            det_phase=float(fm.det_phase),
                            min_sv=float(fm.min_sv),
                            max_sv=float(fm.max_sv),
                            cond_sv=float(fm.cond_sv),
                            offdiag_ratio=float(fm.offdiag_ratio),
                            note=note,
                        )
                    )

    # Write raw CSV
    raw_path = out_dir / "h4_continuum_sweep_raw.csv"
    _write_csv(raw_path, [asdict(r) for r in results])
    print(f"\nWrote {raw_path}")

    # Aggregate per cell
    # Use simple CSV aggregation (no pandas dependency)
    by_cell: Dict[Tuple[int, int, int, float, str], List[SampleResult]] = {}
    for r in results:
        key = (r.n_wires, r.k, r.depth, r.density, r.rule)
        by_cell.setdefault(key, []).append(r)

    summary_rows: List[Dict[str, object]] = []
    for (n_wires, k, depth, density, rule), rs in sorted(by_cell.items()):
        def _mean(xs: List[float]) -> float:
            xs2 = [x for x in xs if not (math.isnan(x) or math.isinf(x))]
            return float(np.mean(xs2)) if xs2 else float("nan")

        def _std(xs: List[float]) -> float:
            xs2 = [x for x in xs if not (math.isnan(x) or math.isinf(x))]
            return float(np.std(xs2)) if xs2 else float("nan")

        scdc_ok_frac = float(np.mean([1.0 if r.scdc_ok else 0.0 for r in rs])) if rs else float("nan")

        row = {
            "n_wires": n_wires,
            "k": k,
            "depth": depth,
            "density": density,
            "rule": rule,
            "samples": len(rs),
            "scdc_ok_frac": scdc_ok_frac,
            "perm_group_order_mode": int(np.bincount([max(0, int(r.perm_group_order)) for r in rs]).argmax()) if rs else 0,
            "perm_group_order_mean": _mean([float(r.perm_group_order) for r in rs]),
            "unitarity_defect_mean": _mean([r.unitarity_defect for r in rs]),
            "unitarity_defect_std": _std([r.unitarity_defect for r in rs]),
            "fit_rel_rmse_mean": _mean([r.fit_rel_rmse for r in rs]),
            "fit_rel_rmse_std": _std([r.fit_rel_rmse for r in rs]),
            "det_mag_mean": _mean([r.det_mag for r in rs]),
            "offdiag_ratio_mean": _mean([r.offdiag_ratio for r in rs]),
        }
        summary_rows.append(row)

    summary_path = out_dir / "h4_continuum_sweep_summary.csv"
    _write_csv(summary_path, summary_rows)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
