"""
H1 radiation / pulse experiment with finite-speed option.

Why this exists:
- schedule.tick_update in this codebase updates SCCs sequentially, so a signal can
  propagate through many SCC layers within *one tick* (cascading within the sweep).
- For "radiation" we want to test finite-speed propagation: every vertex updates
  from the prior-tick state (global synchronous update).

This script supports:
  --update_mode synchronous   (default; finite-speed)
  --update_mode sequential    (old behavior; SCC-sweep within tick)

It also performs a topology reachability check (ignoring dynamics) so you don't
waste time on runs where the detector layer is unreachable.

CSV columns:
t, active_size, active_centroid_depth, active_max_depth, active_p95_depth,
detector_count, detector_ge_count, active_jaccard_prev
"""

from __future__ import annotations

import argparse
import inspect
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from scdc_lab.graphs import layered_random_dag, inject_knot
from scdc_lab.world import WorldInstance, ThresholdRule, XorRule, random_lookup_rule, LocalRule
from scdc_lab.scdc import compute_lambda_star, quotient_world, SCDCConfig
from scdc_lab.schedule import ScheduleContext, random_topological_order, tick_update


def _layer_of(v: int, layer_size: int) -> int:
    return int(v) // int(layer_size)


def _jaccard(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return float("nan")
    inter = len(a & b)
    uni = len(a | b)
    return float(inter / uni) if uni else float("nan")


def _reachability_summary(G: nx.MultiDiGraph, sources: List[int], layer_size: int, detector_layer: int) -> Tuple[int, bool]:
    """Return (max_reachable_layer, detector_reachable?) ignoring dynamics."""
    D = nx.DiGraph()
    D.add_nodes_from(G.nodes())
    D.add_edges_from((u, v) for u, v, _k in G.edges(keys=True))

    seen: Set[int] = set()
    stack = list(sources)
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        for v in D.successors(u):
            if v not in seen:
                stack.append(v)

    if not seen:
        return -1, False
    max_layer = max(_layer_of(v, layer_size) for v in seen)
    det_ok = any(_layer_of(v, layer_size) >= detector_layer for v in seen)
    return int(max_layer), bool(det_ok)


def _quotient_world_compat(world: WorldInstance, profile: Dict[int, object], **kwargs) -> WorldInstance:
    """Call quotient_world with only kwargs supported by the installed scdc_lab version."""
    try:
        sig = inspect.signature(quotient_world)
        allowed = {}
        for k, v in kwargs.items():
            if k in sig.parameters:
                allowed[k] = v
        return quotient_world(world, profile, **allowed)
    except TypeError:
        # Old signature: quotient_world(world, profile)
        return quotient_world(world, profile)


def _tick_update_synchronous(world: WorldInstance, state_t: Dict[int, int]) -> Dict[int, int]:
    """Global synchronous tick: x_{t+1}(v) depends only on x_t(u)."""
    old = state_t
    new: Dict[int, int] = {}
    for v in world.G.nodes():
        inputs = [old[u] for (u, _key) in world.in_edges[v]]
        new[v] = int(world.local_rule[v](tuple(inputs)))
    return new


def build_layered_world(
    *,
    n: int,
    layers: int,
    p_forward: float,
    p_skip: float,
    knot_k: int,
    knot_layer: int,
    rule: str,
    threshold: int,
    k: int,
    vacuum: int,
    seed: int,
    p_internal: float = 0.9,
    add_back_edges: bool = True,
    use_scdc: bool = True,
    scdc_max_iter: int = 50,
    q_max_precompute: int = 65536,
    q_lazy_cache: int = 200000,
) -> Tuple[WorldInstance, List[int], int, nx.MultiDiGraph]:
    """
    Returns: (world, knot_nodes, layer_size, raw_graph)
    """
    if layers <= 0:
        raise ValueError("--layers must be positive.")
    if n % layers != 0:
        raise ValueError(f"--n must be divisible by --layers for layered graphs. Got n={n}, layers={layers}.")
    layer_size = n // layers
    if knot_layer < 0 or knot_layer >= layers:
        raise ValueError(f"--knot_layer must be in [0, layers-1]. Got {knot_layer} with layers={layers}.")

    G = layered_random_dag(n_layers=layers, layer_size=layer_size, p_forward=p_forward, p_skip=p_skip, seed=seed)

    rng = np.random.default_rng(seed)
    layer_nodes = list(range(knot_layer * layer_size, (knot_layer + 1) * layer_size))
    if knot_k > len(layer_nodes):
        raise ValueError(f"--knot_k too large for a single layer: knot_k={knot_k}, layer_size={layer_size}")
    knot_nodes = rng.choice(layer_nodes, size=int(knot_k), replace=False).tolist()

    G = inject_knot(G, knot_nodes=knot_nodes, p_internal=p_internal, extra_parallel=2, add_back_edges=add_back_edges, seed=seed + 17)

    def rule_factory(_v: int, in_sizes: List[int], rrng: np.random.Generator) -> LocalRule:
        if rule == "xor":
            if int(k) != 2:
                raise ValueError("--rule xor requires --k 2.")
            return XorRule()
        if rule == "threshold":
            if int(k) != 2:
                raise ValueError("--rule threshold requires --k 2.")
            return ThresholdRule(threshold=int(threshold))
        if rule == "random":
            return random_lookup_rule(in_sizes=in_sizes, out_size=int(k), rng=rrng, vacuum_fixed=int(vacuum))
        raise ValueError(f"Unknown rule: {rule}")

    base_world = WorldInstance.homogeneous(G=G, k=int(k), rule_factory=rule_factory, seed=seed + 123)

    if not use_scdc:
        return base_world, knot_nodes, layer_size, G

    cfg = SCDCConfig(max_iterations=int(scdc_max_iter), seed=int(seed))
    profile = compute_lambda_star(base_world, cfg=cfg)

    # These kwargs may or may not exist depending on your scdc_lab version.
    qworld = _quotient_world_compat(
        base_world,
        profile,
        max_precompute_tuples_per_vertex=int(q_max_precompute),
        lazy_cache_limit=int(q_lazy_cache),
    )
    return qworld, knot_nodes, layer_size, G


def excite_state(world: WorldInstance, knot_nodes: List[int], *, mode: str, vacuum: int, seed: int) -> Dict[int, int]:
    x = world.vacuum_state(int(vacuum))
    rng = np.random.default_rng(seed)
    for v in knot_nodes:
        kA = int(world.alphabet_size.get(v, 1))
        if kA <= 1:
            x[v] = int(vacuum)
        else:
            if mode == "ones":
                x[v] = 1
            elif mode == "random":
                x[v] = int(rng.integers(0, kA))
            else:
                raise ValueError(f"Unknown excite mode: {mode}")
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--p_forward", type=float, default=0.10)
    ap.add_argument("--p_skip", type=float, default=0.01)
    ap.add_argument("--knot_k", type=int, default=8)
    ap.add_argument("--knot_layer", type=int, default=2)

    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--vacuum", type=int, default=0)

    ap.add_argument("--rule", choices=["xor", "threshold", "random"], default="xor")
    ap.add_argument("--threshold", type=int, default=2)

    ap.add_argument("--update_mode", choices=["synchronous", "sequential"], default="synchronous")
    ap.add_argument("--fixed_schedule", action="store_true", help="(Sequential mode) Use one fixed SCC schedule for all ticks.")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--detector_layer", type=int, default=30)
    ap.add_argument("--excite", choices=["ones", "random"], default="ones")

    ap.add_argument("--kick_tick", type=int, default=-1)
    ap.add_argument("--kick_frac", type=float, default=0.25)

    ap.add_argument("--no_scdc", action="store_true")
    ap.add_argument("--scdc_max_iter", type=int, default=50)

    # Optional quotient-map resource controls (used only if supported by quotient_world)
    ap.add_argument("--q_max_precompute", type=int, default=65536)
    ap.add_argument("--q_lazy_cache", type=int, default=200000)

    ap.add_argument("--out_csv", type=str, default="h1_radiation.csv")

    args = ap.parse_args()

    world, knot_nodes, layer_size, rawG = build_layered_world(
        n=int(args.n),
        layers=int(args.layers),
        p_forward=float(args.p_forward),
        p_skip=float(args.p_skip),
        knot_k=int(args.knot_k),
        knot_layer=int(args.knot_layer),
        rule=str(args.rule),
        threshold=int(args.threshold),
        k=int(args.k),
        vacuum=int(args.vacuum),
        seed=int(args.seed),
        use_scdc=(not args.no_scdc),
        scdc_max_iter=int(args.scdc_max_iter),
        q_max_precompute=int(args.q_max_precompute),
        q_lazy_cache=int(args.q_lazy_cache),
    )

    print(f"Knot nodes: {knot_nodes}")

    detector_layer = int(args.detector_layer)
    max_reach, det_reach = _reachability_summary(rawG, knot_nodes, layer_size, detector_layer)
    print(f"[Topology Check] Max reachable layer from knot (ignoring dynamics): {max_reach}")
    print(f"[Topology Check] Detector layer {detector_layer} reachable in principle? {det_reach}")

    x = excite_state(world, knot_nodes, mode=str(args.excite), vacuum=int(args.vacuum), seed=int(args.seed) + 999)

    rng = np.random.default_rng(int(args.seed) + 2024)

    # For sequential mode, build schedule context once
    ctx = ScheduleContext.from_world(world)
    fixed_sched = random_topological_order(ctx.condensation.dag, rng) if args.fixed_schedule else None

    records = []
    prev_active: Optional[Set[int]] = None
    max_depth_seen = -1
    first_hit: Optional[int] = None

    for t in range(int(args.steps) + 1):
        active = {v for v, val in x.items() if int(val) != int(args.vacuum)}
        if active:
            depths = np.fromiter((_layer_of(v, layer_size) for v in active), dtype=int)
            active_size = int(depths.size)
            active_centroid = float(depths.mean())
            active_max = int(depths.max())
            active_p95 = float(np.quantile(depths, 0.95))
            det_count = int(np.sum(depths == detector_layer))
            det_ge = int(np.sum(depths >= detector_layer))
        else:
            active_size = 0
            active_centroid = float("nan")
            active_max = -1
            active_p95 = float("nan")
            det_count = 0
            det_ge = 0

        max_depth_seen = max(max_depth_seen, active_max)
        if first_hit is None and active_max >= detector_layer:
            first_hit = t

        aj = float("nan") if prev_active is None else _jaccard(active, prev_active)
        prev_active = set(active)

        records.append(
            dict(
                t=t,
                active_size=active_size,
                active_centroid_depth=active_centroid,
                active_max_depth=active_max,
                active_p95_depth=active_p95,
                detector_count=det_count,
                detector_ge_count=det_ge,
                active_jaccard_prev=aj,
            )
        )

        # optional kick
        if int(args.kick_tick) >= 0 and t == int(args.kick_tick):
            kk = max(1, int(round(float(args.kick_frac) * len(knot_nodes))))
            kick_nodes = rng.choice(knot_nodes, size=kk, replace=False).tolist()
            for v in kick_nodes:
                kA = int(world.alphabet_size.get(v, 1))
                if kA > 1:
                    x[v] = int(rng.integers(0, kA))

        if t < int(args.steps):
            if args.update_mode == "synchronous":
                x = _tick_update_synchronous(world, x)
            else:
                # sequential SCC sweep
                sched = random_topological_order(ctx.condensation.dag, rng) if (not args.fixed_schedule) else fixed_sched
                assert sched is not None
                x = tick_update(world, x, sched, ctx.condensation)

    df = pd.DataFrame.from_records(records)
    df.to_csv(str(args.out_csv), index=False)
    print(f"Wrote {args.out_csv}")

    print("\n=== Radiation Summary ===")
    print(f"Update mode: {args.update_mode}")
    print(f"Max active depth seen: {max_depth_seen}")
    if first_hit is None:
        print(f"Detector layer {detector_layer}: NOT reached.")
    else:
        print(f"Detector layer {detector_layer}: first reached at tick {first_hit}.")

    if args.update_mode == "sequential":
        print("Note: sequential mode can propagate multiple layers within one tick (SCC sweep).")


if __name__ == "__main__":
    main()
