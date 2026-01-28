"""
scdc_lab/experiments/unified_consistency_universe2.py
(Final Fix: Direct World Instantiation + Local Physics)
"""
from __future__ import annotations

import argparse
import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

# --- Imports ---
# We import ThresholdRule explicitly to avoid factory errors
from ..graphs import er_directed_multigraph, layered_random_dag, inject_knot, compute_condensation
from ..world import WorldInstance, ThresholdRule 
from ..scdc import SCDCConfig
from ..schedule import simulate
from ..pockets import compute_essential_inputs, active_set_from_state, pockets_from_active_set
from ..geometry import (
    measure_light_cone_growth,
    embed_condensation_mds,
)

# --- 1. LOCAL PHYSICS ENGINE (Self-Contained) ---

def get_counts(G):
    """Count Diamonds (Time) and FF-Triangles (Space)."""
    nodes = list(G.nodes())
    adj = nx.to_scipy_sparse_array(G, nodelist=nodes, format='csr')
    A2 = adj @ adj
    data = A2.data
    # Diamonds: pairs with >=2 paths
    mask = data >= 2
    k = data[mask]
    n_diamonds = np.sum(k * (k - 1) / 2)
    # Triangles: overlap between 1-paths (adj) and 2-paths (A2)
    overlap = A2.multiply(adj)
    n_FF_triangles = overlap.sum()
    return n_diamonds, n_FF_triangles

def calculate_energy(G, w_d=1.0, w_t=2.0):
    d, t = get_counts(G)
    return - (w_d * d + w_t * t)

def propose_rewire(G, rng):
    G_new = G.copy()
    if G_new.number_of_edges() == 0: return G
    edges = list(G_new.edges())
    idx = rng.choice(len(edges))
    u, v = edges[idx]
    nodes = list(G.nodes())
    new_v = nodes[rng.integers(0, len(nodes))]
    
    # Simple rejection criteria
    if u == new_v or G_new.has_edge(u, new_v): return G
    
    G_new.remove_edge(u, v)
    G_new.add_edge(u, new_v)
    return G_new

def run_genesis_local(G, steps, seed):
    print(f"--- GENESIS (Local Motif) START ---")
    rng = np.random.default_rng(seed)
    current_E = calculate_energy(G)
    temp = 2.0
    cooling = 0.999
    
    history = []
    
    for t in range(steps):
        G_cand = propose_rewire(G, rng)
        cand_E = calculate_energy(G_cand)
        delta = cand_E - current_E
        
        if delta < 0 or rng.random() < np.exp(-delta / temp):
            G = G_cand
            current_E = cand_E
            
        temp *= cooling
        history.append(current_E)
        
        if t % 1000 == 0:
            print(f"  Step {t:5d}: T={temp:.4f}  E={current_E:.1f}")
            
    print("--- GENESIS COMPLETE ---")
    return G, history

# --- 2. ROBUST VISUALIZATION & LENSING ---

def plot_spacetime_embedding(G, out_png):
    """Robust 2D embedding that handles 1D collapse."""
    try:
        cond = compute_condensation(G)
    except:
        return

    # Fallback for trivial graphs
    if cond.dag.number_of_nodes() < 3:
        plt.figure(figsize=(8, 8))
        pos = nx.spring_layout(nx.Graph(G), seed=42)
        nx.draw(G, pos, node_size=10, alpha=0.3, width=0.1)
        plt.title("Spacetime (Single Block)")
        plt.savefig(out_png)
        plt.close()
        return

    X = embed_condensation_mds(G, n_components=2)
    if X is None: return

    # FIX: Pad if 1D result
    if len(X.shape) == 2 and X.shape[1] == 1:
        X = np.hstack([X, np.zeros((X.shape[0], 1))])

    depth_map = cond.depth_map
    depths = [depth_map.get(i, 0) for i in range(len(X))]

    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1], c=depths, cmap="plasma", s=30, alpha=0.8)
    plt.colorbar(label="Causal Depth")
    plt.title("Spacetime Embedding")
    plt.axis('off')
    plt.savefig(out_png)
    plt.close()

def plot_geodesic_lensing(G, matter_scc, out_png, n_pairs=50, seed=42):
    """
    Visualizes geodesics with robust handling for single-SCC universes.
    """
    rng = np.random.default_rng(seed)
    
    try:
        cond = compute_condensation(G)
        dag = cond.dag
    except:
        return -1.0

    # DAG sources/sinks
    sources = [n for n in dag.nodes() if dag.in_degree(n) == 0]
    targets = [n for n in dag.nodes() if dag.out_degree(n) == 0]
    candidates = list(dag.nodes())
    
    use_dag = True
    # If DAG is trivial (cyclic universe), use raw nodes
    if len(candidates) < 2:
        use_dag = False
        candidates = list(G.nodes())

    plt.figure(figsize=(10, 10))
    
    # Layout
    if use_dag:
        pos = nx.spring_layout(dag, seed=seed)
        nx.draw_networkx_edges(dag, pos, alpha=0.1, edge_color='gray', arrows=False)
        nx.draw_networkx_nodes(dag, pos, node_size=20, node_color='blue', alpha=0.3)
        if matter_scc:
            valid = [m for m in matter_scc if m in dag.nodes()]
            nx.draw_networkx_nodes(dag, pos, nodelist=valid, node_size=100, node_color='cyan')
    else:
        # Raw graph layout
        pos = nx.spring_layout(nx.Graph(G), seed=seed)
        nx.draw_networkx_edges(G, pos, alpha=0.05, edge_color='gray')

    lensing_dists = []
    
    for _ in range(n_pairs):
        # Robust Sampling
        try:
            if use_dag and sources and targets:
                s = rng.choice(sources)
                t = rng.choice(targets)
            else:
                if len(candidates) >= 2:
                    s, t = rng.choice(candidates, size=2, replace=False)
                else:
                    break
                    
            if use_dag:
                path = nx.shortest_path(dag, s, t)
                # Draw
                edges = list(zip(path, path[1:]))
                nx.draw_networkx_edges(dag, pos, edgelist=edges, alpha=0.3, edge_color='orange')
                # Lensing
                if matter_scc:
                    if set(path).intersection(matter_scc):
                        lensing_dists.append(0.0)
                    else:
                        lensing_dists.append(1.0)
            else:
                path = nx.shortest_path(G, s, t)
                lensing_dists.append(1.0) # Placeholder for single-block lensing
                
        except (nx.NetworkXNoPath, ValueError):
            pass

    plt.title("Geodesic Flow")
    plt.savefig(out_png)
    plt.close()
    
    return np.mean(lensing_dists) if lensing_dists else -1.0

# --- 3. MAIN PIPELINE ---

def run_pipeline(args):
    print(f"Initial graph: N={args.n} p={args.p}")
    G = er_directed_multigraph(args.n, args.p, seed=args.seed)
    
    # 1. GENESIS (Using Local Engine)
    if args.genesis_steps > 0:
        G, _ = run_genesis_local(G, args.genesis_steps, args.seed)
        print(f"Final graph: N={G.number_of_nodes()} E={G.number_of_edges()}")

    # 2. MATTER
    if args.inject_knot_k > 0:
        nodes = list(G.nodes())
        rng = np.random.default_rng(args.seed + 1)
        knot_nodes = list(rng.choice(nodes, size=args.inject_knot_k, replace=False))
        inject_knot(G, knot_nodes, p_internal=0.9, seed=args.seed)
        print(f"Injected knot: |K|={args.inject_knot_k}")
    else:
        knot_nodes = []

    # 3. DYNAMICS
    # FIX: Define rule locally and use constructor directly
    def make_rule(v, in_sizes, rng):
        return ThresholdRule(threshold=1)

    # Use constructor instead of from_graph to be safe
    world = WorldInstance(G, 2, make_rule, seed=args.seed)
    
    state = world.vacuum_state(0)
    for n in knot_nodes: state[n] = 1
    
    trajectory = simulate(world, state, steps=args.steps, seed=args.seed)
    
    # 4. ANALYSIS
    ess = compute_essential_inputs(world, SCDCConfig(samples=512))
    
    pocket_stats = []
    final_active = set()
    for t, st in enumerate(trajectory):
        active = active_set_from_state(st)
        if t == len(trajectory) - 1: final_active = active
        pockets = pockets_from_active_set(world, active, ess)
        largest = max([len(p) for p in pockets]) if pockets else 0
        pocket_stats.append({"t": t, "active": len(active), "pocket": largest})
        
    found_pocket = False
    if pocket_stats:
        last = pocket_stats[-1]
        if 0 < last["pocket"] < (args.n * 0.9):
            found_pocket = True

    matter_scc = set()
    if found_pocket:
        cond = compute_condensation(G)
        for n in final_active:
            sid = cond.node_to_scc.get(n)
            if sid is not None: matter_scc.add(sid)
                
    lc = measure_light_cone_growth(G, num_sources=5, max_t=8)
    
    # 5. VISUALIZATION
    plot_spacetime_embedding(G, f"{args.out_prefix}_spacetime.png")
    lens_score = plot_geodesic_lensing(G, matter_scc, f"{args.out_prefix}_geodesics.png", n_pairs=args.n_pairs)

    return {
        "light_cones": [asdict(r) for r in lc],
        "pockets": {"found": found_pocket, "trajectory": pocket_stats},
        "lensing": lens_score
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_type", default="er")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--p", type=float, default=0.06)
    parser.add_argument("--genesis_steps", type=int, default=5000)
    parser.add_argument("--energy_mode", default="motif") 
    parser.add_argument("--inject_knot_k", type=int, default=15)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--out_prefix", default="golden_universe")
    parser.add_argument("--out_json", default="golden_universe.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_pairs", type=int, default=25)
    
    args = parser.parse_args()
    summary = run_pipeline(args)
    
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f">> SUCCESS. Outputs: {args.out_prefix}_geodesics.png")
    if summary["pockets"]["found"]:
        print(">> Stable Matter Pocket Detected!")
    print(f"Lensing Score: {summary['lensing']}")

if __name__ == "__main__":
    main()