from __future__ import annotations

"""
H4: Knot Renormalization & Symmetry Hunter (Sweep Mode).

Performs a Renormalization Group (RG) flow analysis on causal knots.
We verify if the "Effective Map" of a knot locks into specific Universality Classes
as we increase the resolution (alphabet k) and complexity (depth).

Targets:
- N=2: Look for 24 (Binary Tetrahedral), 48 (Binary Octahedral).
- N=3: Look for 60 (A5, Icosahedral), 168 (PSL(2,7)), or 1080 (Hessian).
"""

import argparse
import itertools
import csv
import time
from typing import Dict, List, Tuple

import numpy as np
import networkx as nx
from sympy.combinatorics import Permutation, PermutationGroup

from ..world import WorldInstance, make_rule_factory
from ..scdc import SCDCConfig, compute_lambda_star, quotient_world

# -----------------------------------------------------------------------------
# 1. Knot Generator
# -----------------------------------------------------------------------------

def generate_random_knot(
    n_wires: int,
    depth: int,
    density: float,
    rng: np.random.Generator
) -> Tuple[nx.MultiDiGraph, List[int], List[int]]:
    """Generates a random directed feedforward knot with skip connections."""
    G = nx.MultiDiGraph()
    input_nodes = list(range(n_wires))
    
    # Hidden layers
    hidden_start = n_wires
    n_hidden = depth * n_wires
    output_start = hidden_start + n_hidden
    output_nodes = list(range(output_start, output_start + n_wires))
    
    all_nodes = input_nodes + list(range(hidden_start, output_start)) + output_nodes
    G.add_nodes_from(all_nodes)
    
    # 1. Backbone (Feedforward)
    layers = [input_nodes]
    for i in range(depth):
        start = hidden_start + i * n_wires
        layers.append(list(range(start, start + n_wires)))
    layers.append(output_nodes)
    
    for i in range(len(layers) - 1):
        for u in layers[i]:
            for v in layers[i+1]:
                if rng.random() < density:
                    G.add_edge(u, v)

    # 2. Tangle (Skip connections / Cross-talk)
    # Allows non-trivial braiding
    potential_sources = input_nodes + list(range(hidden_start, output_start))
    potential_targets = list(range(hidden_start, output_start)) + output_nodes
    
    n_tangle = int(len(all_nodes) * density * 2)
    for _ in range(n_tangle):
        u = rng.choice(potential_sources)
        v = rng.choice(potential_targets)
        if u < v: # Maintain DAG structure for transfer function validity
            G.add_edge(u, v)
            
    return G, input_nodes, output_nodes

# -----------------------------------------------------------------------------
# 2. Renormalization (Effective Map Extraction)
# -----------------------------------------------------------------------------

def extract_effective_map(
    qworld: WorldInstance, 
    input_nodes: List[int], 
    output_nodes: List[int]
) -> Dict[Tuple[int, ...], Tuple[int, ...]]:
    """Computes f_eff: Inputs -> Outputs for the stabilized quotient knot."""
    in_sizes = [qworld.alphabet_size[u] for u in input_nodes]
    
    # Safety: Caps input space for large k
    total_states = np.prod(in_sizes)
    if total_states > 50_000: 
        # Too expensive to compute full map table for symmetry check
        return None

    input_states = list(itertools.product(*[range(s) for s in in_sizes]))
    mapping = {}
    
    topo_order = list(nx.topological_sort(nx.DiGraph(qworld.G)))
    
    for state_tuple in input_states:
        state = {u: 0 for u in qworld.G.nodes()}
        for i, val in enumerate(state_tuple):
            state[input_nodes[i]] = val
            
        for v in topo_order:
            if v in input_nodes: continue
            inputs = [state[u] for u, _ in qworld.in_edges[v]]
            state[v] = qworld.local_rule[v](tuple(inputs))
            
        mapping[state_tuple] = tuple(state[u] for u in output_nodes)
        
    return mapping

# -----------------------------------------------------------------------------
# 3. Symmetry Fingerprinting
# -----------------------------------------------------------------------------

def detect_symmetry_group_order(
    mapping: Dict[Tuple[int, ...], Tuple[int, ...]], 
    n_wires: int
) -> int:
    """Finds order of the subgroup of S_n commuting with the map."""
    if mapping is None: return -1 # Indicator for "Too Complex"
    
    base_perm = list(range(n_wires))
    valid_perms = []
    
    for p in itertools.permutations(base_perm):
        is_symmetry = True
        for inp, out in mapping.items():
            perm_inp = tuple(inp[i] for i in p)
            res = mapping.get(perm_inp)
            perm_out = tuple(out[i] for i in p)
            if res != perm_out:
                is_symmetry = False
                break
        if is_symmetry:
            valid_perms.append(Permutation(p))
            
    if not valid_perms: return 1
    return int(PermutationGroup(valid_perms).order())

# -----------------------------------------------------------------------------
# 4. Main Sweep Logic
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="H4: Renormalization Group Sweep")
    p.add_argument("--n_wires", type=int, default=3, help="Number of input/output wires")
    p.add_argument("--k_list", type=int, nargs="+", default=[2, 3, 5, 8, 12], help="List of alphabet sizes to sweep")
    p.add_argument("--depth_list", type=int, nargs="+", default=[3, 4, 5], help="List of knot depths to sweep")
    p.add_argument("--samples", type=int, default=50, help="Samples per (k, depth) config")
    p.add_argument("--out_csv", type=str, default="h4_rg_sweep.csv")
    args = p.parse_args()
    
    results = []
    
    print(f"--- Starting RG Sweep (Wires={args.n_wires}) ---")
    print(f"K-values: {args.k_list}")
    print(f"Depths:   {args.depth_list}")
    
    for k in args.k_list:
        for depth in args.depth_list:
            print(f"Running batch: k={k}, depth={depth}...")
            
            rng = np.random.default_rng(42 + k + depth)
            rule_factory = make_rule_factory("random", alphabet_k=k, vacuum_fixed=0)
            # Higher k requires more SCDC iterations to stabilize
            scdc_cfg = SCDCConfig(max_iterations=30 + k, seed=42) 
            
            for i in range(args.samples):
                G, ins, outs = generate_random_knot(args.n_wires, depth, 0.4, rng)
                world = WorldInstance.homogeneous(G, k=k, rule_factory=rule_factory, seed=i)
                
                try:
                    profile = compute_lambda_star(world, cfg=scdc_cfg)
                    qworld = quotient_world(world, profile)
                    eff_map = extract_effective_map(qworld, ins, outs)
                    order = detect_symmetry_group_order(eff_map, args.n_wires)
                    
                    results.append({
                        "k": k,
                        "depth": depth,
                        "id": i,
                        "order": order,
                        "nodes": G.number_of_nodes()
                    })
                except Exception:
                    continue

    # Save
    fieldnames = ["k", "depth", "id", "order", "nodes"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    
    print(f"Sweep complete. Saved to {args.out_csv}")

if __name__ == "__main__":
    main()