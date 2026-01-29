from __future__ import annotations

"""
H6: Gauge Hunter (State-Space Symmetries).

This script removes the 'Wire Permutation' restriction and hunts for 
symmetries acting on the full 'State Space' of the knot.

Physics Goal:
Recover finite subgroups of SU(2) and SU(3) that were previously hidden 
because they involve mixing states, not just swapping wires.

Target Groups:
- Order 8 (Quaternion Group Q8)
- Order 12 (Alternating A4 / T)
- Order 24 (Binary Tetrahedral / SL(2,3))
- Order 48 (Binary Octahedral)
- Order 60 (A5)
"""

import argparse
import itertools
import csv
import math
import time
from typing import Dict, List, Tuple, Set

import numpy as np
import networkx as nx
from sympy.combinatorics import Permutation, PermutationGroup

from ..world import WorldInstance, make_rule_factory
from ..scdc import SCDCConfig, compute_lambda_star, quotient_world

# -----------------------------------------------------------------------------
# 1. Knot Generator (Same as before)
# -----------------------------------------------------------------------------

def generate_random_knot(
    n_wires: int,
    depth: int,
    density: float,
    rng: np.random.Generator
) -> Tuple[nx.MultiDiGraph, List[int], List[int]]:
    G = nx.MultiDiGraph()
    input_nodes = list(range(n_wires))
    
    hidden_start = n_wires
    n_hidden = depth * n_wires
    output_start = hidden_start + n_hidden
    output_nodes = list(range(output_start, output_start + n_wires))
    
    all_nodes = input_nodes + list(range(hidden_start, output_start)) + output_nodes
    G.add_nodes_from(all_nodes)
    
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

    n_tangle = int(len(all_nodes) * density * 2)
    potential_sources = input_nodes + list(range(hidden_start, output_start))
    potential_targets = list(range(hidden_start, output_start)) + output_nodes
    
    for _ in range(n_tangle):
        u = rng.choice(potential_sources)
        v = rng.choice(potential_targets)
        if u < v: 
            G.add_edge(u, v)
            
    return G, input_nodes, output_nodes

# -----------------------------------------------------------------------------
# 2. Renormalization (State-Space Map)
# -----------------------------------------------------------------------------

def get_state_map(
    qworld: WorldInstance, 
    input_nodes: List[int], 
    output_nodes: List[int]
) -> Tuple[List[int], List[int]]:
    """
    Returns the transfer function as a list of outputs indices.
    Index i corresponds to the i-th lexicographic input state.
    """
    in_sizes = [qworld.alphabet_size[u] for u in input_nodes]
    input_states = list(itertools.product(*[range(s) for s in in_sizes]))
    
    # Map each tuple state to an integer index
    state_to_idx = {s: i for i, s in enumerate(input_states)}
    
    # We assume output space size == input space size for automorphism check
    # (Square matrix condition for defining a symmetry group on the domain)
    # If sizes differ, symmetry is less well-defined (only kernel symmetries).
    # We will strictly look for Automorphisms where Domain == Codomain.
    
    topo_order = list(nx.topological_sort(nx.DiGraph(qworld.G)))
    
    outputs_as_indices = []
    
    for state_tuple in input_states:
        state = {u: 0 for u in qworld.G.nodes()}
        for i, val in enumerate(state_tuple):
            state[input_nodes[i]] = val
            
        for v in topo_order:
            if v in input_nodes: continue
            inputs = [state[u] for u, _ in qworld.in_edges[v]]
            state[v] = qworld.local_rule[v](tuple(inputs))
            
        out_tuple = tuple(state[u] for u in output_nodes)
        
        # If output is not in the valid input state set (e.g. different k), 
        # we map to -1 (symmetry broken).
        if out_tuple in state_to_idx:
            outputs_as_indices.append(state_to_idx[out_tuple])
        else:
            outputs_as_indices.append(-1)
            
    return outputs_as_indices

# -----------------------------------------------------------------------------
# 3. Automorphism Hunter (Brute Force / Heuristic)
# -----------------------------------------------------------------------------

def find_automorphism_group_order(func_map: List[int]) -> int:
    """
    Finds the order of the group G of permutations p such that:
    p(func_map(x)) = func_map(p(x))
    
    This is the "Centralizer" of the function in the Symmetric Group S_M.
    M = len(func_map).
    """
    M = len(func_map)
    domain = list(range(M))
    
    # Optimization: 
    # If the map is the Identity, Order is M! (Too big).
    # If the map is Constant, Order is M! (if we map output to same).
    # We cap the search for M > 8 because M! explodes.
    
    if M > 8:
        return -1 # Too expensive to brute force S_M
        
    valid_perms = []
    base_perm = list(range(M))
    
    # Brute force all permutations of the STATES
    # For M=4 (N=2, k=2), 4! = 24 checks. Trivial.
    # For M=8 (N=3, k=2), 8! = 40,320 checks. ~1 second.
    # For M=9 (N=2, k=3), 9! = 362,880 checks. ~5 seconds.
    
    for p_tuple in itertools.permutations(base_perm):
        # p_tuple[i] is where i goes. 
        # So p(x) is p_tuple[x].
        
        is_symmetry = True
        for x in domain:
            y = func_map[x]
            if y == -1: # Map broken
                is_symmetry = False
                break
                
            # Check p(f(x)) == f(p(x))
            # LHS: p(y) = p_tuple[y]
            # RHS: f(p_tuple[x]) = func_map[p_tuple[x]]
            
            lhs = p_tuple[y]
            rhs = func_map[p_tuple[x]]
            
            if lhs != rhs:
                is_symmetry = False
                break
        
        if is_symmetry:
            valid_perms.append(Permutation(p_tuple))
            
    if not valid_perms:
        return 1
        
    return int(PermutationGroup(valid_perms).order())

# -----------------------------------------------------------------------------
# 4. Main Experiment
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="H6: State-Space Gauge Hunter")
    p.add_argument("--n_wires", type=int, default=2)
    p.add_argument("--k", type=int, default=3, help="Alphabet size. N=2, k=3 -> 9 states.")
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--density", type=float, default=0.5)
    p.add_argument("--out_csv", type=str, default="h6_gauge_hunt.csv")
    args = p.parse_args()
    
    results = []
    state_space_size = args.k ** args.n_wires
    
    print(f"--- Hunting for Gauge Groups ---")
    print(f"Wires: {args.n_wires}, k: {args.k} => State Space Size: {state_space_size}")
    if state_space_size > 8:
        print("Warning: State space > 8, brute force might be slow per sample.")
    
    rng = np.random.default_rng(42)
    
    for i in range(args.samples):
        # Generate
        G, inputs, outputs = generate_random_knot(args.n_wires, args.depth, args.density, rng)
        
        # SCDC
        rule_factory = make_rule_factory("random", alphabet_k=args.k, vacuum_fixed=0)
        world = WorldInstance.homogeneous(G, k=args.k, rule_factory=rule_factory, seed=i)
        
        try:
            # Low iterations since we want to catch structure before it becomes trivial
            cfg = SCDCConfig(max_iterations=15, seed=i)
            profile = compute_lambda_star(world, cfg=cfg)
            qworld = quotient_world(world, profile)
            
            # Extract Map
            func_map = get_state_map(qworld, inputs, outputs)
            
            # Compute Symmetry
            order = find_automorphism_group_order(func_map)
            
            results.append({
                "id": i,
                "order": order,
                "map_size": len(func_map)
            })
            
            if order > 2:
                print(f"Sample {i}: Found Group Order {order}")
                
        except Exception as e:
            continue
            
    # Save
    fieldnames = ["id", "order", "map_size"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
        
    print(f"Saved to {args.out_csv}")
    
    # Histogram
    orders = [r["order"] for r in results]
    unique, counts = np.unique(orders, return_counts=True)
    print("\n--- Group Orders Found ---")
    for o, c in zip(unique, counts):
        print(f"Order {o}: {c} hits")

if __name__ == "__main__":
    main()