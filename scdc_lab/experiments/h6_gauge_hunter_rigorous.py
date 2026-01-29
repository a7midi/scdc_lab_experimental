from __future__ import annotations
import argparse
import itertools
import json
import os
import numpy as np
import networkx as nx
from sympy.combinatorics import Permutation, PermutationGroup
from collections import Counter

from scdc_lab.world import WorldInstance, make_rule_factory
from scdc_lab.scdc import SCDCConfig, compute_lambda_star, quotient_world

# --- Generator (Standard) ---
def generate_random_knot(n_wires, depth, density, rng):
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
                if rng.random() < density: G.add_edge(u, v)
    potential_sources = input_nodes + list(range(hidden_start, output_start))
    potential_targets = list(range(hidden_start, output_start)) + output_nodes
    n_tangle = int(len(all_nodes) * density * 2)
    for _ in range(n_tangle):
        u = rng.choice(potential_sources)
        v = rng.choice(potential_targets)
        if u < v: G.add_edge(u, v)
    return G, input_nodes, output_nodes

# --- Map Extraction ---
def get_state_map(qworld, input_nodes, output_nodes):
    in_sizes = [qworld.alphabet_size[u] for u in input_nodes]
    total_states = np.prod(in_sizes)
    # Limit state space to avoid explosion (M=8 is fine, M=16 is limit)
    if total_states > 9: return None 

    input_states = list(itertools.product(*[range(s) for s in in_sizes]))
    state_to_idx = {s: i for i, s in enumerate(input_states)}
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
        if out_tuple in state_to_idx:
            outputs_as_indices.append(state_to_idx[out_tuple])
        else:
            outputs_as_indices.append(-1) # Broken map
            
    return outputs_as_indices

# --- Diagnostics ---
def analyze_map_properties(func_map):
    """Check if the map is degenerate (constant) or trivial."""
    valid_outputs = [y for y in func_map if y != -1]
    if not valid_outputs: return {"valid": False}
    
    counts = Counter(valid_outputs)
    image_size = len(counts)
    is_constant = (image_size == 1)
    is_identity = all(i == y for i, y in enumerate(func_map) if y != -1)
    
    return {
        "valid": True,
        "image_size": image_size,
        "is_constant": is_constant,
        "is_identity": is_identity,
        "entropy": image_size  # Simple proxy
    }

# --- Symmetry Hunter ---
def find_automorphism_group(func_map):
    if not func_map: return None, 0
    M = len(func_map)
    domain = list(range(M))
    valid_perms = []
    
    # Brute force permutations of state space
    # Warning: M=8 -> 40k checks. M=9 -> 360k checks.
    base_perm = list(range(M))
    for p_tuple in itertools.permutations(base_perm):
        is_symmetry = True
        for x in domain:
            y = func_map[x]
            if y == -1: 
                is_symmetry = False; break
            # Commutativity: p(f(x)) == f(p(x))
            if p_tuple[y] != func_map[p_tuple[x]]:
                is_symmetry = False; break
        if is_symmetry:
            valid_perms.append(Permutation(p_tuple))
            
    if not valid_perms: return None, 1
    G = PermutationGroup(valid_perms)
    return G, int(G.order())

# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_wires", type=int, default=3)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--out_dir", type=str, default="h6_candidates")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"--- RIGOROUS GAUGE HUNT (N={args.n_wires}, k={args.k}) ---")
    print(f"Skipping constant maps (Order {(args.k**args.n_wires - 1)}!)")
    
    hits = 0
    
    for i in range(args.samples):
        rng = np.random.default_rng(i) # Deterministic per sample ID
        # Vary density slightly to explore topology space
        density = 0.3 + (rng.random() * 0.4) 
        G, ins, outs = generate_random_knot(args.n_wires, 3, density, rng)
        
        rule_factory = make_rule_factory("random", alphabet_k=args.k, vacuum_fixed=0)
        world = WorldInstance.homogeneous(G, k=args.k, rule_factory=rule_factory, seed=i)
        
        try:
            cfg = SCDCConfig(max_iterations=20, seed=i)
            profile = compute_lambda_star(world, cfg=cfg)
            qworld = quotient_world(world, profile)
            
            # 1. Get Map
            func_map = get_state_map(qworld, ins, outs)
            if not func_map: continue
            
            # 2. Diagnose (FILTER TRIVIAL MAPS)
            diag = analyze_map_properties(func_map)
            if not diag["valid"]: continue
            if diag["is_constant"]: continue # SKIP THE TRASH
            if diag["is_identity"]: continue
            
            # 3. Compute Symmetry
            G_perm, order = find_automorphism_group(func_map)
            
            # 4. Filter for Interesting Orders
            # We want Order > 2. 
            if order > 2:
                print(f"Sample {i}: Order {order} | Image Size {diag['image_size']}")
                
                # 5. SAVE CANDIDATE
                # We save the map and the perm group elements for offline analysis
                candidate_data = {
                    "sample_id": i,
                    "order": order,
                    "map": func_map,
                    "diagnostics": diag,
                    "generators": [p.array_form for p in G_perm.generators]
                }
                with open(f"{args.out_dir}/cand_{i}_ord_{order}.json", "w") as f:
                    json.dump(candidate_data, f)
                hits += 1
                
        except Exception as e:
            continue
            
    print(f"\nScan complete. Found {hits} non-trivial candidates.")

if __name__ == "__main__":
    main()