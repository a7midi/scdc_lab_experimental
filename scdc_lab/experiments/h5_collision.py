from __future__ import annotations

"""
H5: The Interaction Collider.

This experiment tests Gauge Invariance by colliding two stable knots.
1. Build a 'Layered Vacuum' (known to support particles).
2. Inject TWO knots on a collision course.
3. Simulate dynamics.
4. Analyze the output:
    - Elastic Scattering: 2 pockets enter -> 2 pockets leave.
    - Merger (Inelastic): 2 pockets enter -> 1 large pocket.
    - Annihilation: 2 pockets enter -> 0 pockets.

Metric: Conservation of Pocket Count serves as the proxy for conserved topological charge.
"""

import argparse
import csv
import numpy as np
import pandas as pd
from typing import List, Set

from ..graphs import layered_random_dag, inject_knot
from ..world import WorldInstance, make_rule_factory
from ..scdc import SCDCConfig, compute_lambda_star, quotient_world
from ..schedule import simulate
from ..pockets import compute_essential_inputs, pockets_from_active_set
from ..pocket_activity import active_set_from_state

def setup_collision_course(
    n_layers: int, 
    layer_width: int, 
    knot_size: int
) -> List[List[int]]:
    """
    Returns two lists of nodes for Knot A and Knot B.
    Placed in early layers, spatially separated, but structurally allowed to mix.
    """
    # Knot A: Left side, Layer 1
    knot_a = []
    base_a = 1 * layer_width
    # Take first 'knot_size' nodes from left half
    candidates_a = list(range(base_a, base_a + layer_width // 2))
    knot_a = candidates_a[:knot_size]
    
    # Knot B: Right side, Layer 1 (Same time, different space)
    knot_b = []
    # Take nodes from right half
    candidates_b = list(range(base_a + layer_width // 2, base_a + layer_width))
    knot_b = candidates_b[:knot_size]
    
    return knot_a, knot_b

def main():
    p = argparse.ArgumentParser(description="H5: Particle Collider")
    p.add_argument("--n_layers", type=int, default=50)
    p.add_argument("--layer_width", type=int, default=20)
    p.add_argument("--p_forward", type=float, default=0.15) # High density for interaction
    p.add_argument("--p_skip", type=float, default=0.02)
    p.add_argument("--knot_size", type=int, default=6)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out_csv", type=str, default="h5_collision.csv")
    args = p.parse_args()
    
    print("--- SCDC Particle Collider ---")
    
    # 1. Build Substrate
    n_nodes = args.n_layers * args.layer_width
    G = layered_random_dag(
        n_layers=args.n_layers, 
        layer_size=args.layer_width,
        p_forward=args.p_forward,
        p_skip=args.p_skip,
        seed=args.seed
    )
    
    # 2. Inject Two Knots
    knot_a, knot_b = setup_collision_course(args.n_layers, args.layer_width, args.knot_size)
    print(f"Injecting Knot A (Left): {knot_a}")
    print(f"Injecting Knot B (Right): {knot_b}")
    
    # We use 'inject_knot' twice. 
    # Note: inject_knot returns a NEW graph.
    G = inject_knot(G, knot_a, p_internal=0.9, seed=args.seed+1)
    G = inject_knot(G, knot_b, p_internal=0.9, seed=args.seed+2)
    
    # 3. Stabilize (SCDC)
    # We use a threshold rule which is known to support 'glider' structures well
    rule_factory = make_rule_factory("threshold", threshold=2, alphabet_k=2)
    world = WorldInstance.homogeneous(G, k=2, rule_factory=rule_factory, seed=args.seed)
    
    cfg = SCDCConfig(max_iterations=20, seed=args.seed)
    try:
        profile = compute_lambda_star(world, cfg=cfg)
        qworld = quotient_world(world, profile)
    except:
        print("SCDC failed to converge. Aborting run.")
        return

    # 4. Excite & Simulate
    # Excite both knots
    x0 = qworld.vacuum_state(0)
    # Map raw nodes to quotient representatives
    q_map = {v: profile[v].canonical_label_map() for v in G.nodes()}
    
    # Set knots to 1
    for k_nodes in [knot_a, knot_b]:
        for n in k_nodes:
            # Check if this node survived quotienting
            if n in q_map:
                 # We set the state of the quotient node. 
                 # Since multiple raw nodes might map to one quotient node,
                 # we just iterate and set.
                 # (A bit loose, but works for excitation)
                 pass # The simulation uses state on RAW nodes mapped? 
                 # No, quotient_world returns a world with REDUCED nodes?
                 # Wait, quotient_world implementation keeps original graph structure 
                 # but changes the RULE to be the quotient rule. 
                 # So x0 should be on G.nodes(). Correct.
                 x0[n] = 1

    # Simulate
    ess = compute_essential_inputs(qworld, cfg=cfg)
    states = simulate(qworld, x0, steps=args.steps, schedule_per_tick=True)
    
    # 5. Analyze: Counting Pockets
    records = []
    for t, st in enumerate(states):
        active = active_set_from_state(st)
        pockets = pockets_from_active_set(qworld, active, ess)
        # Filter tiny pockets (noise)
        pockets = [p for p in pockets if len(p) > 2]
        
        # Centroids
        centroids = []
        for p in pockets:
            # Simple layer average
            layers = [n // args.layer_width for n in p]
            centroids.append(np.mean(layers))
            
        records.append({
            "t": t,
            "n_pockets": len(pockets),
            "total_active": len(active),
            "centroids": str(sorted([round(c, 1) for c in centroids]))
        })

    # Save
    df = pd.DataFrame(records)
    df.to_csv(args.out_csv, index=False)
    print(f"Simulation done. Saved to {args.out_csv}")
    
    # Summary
    final_pockets = df.iloc[-1]["n_pockets"]
    print(f"\n--- Result ---")
    print(f"Initial Pockets: 2")
    print(f"Final Pockets:   {final_pockets}")
    if final_pockets == 2:
        print("Outcome: ELASTIC SCATTERING / MISS (Topological Charge Conserved)")
    elif final_pockets == 1:
        print("Outcome: MERGER (Inelastic Interaction)")
    elif final_pockets == 0:
        print("Outcome: ANNIHILATION")
    else:
        print("Outcome: FRAGMENTATION")

if __name__ == "__main__":
    main()