from __future__ import annotations

"""
H7: Force Law & Coupling Constant Probe.

We place two static 'particles' (knots) at distance r and measure the 
effective attractive force induced by the causal medium.

Physics Goal:
1. Determine the Force Law: F(r) ~ r^(-k). If k ~ 2, we have emergent 3D boson propagators.
2. Measure the Coupling Constant (Alpha): The coefficient F = Alpha / r^2.

Methodology:
- Initialize Universe (Layered + SCDC).
- Inject Knot A at x=0, Knot B at x=r.
- Run simulation for T steps.
- Measure change in distance: Delta_r = r_final - r_initial.
- Acceleration a ~ 2 * Delta_r / T^2.
- Plot a vs r.
"""

import argparse
import numpy as np
import pandas as pd
from typing import List, Tuple

from ..graphs import layered_random_dag, inject_knot
from ..world import WorldInstance, make_rule_factory
from ..scdc import SCDCConfig, compute_lambda_star, quotient_world
from ..schedule import simulate
from ..pockets import compute_essential_inputs, pockets_from_active_set
from ..pocket_activity import active_set_from_state

def get_centroid(pocket: List[int], layer_width: int) -> float:
    # Calculate spatial position (mod layer_width) to track 'horizontal' movement
    # In a layered graph, nodes are 0..N-1.
    # We assume "Space" is the index within the layer.
    positions = [n % layer_width for n in pocket]
    # Handle periodic boundary conditions if necessary? 
    # For now, simple mean.
    return float(np.mean(positions))

def run_force_trial(
    distance: int,
    args
) -> float:
    # 1. Setup Universe
    # To measure 1/r^2, we need a "3D" effective space.
    # Layered graph: Time (1D) + Space (layer_width nodes).
    # If layer_width is 1D array, we are in 1+1D. Force should be constant (F ~ r^0) or linear?
    # Actually, in 1+1D, Coulomb force is linear or constant. 
    # To get 1/r^2, we effectively need 3 spatial dimensions. 
    # But let's see what the graph gives us. 
    
    G = layered_random_dag(
        n_layers=args.n_layers, 
        layer_size=args.layer_width,
        p_forward=args.p_forward,
        p_skip=args.p_skip,
        seed=args.seed
    )
    
    # 2. Inject Two Knots at specific spatial separation
    # Center of space
    mid = args.layer_width // 2
    pos_a = mid - distance // 2
    pos_b = mid + distance // 2
    
    # Define Knot A and Knot B nodes (simple clumps in layer 1)
    knot_nodes_a = list(range(pos_a, pos_a + args.knot_size))
    knot_nodes_b = list(range(pos_b, pos_b + args.knot_size))
    
    # Flatten indices to actual node IDs in layer 1
    # Layer 1 offset = layer_width
    offset = args.layer_width
    knot_a = [n + offset for n in knot_nodes_a]
    knot_b = [n + offset for n in knot_nodes_b]
    
    G = inject_knot(G, knot_a, p_internal=0.9, seed=args.seed+1)
    G = inject_knot(G, knot_b, p_internal=0.9, seed=args.seed+2)
    
    # 3. Stabilize
    rule_factory = make_rule_factory("threshold", threshold=2, alphabet_k=2)
    world = WorldInstance.homogeneous(G, k=2, rule_factory=rule_factory, seed=args.seed)
    
    try:
        cfg = SCDCConfig(max_iterations=15, seed=args.seed)
        profile = compute_lambda_star(world, cfg=cfg)
        qworld = quotient_world(world, profile)
    except:
        return None
        
    # 4. Excite
    x0 = qworld.vacuum_state(0)
    q_map = {v: profile[v].canonical_label_map() for v in G.nodes()}
    for k_nodes in [knot_a, knot_b]:
        for n in k_nodes:
             x0[n] = 1 # Simple excitation

    # 5. Simulate
    # Run long enough to drift, but not collide
    ess = compute_essential_inputs(qworld, cfg=cfg)
    states = simulate(qworld, x0, steps=args.steps, schedule_per_tick=True)
    
    # 6. Measure Drift
    # Get centroids at start and end
    # Note: t=0 might be transient. Use t=5 vs t=End.
    
    def get_pockets_at_t(t):
        if t >= len(states): return []
        act = active_set_from_state(states[t])
        pks = pockets_from_active_set(qworld, act, ess)
        return [p for p in pks if len(p) > 2]
    
    pockets_start = get_pockets_at_t(5)
    pockets_end = get_pockets_at_t(args.steps)
    
    if len(pockets_start) < 2 or len(pockets_end) < 2:
        return None # Particles died or merged too fast
        
    # Assume 2 largest are our particles
    pockets_start.sort(key=len, reverse=True)
    pockets_end.sort(key=len, reverse=True)
    
    pk_a_start = get_centroid(pockets_start[0], args.layer_width)
    pk_b_start = get_centroid(pockets_start[1], args.layer_width)
    dist_start = abs(pk_a_start - pk_b_start)
    
    pk_a_end = get_centroid(pockets_end[0], args.layer_width)
    pk_b_end = get_centroid(pockets_end[1], args.layer_width)
    dist_end = abs(pk_a_end - pk_b_end)
    
    # Drift = (Start Distance) - (End Distance)
    # Positive drift = Attraction
    drift = dist_start - dist_end
    return drift

def main():
    p = argparse.ArgumentParser(description="H7: Force Law Probe")
    p.add_argument("--n_layers", type=int, default=80)
    p.add_argument("--layer_width", type=int, default=100) # Wide space for distance
    p.add_argument("--p_forward", type=float, default=0.12)
    p.add_argument("--p_skip", type=float, default=0.01)
    p.add_argument("--knot_size", type=int, default=5)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--distances", type=int, nargs="+", default=[10, 15, 20, 25, 30, 40])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_csv", type=str, default="h7_force_law.csv")
    args = p.parse_args()
    
    results = []
    print(f"--- Probing Force Law F(r) ---")
    
    for r in args.distances:
        drifts = []
        print(f"Testing distance r={r}...", end="", flush=True)
        for i in range(args.trials):
            args.seed += 1
            d = run_force_trial(r, args)
            if d is not None:
                drifts.append(d)
        
        if drifts:
            mean_drift = np.mean(drifts)
            std_drift = np.std(drifts)
            # Force ~ Acceleration ~ Drift / T^2
            force_proxy = mean_drift
            print(f" Mean Drift: {mean_drift:.4f}")
            results.append({
                "r": r,
                "force": force_proxy,
                "std": std_drift,
                "n": len(drifts)
            })
        else:
            print(" No stable data.")
            
    df = pd.DataFrame(results)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved to {args.out_csv}")

if __name__ == "__main__":
    main()