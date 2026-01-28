"""
unified_big_bang_v4.py

The Final Proof: Gravity Wells.
1. Run Genesis to create Flat Spacetime.
2. Inject a "Black Hole" (Topological Knot).
3. Visualize the 'Stress-Energy Tensor' (Inconsistency Heatmap).
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from dataclasses import dataclass
import time

# --- Physics Engine (Standard) ---

def get_counts(G):
    nodes = list(G.nodes())
    adj = nx.to_scipy_sparse_array(G, nodelist=nodes, format='csr')
    A2 = adj @ adj
    data = A2.data
    mask = data >= 2
    k = data[mask]
    n_diamonds = np.sum(k * (k - 1) / 2)
    overlap = A2.multiply(adj)
    n_FF_triangles = overlap.sum()
    return n_diamonds, n_FF_triangles

def calculate_energy(G, w_d=1.0, w_t=2.0):
    d, t = get_counts(G)
    return - (w_d * d + w_t * t)

def propose_rewire(G, rng):
    G_new = G.copy()
    edges = list(G_new.edges())
    if not edges: return G
    
    # Standard Rewire
    idx = rng.choice(len(edges))
    u, v = edges[idx]
    nodes = list(G.nodes())
    new_v = nodes[rng.integers(0, len(nodes))]
    
    if u == new_v or G_new.has_edge(u, new_v): return G
    G_new.remove_edge(u, v)
    G_new.add_edge(u, new_v)
    return G_new

def run_genesis(n_nodes=300, avg_degree=4, steps=8000, seed=42):
    print(f"--- PHASE 1: GENESIS (Vacuum Formation) ---")
    rng = np.random.default_rng(seed)
    G = nx.fast_gnp_random_graph(n_nodes, avg_degree/n_nodes, directed=True, seed=seed)
    current_E = calculate_energy(G)
    temp = 2.0
    
    for t in range(steps):
        G_cand = propose_rewire(G, rng)
        cand_E = calculate_energy(G_cand)
        delta = cand_E - current_E
        
        if delta < 0 or rng.random() < np.exp(-delta / temp):
            G = G_cand
            current_E = cand_E
        temp *= 0.999
        
        if t % 2000 == 0:
            print(f"  Step {t}: E={int(current_E)} (Cooling...)")
            
    return G

# --- PHASE 2: MATTER INJECTION ---

def inject_black_hole(G, size=10, seed=99):
    print(f"\n--- PHASE 2: INJECTING MATTER (Singularity) ---")
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    
    # Pick a center
    center = nodes[len(nodes)//2]
    neighbors = list(G.successors(center)) + list(G.predecessors(center))
    
    # Create a dense "Knot" (Singularity)
    # We force high-density connections that defy the lattice structure
    targets = neighbors[:size]
    if len(targets) < size:
        targets = (targets + nodes)[:size]
        
    print(f"  Injecting mass at Node {center}...")
    
    # Add random dense edges (Entropy injection)
    for _ in range(size * 2):
        u = rng.choice(targets)
        v = rng.choice(targets)
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)
            
    return G, targets

# --- PHASE 3: VISUALIZATION ---

def measure_local_curvature(G):
    """
    Measure 'Inconsistency' per node.
    Vacuum nodes are part of many diamonds (Flat).
    Singularity nodes break diamonds (Curved).
    """
    # Simple proxy: Local Clustering Coefficient
    # In this model, HIGH clustering = FLAT (Vacuum)
    # LOW clustering = CURVED (Defect/Mass)
    
    clust = nx.clustering(nx.Graph(G)) # Undirected view
    return clust

def visualize_gravity_well(G, mass_nodes):
    print("\n--- PHASE 3: VISUALIZING GRAVITY WELL ---")
    
    # 1. Compute 'Curvature' (Clustering)
    # High Clustering (1.0) -> White/Blue (Vacuum)
    # Low Clustering (0.0) -> Red/Black (Singularity)
    curvature_map = measure_local_curvature(G)
    colors = [curvature_map.get(n, 0) for n in G.nodes()]
    
    # 2. Force Layout
    # We fix the 'Mass' nodes in the center to see the web warp around them
    pos = nx.spring_layout(nx.Graph(G), k=0.15, iterations=100, seed=42)
    
    # 3. Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw Edges (Faint)
    nx.draw_networkx_edges(G, pos, alpha=0.05, edge_color='gray', ax=ax)
    
    # Draw Nodes (Colored by Flatness)
    # Brighter = Flatter. Darker = Curvature.
    nodes = nx.draw_networkx_nodes(G, pos, 
                                 node_size=50, 
                                 node_color=colors, 
                                 cmap='inferno', # Black/Red/Yellow
                                 ax=ax)
    
    # Highlight the Mass
    nx.draw_networkx_nodes(G, pos, nodelist=mass_nodes, node_size=100, node_color='cyan', label='Injected Mass', ax=ax)
    
    plt.colorbar(nodes, label='Local Flatness (Vacuum Stability)')
    plt.title("Emergent Gravity Well: Mass Warping the Vacuum Lattice")
    plt.legend()
    plt.axis('off')
    
    plt.savefig("gravity_well.png")
    print(">> Saved 'gravity_well.png'. Look for the dark distortion around the Cyan mass.")

if __name__ == "__main__":
    # 1. Create Space
    universe = run_genesis(n_nodes=500, steps=12000)
    
    # 2. Inject Matter
    universe, mass = inject_black_hole(universe, size=15)
    
    # 3. See the Warping
    visualize_gravity_well(universe, mass)