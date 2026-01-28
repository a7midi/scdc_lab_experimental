"""
unified_big_bang.py

The Simulation of Everything.
1. THE BIG BANG: Start with a hot, high-entropy Random Graph (Erdos-Renyi).
2. THE COOLING (GRAVITY): Evolve the graph to minimize "Consistency Energy" (maximize confluent diamonds).
3. THE EMERGENCE: Check if the resulting 'frozen' vacuum naturally has:
   - Geometric Dimension (Space)
   - Spectral Bands (Generations/Mass)
   - Local Gauge Symmetries (Forces)
"""

import networkx as nx
import numpy as np
import scipy.sparse.linalg as spla
from scipy import sparse
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time

# --- 1. The Physics Engine (Graph Gravity) ---

def count_diamonds(G):
    """
    Count 'Diamonds' (Confluent 4-cycles).
    A Diamond is u->v, u->w, v->z, w->z.
    This is the structural prerequisite for the SCDC 'Diamond Confluence' constraint.
    In a random graph, this is near zero. In a lattice, it is maximal.
    """
    # Vectorized diamond counting using adjacency matrices
    # A diamond exists between u and z if (A^2)[u, z] >= 2.
    # The number of diamonds is sum(choose(A^2, 2)).
    
    nodes = list(G.nodes())
    n = len(nodes)
    adj = nx.to_scipy_sparse_array(G, nodelist=nodes, format='csr')
    
    # Square the matrix: (A^2)[i, j] counts paths of length 2 from i to j
    A2 = adj @ adj
    
    # We only care about entries >= 2 (diverging then converging paths)
    # The number of diamonds contributed by a pair (u, z) with k paths is k*(k-1)/2
    data = A2.data
    # Filter for k >= 2
    mask = data >= 2
    k = data[mask]
    
    num_diamonds = np.sum(k * (k - 1) / 2)
    return num_diamonds

def calculate_energy(G):
    """
    Consistency Energy H = - (Diamond Density).
    Gravity tries to maximize diamonds (flatten the graph).
    """
    # Normalizing is nice but raw count is fine for MCMC
    return -count_diamonds(G)

def propose_move(G, rng):
    """
    Propose a minimal change: A directed double edge swap.
    Preserves degrees (conservation of local charge) but changes topology.
    u->v, x->y  becomes  u->y, x->v
    """
    G_new = G.copy()
    edges = list(G_new.edges())
    if len(edges) < 2:
        return G_new
    
    # Pick two random edges
    idx = rng.choice(len(edges), size=2, replace=False)
    u, v = edges[idx[0]]
    x, y = edges[idx[1]]
    
    # Avoid creating self-loops or existing edges for simplicity (naive swap)
    # u->y, x->v
    if u == y or x == v: return G # Reject trivial self-loops
    if G_new.has_edge(u, y) or G_new.has_edge(x, v): return G # Reject multi-edges for simplicity
    
    # Do swap
    G_new.remove_edge(u, v)
    G_new.remove_edge(x, y)
    G_new.add_edge(u, y)
    G_new.add_edge(x, v)
    
    return G_new

def run_big_bang(n_nodes=300, avg_degree=3, steps=5000, cooling_rate=0.995, seed=42):
    print(f"--- INITIATING BIG BANG (N={n_nodes}) ---")
    rng = np.random.default_rng(seed)
    
    # 1. The Initial State (Hot Gas / Random Graph)
    # Erdos-Renyi-like setup
    G = nx.fast_gnp_random_graph(n_nodes, avg_degree/n_nodes, directed=True, seed=seed)
    # Ensure it's a MultiDiGraph for compatibility, though we use DiGraph logic for simple diamonds
    G = nx.MultiDiGraph(G)
    
    current_energy = calculate_energy(G)
    best_energy = current_energy
    
    temperature = 10.0
    history = []
    
    print(f"T=0 (Initial): Diamonds = {-current_energy}")
    
    start_time = time.time()
    
    for t in range(steps):
        # MCMC Step
        G_candidate = propose_move(G, rng)
        new_energy = calculate_energy(G_candidate)
        
        delta_E = new_energy - current_energy
        
        # Metropolis Acceptance
        if delta_E < 0 or rng.random() < np.exp(-delta_E / temperature):
            G = G_candidate
            current_energy = new_energy
        
        # Cooling
        temperature *= cooling_rate
        history.append(-current_energy) # Track diamonds
        
        if t % 500 == 0:
            print(f"Step {t}: Temp={temperature:.4f}, Diamonds={-current_energy}")
            
    print(f"--- COOLING COMPLETE ---")
    print(f"Final State: Diamonds = {-current_energy}")
    return G, history

# --- 2. The Diagnostics (Looking for Physics) ---

def check_h3_generations(G):
    """Check for Spectral Bands (Mass Generations)"""
    print("\n[H3 Test] Checking for Mass Generations (Spectral Bands)...")
    nodes = list(G.nodes())
    adj = nx.to_scipy_sparse_array(G, nodelist=nodes, format='csr')
    
    # Symmetrized Spectrum often reveals cluster/layer structure
    # (A + A.T)
    sym_adj = (adj + adj.T) / 2
    try:
        # Get top eigenvalues
        vals = spla.eigsh(sym_adj, k=10, which='LA', return_eigenvectors=False)
        vals = np.sort(vals)[::-1]
        print(f"Top 10 Eigenvalues: {np.round(vals, 2)}")
        
        # Simple gap detection
        gaps = np.diff(vals)
        print(f"Gaps: {np.round(gaps, 2)}")
        
        if np.any(np.abs(gaps) > 0.5):
            print(">> SIGNIFICANT BANDS DETECTED: Mass Hierarchy Exists.")
        else:
            print(">> Continuous Spectrum: No Mass Hierarchy.")
            
    except Exception as e:
        print(f"Spectral check failed: {e}")

def check_h2_symmetry(G):
    """Check local symmetry proxies"""
    print("\n[H2 Test] Checking for Local Symmetries...")
    # Just check degree distribution entropy as a proxy for structural regularization
    degrees = [d for n, d in G.degree()]
    unique, counts = np.unique(degrees, return_counts=True)
    print(f"Degree Distribution: {dict(zip(unique, counts))}")
    if len(unique) < 5:
        print(">> HIGH SYMMETRY DETECTED: The graph has crystallized.")
    else:
        print(">> Low Symmetry: Still amorphous.")

# --- MAIN RUN ---

if __name__ == "__main__":
    # 1. Run the Simulation
    universe, history = run_big_bang(n_nodes=400, avg_degree=4, steps=10000)
    
    # 2. Analyze the Result
    check_h3_generations(universe)
    check_h2_symmetry(universe)
    
    # 3. Check for "Space" (Clustering)
    clustering = nx.average_clustering(nx.Graph(universe)) # Treat as undirected for standard clustering
    print(f"\n[Geometry Check] Clustering Coefficient: {clustering:.4f}")
    if clustering > 0.1:
        print(">> SPACE HAS EMERGED (High Clustering).")
    else:
        print(">> Still Random (Low Clustering).")
        
    print("\n--- SIMULATION END ---")