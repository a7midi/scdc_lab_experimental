import json
import glob
from sympy.combinatorics import Permutation, PermutationGroup

def analyze_candidate(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    gens = [Permutation(g) for g in data['generators']]
    G = PermutationGroup(gens)
    
    print(f"--- Analyzing {filepath} ---")
    print(f"Order: {G.order()}")
    print(f"Center Size: {G.center().order()}")
    print(f"Solvable: {G.is_solvable}")
    print(f"Abelian: {G.is_abelian}")
    
    # THE TEST:
    # Binary Polyhedral groups (2T, 2O, 2I) have center size 2 (Z2).
    # S5 (Order 120) has center size 1.
    if G.order() == 120 and G.center().order() == 2:
        print(">>> RESULT: BINARY ICOSAHEDRAL GROUP CONFIRMED <<<")
    elif G.order() == 48 and G.center().order() == 2:
        print(">>> RESULT: BINARY OCTAHEDRAL GROUP CONFIRMED <<<")
    elif G.order() == 24 and G.center().order() == 2:
        print(">>> RESULT: BINARY TETRAHEDRAL GROUP CONFIRMED <<<")
    else:
        print("Result: Generic/Other Group")

files = glob.glob("h6_candidates/*.json")
for f in files:
    analyze_candidate(f)