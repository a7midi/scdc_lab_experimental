from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

"""
H8: The Prediction Engine.
Extracts dimensionless physical constants from SCDC simulation data.
"""

def fit_force_law(df):
    """
    Fits F = alpha / r^k to the data.
    Returns alpha (coupling constant) and k (dimension).
    """
    # Filter valid data
    df = df[df['force'] > 0].copy()
    if len(df) < 3:
        return 0.0, 0.0, 0.0
        
    x = df['r'].values
    y = df['force'].values
    
    # Log-Log Fit
    log_x = np.log(x)
    log_y = np.log(y)
    
    # Linear regression on log-log: ln(y) = ln(alpha) - k * ln(x)
    coeffs = np.polyfit(log_x, log_y, 1)
    k = -coeffs[0]
    ln_alpha = coeffs[1]
    alpha = np.exp(ln_alpha)
    
    r_squared = 1 - (np.sum((y - alpha * x**(-k))**2) / np.sum((y - np.mean(y))**2))
    
    return alpha, k, r_squared

def analyze_mass_spectrum(df):
    """
    Analyzes the Group Orders from H6 to predict mass ratios.
    Hypothesis: Mass ~ C / log(Order) or Mass ~ Order (Topology complexity).
    We use Mass ~ Order for the 'geometric' hypothesis.
    """
    # Filter for the 'Exceptional' groups (non-trivial, non-background)
    # Background is usually small (1, 2) or huge (factorial of states).
    # We look for the "Structure" peaks.
    
    counts = df['order'].value_counts()
    orders = sorted(counts.index.tolist())
    
    # Filter typical noise (1, 2) and background (5040 for N=3, k=2)
    # We want the intermediate "gems".
    candidates = [o for o in orders if o > 2 and o != 5040]
    
    return candidates

def main():
    print("--- SCDC PHYSICAL CONSTANTS PREDICTOR ---")
    
    # 1. Calculate Alpha (Fine Structure)
    try:
        df_force = pd.read_csv("h7_force_law.csv")
        alpha, k, r2 = fit_force_law(df_force)
        print(f"\n[ELECTROMAGNETISM]")
        print(f"Force Law Fit:    F = {alpha:.5f} / r^{k:.3f}")
        print(f"Fit Quality (R2): {r2:.4f}")
        print(f">> Fine Structure Constant (alpha): {alpha:.5f}")
        
        if 1.8 < k < 2.2:
            print("   STATUS: Inverse-Square Law Confirmed (Massless Boson)")
        else:
            print("   STATUS: Anomalous Force Law (Screened or Fractal)")
            
    except FileNotFoundError:
        print("\n[ELECTROMAGNETISM] h7_force_law.csv not found.")

    # 2. Calculate Mass Ratios
    try:
        df_gauge = pd.read_csv("h6_gauge_hunt.csv")
        candidates = analyze_mass_spectrum(df_gauge)
        
        print(f"\n[PARTICLE MASSES]")
        print(f"Identified Topological Mass States (Group Orders): {candidates}")
        
        if len(candidates) >= 2:
            # Calculate ratios of adjacent candidates
            print("Predicted Mass Ratios:")
            for i in range(len(candidates)-1):
                m1 = candidates[i+1]
                m2 = candidates[i]
                ratio = m1 / m2
                print(f"   m({m1}) / m({m2}) = {ratio:.4f}")
                
            # Check for Koide-like triplets if we have 3
            if len(candidates) >= 3:
                # Top 3 heaviest
                m_light, m_med, m_heavy = candidates[-3:]
                # Koide Check: (sqrt(m1)+sqrt(m2)+sqrt(m3))^2 / (m1+m2+m3) approx 1.5? (Using masses directly)
                # Or standard Koide uses Mass, let's try using Order as Mass.
                k_num = (np.sqrt(m_light) + np.sqrt(m_med) + np.sqrt(m_heavy))**2
                k_den = m_light + m_med + m_heavy
                k_val = k_num / k_den
                print(f"   Koide Parameter K (Target 0.666): {k_val:.4f}")
                
    except FileNotFoundError:
        print("\n[PARTICLE MASSES] h6_gauge_hunt.csv not found.")

if __name__ == "__main__":
    main()