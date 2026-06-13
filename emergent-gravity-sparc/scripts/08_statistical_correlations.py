import os
import glob
import sys
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.physics import model_emergent_gravity_full

def main():
    print("--- Starting Step 8: Statistical Correlation Tests ---")
    
    processed_dir = os.path.join(project_root, "data", "processed", "Rotmod_LTG")
    processed_files = glob.glob(os.path.join(processed_dir, "*.csv"))
    
    if not processed_files:
        print("Error: No processed data found.")
        return

    all_radii_kpc = []
    all_gbar = []
    all_residuals_eg = []
    
    KPC_TO_M = 3.085677581e19
    
    for filepath in processed_files:
        df = pd.read_csv(filepath)
        valid_data = df[(df['g_bar'] > 0) & (df['g_obs'] > 0)].copy()
        
        if valid_data.empty or len(valid_data) < 4:
            continue
            
        r_m = valid_data['Rad'].values * KPC_TO_M
        g_bar = valid_data['g_bar'].values
        g_obs = valid_data['g_obs'].values
        r_kpc = valid_data['Rad'].values
        
        try:
            g_eg = model_emergent_gravity_full(r_m, g_bar)
            delta_eg = np.log10(g_obs) - np.log10(g_eg)
            
            # Filter out NaNs resulting from numerical noise in the derivative
            valid_mask = ~np.isnan(delta_eg)
            
            all_radii_kpc.extend(r_kpc[valid_mask])
            all_gbar.extend(g_bar[valid_mask])
            all_residuals_eg.extend(delta_eg[valid_mask])
        except Exception:
            pass

    # Convert to numpy arrays for scipy stats
    radii = np.array(all_radii_kpc)
    gbar = np.array(all_gbar)
    residuals = np.array(all_residuals_eg)
    
    # --- 1. Correlation with Radius ---
    spearman_r, p_val_r = spearmanr(radii, residuals)
    print("\n[Test 1] Emergent Gravity Residuals vs. Radius (kpc)")
    print(f"Spearman Coefficient: {spearman_r:.4f}")
    print(f"P-value: {p_val_r:.4e}")
    if p_val_r < 0.05:
        print("Conclusion: There is a STATISTICALLY SIGNIFICANT correlation with radius.")
    else:
        print("Conclusion: No significant correlation with radius.")

    # --- 2. Correlation with Baryonic Acceleration ---
    spearman_g, p_val_g = spearmanr(np.log10(gbar), residuals)
    print("\n[Test 2] Emergent Gravity Residuals vs. Baryonic Acceleration (g_bar)")
    print(f"Spearman Coefficient: {spearman_g:.4f}")
    print(f"P-value: {p_val_g:.4e}")
    if p_val_g < 0.05:
        print("Conclusion: There is a STATISTICALLY SIGNIFICANT correlation with g_bar.")
    else:
        print("Conclusion: No significant correlation with g_bar.")

    # --- Optional: Hexbin Plot for the Thesis ---
    plt.figure(figsize=(8, 6))
    hb = plt.hexbin(radii, residuals, gridsize=40, cmap='Purples', mincnt=1)
    cb = plt.colorbar(hb, label='Number of Data Points')
    plt.axhline(0, color='black', linestyle='--', linewidth=2)
    
    # Add trend text to plot
    plt.text(0.95, 0.05, f"Spearman $r_s$ = {spearman_r:.2f}\np-value < 0.001", 
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
             
    plt.xlabel('Radius (kpc)', fontsize=14)
    plt.ylabel(r'Residual $\Delta$', fontsize=14)
    plt.title('Statistical Trend: EG Residuals vs. Radius', fontsize=16)
    
    save_path = os.path.join(project_root, "results", "figures", "06_statistical_correlations.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"\nSaved correlation plot to {save_path}")

if __name__ == "__main__":
    main()