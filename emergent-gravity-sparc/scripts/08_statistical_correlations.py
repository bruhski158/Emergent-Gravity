import os
import glob
import sys
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.physics import model_emergent_gravity_full

def main():
    print("--- Starting Step 8: Statistical Correlation Tests ---")
    processed_dir = os.path.join(project_root, "data", "processed", "Rotmod_LTG")
    processed_files = glob.glob(os.path.join(processed_dir, "*.csv"))
    
    all_radii_norm, all_gbar, all_residuals_eg = [], [], []
    
    total_data_points = 0
    failed_eg_points = 0
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
        
        r_norm = r_kpc / np.max(r_kpc)
        
        total_data_points += len(g_obs)
        
        try:
            g_eg = model_emergent_gravity_full(r_m, g_bar)
            delta_eg = np.log10(g_obs) - np.log10(g_eg)
            
            valid_mask = ~np.isnan(delta_eg)
            failed_eg_points += np.sum(~valid_mask)
            
            all_radii_norm.extend(r_norm[valid_mask])
            all_gbar.extend(g_bar[valid_mask])
            all_residuals_eg.extend(delta_eg[valid_mask])
        except Exception:
            pass

    radii = np.array(all_radii_norm)
    gbar = np.array(all_gbar)
    residuals = np.array(all_residuals_eg)
    
    failure_rate = (failed_eg_points / total_data_points) * 100
    
    spearman_r, p_val_r = spearmanr(radii, residuals)
    print(f"\n[NOTE]: Statistics run on {100-failure_rate:.1f}% of data. ({failure_rate:.1f}% dropped due to EG NaN output).")
    print("\n[Test 1] Emergent Gravity Residuals vs. Normalized Radius (R/R_max)")
    print(f"Spearman Coefficient: {spearman_r:.4f} | P-value: {p_val_r:.4e}")

    spearman_g, p_val_g = spearmanr(np.log10(gbar), residuals)
    print("\n[Test 2] Emergent Gravity Residuals vs. Baryonic Acceleration (g_bar)")
    print(f"Spearman Coefficient: {spearman_g:.4f} | P-value: {p_val_g:.4e}")

    plt.figure(figsize=(8, 6))
    hb = plt.hexbin(radii, residuals, gridsize=40, cmap='Purples', mincnt=1)
    cb = plt.colorbar(hb, label='Number of Data Points')
    plt.axhline(0, color='black', linestyle='--', linewidth=2)
    
    plt.text(0.95, 0.05, f"Spearman $r_s$ = {spearman_r:.2f}\n(Excludes {failure_rate:.1f}% NaN failures)", 
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
             
    plt.xlabel('Normalized Radius ($R/R_{max}$)', fontsize=14)
    plt.ylabel(r'Residual $\Delta$', fontsize=14)
    plt.title('Statistical Trend: EG Residuals vs. Normalized Radius', fontsize=16)
    
    save_path = os.path.join(project_root, "results", "figures", "06_statistical_correlations.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)

if __name__ == "__main__":
    main()