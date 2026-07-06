import os
import glob
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.physics import model_mond_simple, model_emergent_gravity_full

def main():
    print("--- Starting Step 4: Calculating Residuals for ALL Galaxies ---")
    
    processed_dir = os.path.join(project_root, "data", "processed", "Rotmod_LTG")
    processed_files = glob.glob(os.path.join(processed_dir, "*.csv"))
    
    if not processed_files:
        print("Error: No processed CSV files found.")
        return
    
    all_radii_kpc_eg, all_radii_kpc_mond = [], []
    residuals_eg, residuals_mond = [], []
    
    total_data_points = 0
    failed_eg_points = 0
    
    KPC_TO_M = 3.085677581e19
    success_count = 0
    
    for filepath in processed_files:
        df = pd.read_csv(filepath)
        valid_data = df[(df['g_bar'] > 0) & (df['g_obs'] > 0)].copy()
        
        if valid_data.empty or len(valid_data) < 4:
            continue
            
        g_bar = valid_data['g_bar'].values
        g_obs = valid_data['g_obs'].values
        r_m = valid_data['Rad'].values * KPC_TO_M
        
        total_data_points += len(g_obs)
        
        try:
            g_mond = model_mond_simple(g_bar)
            g_eg = model_emergent_gravity_full(r_m, g_bar)
            
            delta_mond = np.log10(g_obs) - np.log10(g_mond)
            delta_eg = np.log10(g_obs) - np.log10(g_eg)
            
            valid_mask_eg = ~np.isnan(delta_eg)
            valid_mask_mond = ~np.isnan(delta_mond)
            
            failed_eg_points += np.sum(~valid_mask_eg)
            
            all_radii_kpc_eg.extend(valid_data['Rad'].values[valid_mask_eg])
            residuals_eg.extend(delta_eg[valid_mask_eg])
            
            all_radii_kpc_mond.extend(valid_data['Rad'].values[valid_mask_mond])
            residuals_mond.extend(delta_mond[valid_mask_mond])
            
            success_count += 1
            
        except Exception:
            pass
            
    failure_rate = (failed_eg_points / total_data_points) * 100
    print(f"\n[CRITICAL STATISTIC]: EG formulation yielded unphysical (imaginary) roots for {failure_rate:.1f}% of all SPARC data points.")
    print(f"Successfully aggregated {len(residuals_eg)} surviving data points from {success_count} galaxies.")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    axes[0].scatter(all_radii_kpc_mond, residuals_mond, alpha=0.15, s=10, color='red', edgecolors='none')
    axes[0].axhline(0, color='black', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Radius (kpc)', fontsize=14)
    axes[0].set_ylabel(r'Residual $\Delta = \log_{10}(g_{\rm obs}) - \log_{10}(g_{\rm model})$', fontsize=14)
    axes[0].set_title('MOND Residuals', fontsize=16)
    axes[0].set_ylim(-0.8, 0.8)
    axes[0].grid(True, which="both", ls="--", alpha=0.4)
    
    axes[1].scatter(all_radii_kpc_eg, residuals_eg, alpha=0.15, s=10, color='purple', edgecolors='none')
    axes[1].axhline(0, color='black', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Radius (kpc)', fontsize=14)
    axes[1].set_title(f'Emergent Gravity (Full) Residuals\n(Note: {failure_rate:.1f}% of data dropped due to math failure)', fontsize=14)
    axes[1].grid(True, which="both", ls="--", alpha=0.4)
    
    plt.tight_layout()
    save_path = os.path.join(project_root, "results", "figures", "03_aggregate_residuals.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()

if __name__ == "__main__":
    main()