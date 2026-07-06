import os
import glob
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.physics import model_emergent_gravity_full, model_mond_simple

def main():
    print("--- Starting Step 8: Inner vs. Outer Galaxy Split ---")
    processed_dir = os.path.join(project_root, "data", "processed", "Rotmod_LTG")
    processed_files = glob.glob(os.path.join(processed_dir, "*.csv"))
    
    inner_eg, outer_eg = [], []
    inner_mond, outer_mond = [], []
    
    eg_inner_total, eg_inner_fail = 0, 0
    eg_outer_total, eg_outer_fail = 0, 0
    
    KPC_TO_M = 3.085677581e19
    a0 = 1.2e-10 
    
    for filepath in processed_files:
        df = pd.read_csv(filepath)
        valid_data = df[(df['g_bar'] > 0) & (df['g_obs'] > 0)].copy()
        if valid_data.empty: continue
            
        g_bar = valid_data['g_bar'].values
        g_obs = valid_data['g_obs'].values
        r_m = valid_data['Rad'].values * KPC_TO_M
        
        g_eg = model_emergent_gravity_full(r_m, g_bar)
        g_mond = model_mond_simple(g_bar)
        
        d_eg = np.log10(g_obs) - np.log10(g_eg)
        d_mond = np.log10(g_obs) - np.log10(g_mond)
        
        for i in range(len(g_bar)):
            is_inner = (g_bar[i] >= a0)
            
            if is_inner:
                eg_inner_total += 1
                if not np.isnan(d_mond[i]):
                    inner_mond.append(d_mond[i])
                    
                if np.isnan(d_eg[i]): 
                    eg_inner_fail += 1
                else: 
                    inner_eg.append(d_eg[i])
            else:
                eg_outer_total += 1
                if not np.isnan(d_mond[i]):
                    outer_mond.append(d_mond[i])
                    
                if np.isnan(d_eg[i]): 
                    eg_outer_fail += 1
                else: 
                    outer_eg.append(d_eg[i])

    inner_fail_pct = (eg_inner_fail / eg_inner_total) * 100
    outer_fail_pct = (eg_outer_fail / eg_outer_total) * 100
    ks_eg_stat, ks_eg_p = ks_2samp(inner_eg, outer_eg)
    ks_mond_stat, ks_mond_p = ks_2samp(inner_mond, outer_mond)

    print(f"\n[STATISTICS]")
    print(f"EG Inner Failure Rate: {inner_fail_pct:.1f}% | Outer Failure Rate: {outer_fail_pct:.1f}%")
    print(f"EG KS-Test p-value: {ks_eg_p:.4e}")
    print(f"MOND KS-Test p-value: {ks_mond_p:.4e}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    bins = np.linspace(-0.6, 0.6, 50)
    
    axes[0].hist(inner_eg, bins=bins, alpha=0.6, color='blue', edgecolor='black', density=True, label=fr'Inner ($g_{{\rm bar}} > a_0$)')
    axes[0].hist(outer_eg, bins=bins, alpha=0.6, color='orange', edgecolor='black', density=True, label=fr'Outer ($g_{{\rm bar}} < a_0$)')
    axes[0].axvline(0, color='red', linestyle='dashed', linewidth=3)
    axes[0].set_xlabel(r'Residual $\Delta$', fontsize=14)
    axes[0].set_ylabel('Density', fontsize=14)
    axes[0].set_title(f'Emergent Gravity\n(Outer Failure Rate: {outer_fail_pct:.1f}%) | KS p-val: {ks_eg_p:.2e}', fontsize=14)
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    axes[1].hist(inner_mond, bins=bins, alpha=0.6, color='blue', edgecolor='black', density=True)
    axes[1].hist(outer_mond, bins=bins, alpha=0.6, color='orange', edgecolor='black', density=True)
    axes[1].axvline(0, color='red', linestyle='dashed', linewidth=3)
    axes[1].set_xlabel(r'Residual $\Delta$', fontsize=14)
    axes[1].set_title(f'MOND\nKS p-val: {ks_mond_p:.2e}', fontsize=14)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_path = os.path.join(project_root, "results", "figures", "05_inner_vs_outer_hist.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()

if __name__ == "__main__":
    main()