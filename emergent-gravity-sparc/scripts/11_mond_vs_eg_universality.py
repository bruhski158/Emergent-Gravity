import os
import glob
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.calculus import smooth_derivative_log

def calculate_running_median(x, y, bins=20):
    """Calculates the median and 1-sigma (16th to 84th percentile) bands."""
    bin_means, bin_edges, _ = binned_statistic(x, y, statistic='median', bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    bin_p16, _, _ = binned_statistic(x, y, statistic=lambda v: np.percentile(v, 16), bins=bins)
    bin_p84, _, _ = binned_statistic(x, y, statistic=lambda v: np.percentile(v, 84), bins=bins)
    
    return bin_centers, bin_means, bin_p16, bin_p84

def main():
    print("--- Starting Final Showdown: MOND vs Emergent Gravity a_0 Universality ---")
    
    processed_dir = os.path.join(project_root, "data", "processed", "Rotmod_LTG")
    processed_files = glob.glob(os.path.join(processed_dir, "*.csv"))
    
    KPC_TO_M = 3.085677581e19
    A0_EXPECTED = 1.2e-10
    
    # Aggregation arrays
    r_all_eg, a0_all_eg = [], []
    r_all_mond, a0_all_mond = [], []
    
    print(f"Processing {len(processed_files)} galaxies...")
    
    for filepath in processed_files:
        df = pd.read_csv(filepath)
        
        valid_data = df[(df['g_bar'] > 0) & (df['g_obs'] > 0)].copy()
        if valid_data.empty or len(valid_data) < 5:
            continue
            
        r_kpc = valid_data['Rad'].values
        r_m = r_kpc * KPC_TO_M
        g_bar = valid_data['g_bar'].values
        g_obs = valid_data['g_obs'].values
        
        # --- 1. Emergent Gravity Inversion ---
        dgbar_dr = smooth_derivative_log(r_m, g_bar)
        X_term = (1.0 / 6.0) * (3.0 * g_bar + r_m * dgbar_dr)
        
        valid_X = X_term > 0
        if np.any(valid_X):
            a0_eg_local = ((g_obs[valid_X] - g_bar[valid_X])**2) / X_term[valid_X]
            # Filter out extreme numerical outliers for clean plotting
            valid_eg_bounds = (a0_eg_local > 1e-15) & (a0_eg_local < 1e-7)
            
            r_all_eg.extend(r_kpc[valid_X][valid_eg_bounds])
            a0_all_eg.extend(a0_eg_local[valid_eg_bounds])
            
        # --- 2. MOND Inversion ---
        # a0 = [g_obs * (g_obs - g_bar)] / g_bar
        a0_mond_local = (g_obs * (g_obs - g_bar)) / g_bar
        
        # MOND math breaks if g_bar > g_obs (which only happens due to noise in baryon-dominated centers)
        valid_mond = a0_mond_local > 0
        if np.any(valid_mond):
            r_all_mond.extend(r_kpc[valid_mond])
            a0_all_mond.extend(a0_mond_local[valid_mond])

    # Convert to numpy arrays
    r_all_eg, a0_all_eg = np.array(r_all_eg), np.array(a0_all_eg)
    r_all_mond, a0_all_mond = np.array(r_all_mond), np.array(a0_all_mond)
    
    print("Generating side-by-side Universality plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    
    # --- Plotting MOND (Left Panel) ---
    axes[0].scatter(r_all_mond, a0_all_mond, alpha=0.1, s=15, color='blue', edgecolors='none', label='Raw Data Points')
    
    # Calculate and plot MOND running stats
    bins_mond = np.linspace(0, max(r_all_mond), 20)
    r_cen_m, a0_med_m, p16_m, p84_m = calculate_running_median(r_all_mond, a0_all_mond, bins=bins_mond)
    axes[0].plot(r_cen_m, a0_med_m, 'k-', linewidth=3, label='Running Median')
    axes[0].fill_between(r_cen_m, p16_m, p84_m, color='blue', alpha=0.2, label=r'1$\sigma$ Scatter')
    
    axes[0].axhline(A0_EXPECTED, color='red', linestyle='--', linewidth=3, label='Universal $a_0$ Target')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Radius (kpc)', fontsize=14)
    axes[0].set_ylabel(r'Implied Local $a_0 \ (\rm m/s^2)$', fontsize=14)
    axes[0].set_title('MOND Formulation: Implied $a_0(r)$', fontsize=16)
    axes[0].grid(True, which="both", ls="--", alpha=0.4)
    axes[0].legend(loc='lower right')
    
    # --- Plotting Emergent Gravity (Right Panel) ---
    axes[1].scatter(r_all_eg, a0_all_eg, alpha=0.1, s=15, color='purple', edgecolors='none', label='Raw Data Points')
    
    # Calculate and plot EG running stats
    bins_eg = np.linspace(0, max(r_all_eg), 20)
    r_cen_eg, a0_med_eg, p16_eg, p84_eg = calculate_running_median(r_all_eg, a0_all_eg, bins=bins_eg)
    axes[1].plot(r_cen_eg, a0_med_eg, 'k-', linewidth=3, label='Running Median')
    axes[1].fill_between(r_cen_eg, p16_eg, p84_eg, color='purple', alpha=0.2, label=r'1$\sigma$ Scatter')
    
    axes[1].axhline(A0_EXPECTED, color='red', linestyle='--', linewidth=3)
    axes[1].set_xlabel('Radius (kpc)', fontsize=14)
    axes[1].set_title('Emergent Gravity: Implied $a_0(r)$', fontsize=16)
    axes[1].grid(True, which="both", ls="--", alpha=0.4)
    axes[1].legend(loc='lower right')
    
    plt.tight_layout()
    
    save_path = os.path.join(project_root, "results", "figures", "11_mond_vs_eg_universality.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"✅ Final plot saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    main()