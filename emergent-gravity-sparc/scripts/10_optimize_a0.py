import os
import glob
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.calculus import smooth_derivative_log

def main():
    print("--- Running Analytic Falsification Check for a_0 ---")
    
    processed_dir = os.path.join(project_root, "data", "processed", "Rotmod_LTG")
    processed_files = glob.glob(os.path.join(processed_dir, "*.csv"))
    
    KPC_TO_M = 3.085677581e19
    A0_EXPECTED = 1.2e-10
    
    plt.figure(figsize=(10, 6))
    
    plotted_count = 0
    # Let's plot 10 representative galaxies to avoid a cluttered plot
    for filepath in processed_files[:10]:
        galaxy_name = os.path.basename(filepath).split('_')[0]
        df = pd.read_csv(filepath)
        
        valid_data = df[(df['g_bar'] > 0) & (df['g_obs'] > 0)].copy()
        if valid_data.empty or len(valid_data) < 5:
            continue
            
        r_kpc = valid_data['Rad'].values
        r_m = r_kpc * KPC_TO_M
        g_bar = valid_data['g_bar'].values
        g_obs = valid_data['g_obs'].values
        
        # Calculate the derivative
        dgbar_dr = smooth_derivative_log(r_m, g_bar)
        
        # The X term: (1/6) * (3g_bar + r*dg_bar/dr)
        X_term = (1.0 / 6.0) * (3.0 * g_bar + r_m * dgbar_dr)
        
        # Mask out negative X terms (mathematical failures)
        valid_X = X_term > 0
        
        if not np.any(valid_X):
            continue
            
        # Algebraically extract local a_0(r)
        # g_obs = g_bar + sqrt(a0 * X) --> a0 = (g_obs - g_bar)^2 / X
        a0_local = ((g_obs[valid_X] - g_bar[valid_X])**2) / X_term[valid_X]
        
        plt.plot(r_kpc[valid_X], a0_local, 'o-', alpha=0.6, label=galaxy_name)
        plotted_count += 1
        
    plt.axhline(A0_EXPECTED, color='red', linestyle='--', linewidth=3, label='Theoretical Universal $a_0$')
    
    plt.yscale('log')
    plt.xlabel('Radius (kpc)', fontsize=14)
    plt.ylabel(r'Implied Local $a_0 \ (\rm m/s^2)$', fontsize=14)
    plt.title(r'Universality Falsification: Implied $a_0(r)$ across Radii', fontsize=16)
    plt.legend(loc='best', fontsize=10, ncol=2)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    
    plt.tight_layout()
    
    save_path = os.path.join(project_root, "results", "figures", "10_a0_radial_diagnostic.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Diagnostic plot saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()