import os
import glob
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.physics import model_emergent_gravity_full

def main():
    print("--- Starting Step 8: Inner vs. Outer Galaxy Split ---")
    
    processed_dir = os.path.join(project_root, "data", "processed", "Rotmod_LTG")
    processed_files = glob.glob(os.path.join(processed_dir, "*.csv"))
    
    inner_residuals = []
    outer_residuals = []
    
    KPC_TO_M = 3.085677581e19
    a0 = 1.2e-10 
    
    for filepath in processed_files:
        df = pd.read_csv(filepath)
        valid_data = df[(df['g_bar'] > 0) & (df['g_obs'] > 0)].copy()
        
        if valid_data.empty or len(valid_data) < 4:
            continue
            
        g_bar = valid_data['g_bar'].values
        g_obs = valid_data['g_obs'].values
        r_m = valid_data['Rad'].values * KPC_TO_M
        
        try:
            g_eg = model_emergent_gravity_full(r_m, g_bar)
            
            delta_eg = np.log10(g_obs) - np.log10(g_eg)
            
            for i in range(len(delta_eg)):
                if np.isnan(delta_eg[i]):
                    continue
                    
                if g_bar[i] >= a0:
                    inner_residuals.append(delta_eg[i])
                else:
                    outer_residuals.append(delta_eg[i])
                    
        except Exception:
            pass

    print(f"Collected {len(inner_residuals)} Inner data points and {len(outer_residuals)} Outer data points.")
    print("Generating Histogram...")
    
    plt.figure(figsize=(12, 6))
    
    bins = np.linspace(-0.6, 0.6, 50)
    
    plt.hist(inner_residuals, bins=bins, alpha=0.6, color='blue', edgecolor='black', density=True, label='Inner Galaxy ($g_{\\rm bar} > a_0$)')
    
    plt.hist(outer_residuals, bins=bins, alpha=0.6, color='orange', edgecolor='black', density=True, label='Outer Galaxy ($g_{\\rm bar} < a_0$)')
    
    plt.axvline(0, color='red', linestyle='dashed', linewidth=3, label='Perfect Fit ($\Delta = 0$)')
    
    plt.xlabel(r'Residual $\Delta = \log_{10}(g_{\rm obs}) - \log_{10}(g_{\rm EG})$', fontsize=14)
    plt.ylabel('Density of Data Points', fontsize=14)
    plt.title('Emergent Gravity Residuals: Inner vs. Outer Galaxy Regions', fontsize=16)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    save_path = os.path.join(project_root, "results", "figures", "05_inner_vs_outer_hist.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Figure saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    main()