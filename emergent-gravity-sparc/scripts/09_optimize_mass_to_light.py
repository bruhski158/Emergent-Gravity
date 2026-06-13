import os
import glob
import sys
import pandas as pd
import numpy as np
from scipy.optimize import minimize

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.io import load_and_convert_sparc
from src.physics import model_emergent_gravity_full

KM_TO_M = 1.0e3
KPC_TO_M = 3.085677581e19

def objective_function_chi2(ups_params, df, r_m, g_obs, err_log_g_obs):
    ups_disk, ups_bulge = ups_params
    
    v_gas_sq = np.sign(df['Vgas']) * df['Vgas']**2
    v_disk_sq = np.sign(df['Vdisk']) * df['Vdisk']**2
    v_bulge_sq = np.sign(df['Vbul']) * df['Vbul']**2
    v_bar_sq = v_gas_sq + (ups_disk * v_disk_sq) + (ups_bulge * v_bulge_sq)
    
    g_bar_new = (v_bar_sq * (KM_TO_M**2)) / r_m
    
    if np.any(g_bar_new <= 0):
        return 1e6
    
    try:
        g_eg = model_emergent_gravity_full(r_m, g_bar_new)
        delta_eg_log = np.log10(g_obs) - np.log10(g_eg)
        
        valid_mask = ~np.isnan(delta_eg_log)
        fraction_valid = np.sum(valid_mask) / len(g_obs)
        
        # Severe penalty if the model forces >10% of the galaxy into imaginary numbers
        if fraction_valid < 0.9:
            return 1e6
            
        # Calculate chi-squared using the error bars
        chi2 = np.sum((delta_eg_log[valid_mask] / err_log_g_obs[valid_mask])**2)
        return chi2
    except Exception:
        return 1e6

def main():
    print("--- Starting Optimization (With Chi-Squared & Error Bars) ---")
    
    raw_dir = os.path.join(project_root, "data", "raw", "Rotmod_LTG")
    raw_files = glob.glob(os.path.join(raw_dir, "*.dat"))
    
    results = []
    bnds = ((0.1, 1.2), (0.1, 1.2))
    
    for filepath in raw_files:
        galaxy_name = os.path.basename(filepath).split('_')[0]
        
        try:
            df = load_and_convert_sparc(filepath, ups_disk=0.5, ups_bulge=0.7)
            if df.empty or len(df) < 4:
                continue
        except Exception:
            continue
            
        r_m = df['Rad'].values * KPC_TO_M
        g_obs = df['g_obs'].values
        err_log_g_obs = df['err_log_g_obs'].values
        
        initial_guess = [0.5, 0.7]
        
        try:
            res = minimize(objective_function_chi2, initial_guess, 
                           args=(df, r_m, g_obs, err_log_g_obs),
                           bounds=bnds, method='L-BFGS-B')
            
            best_ups_disk, best_ups_bulge = res.x
            
            # Boundary Hugging Check
            hit_bounds = (best_ups_disk <= 0.11 or best_ups_disk >= 1.19 or 
                          best_ups_bulge <= 0.11 or best_ups_bulge >= 1.19)
            
            results.append({
                'Galaxy': galaxy_name,
                'Opt_Ups_Disk': best_ups_disk,
                'Opt_Ups_Bulge': best_ups_bulge,
                'Chi2_Final': res.fun,
                'Hit_Bounds': hit_bounds,
                'Success': res.success
            })
            print(f"Optimized {galaxy_name}: Disk={best_ups_disk:.2f}, Bulge={best_ups_bulge:.2f} | Bounds Hit: {hit_bounds}")
            
        except Exception as e:
            print(f"Failed {galaxy_name}: {e}")

    results_df = pd.DataFrame(results)
    out_path = os.path.join(project_root, "results", "tables", "optimized_ml_ratios_eg.csv")
    results_df.to_csv(out_path, index=False)
    print(f"✅ Optimization complete! Summary saved to {out_path}")

if __name__ == "__main__":
    main()