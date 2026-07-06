import os
import glob
import sys
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.io import load_and_convert_sparc
from src.physics import model_emergent_gravity_full

KM_TO_M = 1.0e3
KPC_TO_M = 3.085677581e19

def objective_function_chi2(ups_params, df, r_m, g_obs, err_log_g_obs, err_log_g_bar):
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
        
        if fraction_valid < 0.9:
            return 1e6
            
        chi2 = np.sum((delta_eg_log[valid_mask] / err_log_g_obs[valid_mask])**2)
        
        return chi2
    except Exception:
        return 1e6

def main():
    print("--- Starting Step 9: Optimization (Differential Evolution) ---")
    
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
        err_log_g_bar = df['err_log_g_bar'].values 
        
        try:
            res = differential_evolution(
                objective_function_chi2, bounds=bnds, 
                args=(df, r_m, g_obs, err_log_g_obs, err_log_g_bar),
                strategy='best1bin', maxiter=100, tol=0.01, seed=42
            )
            
            best_ups_disk, best_ups_bulge = res.x
            
            hit_bounds = (best_ups_disk <= 0.11 or best_ups_disk >= 1.19 or 
                          best_ups_bulge <= 0.11 or best_ups_bulge >= 1.19)

            SENTINEL = 1e6
            genuine_fit = res.success and (res.fun < SENTINEL - 1.0)
            if not genuine_fit and res.fun >= SENTINEL - 1.0:
                print(f"  [WARNING] {galaxy_name}: optimizer trapped at sentinel "
                      f"chi2 ({res.fun:.1f}) -- no valid (Y_disk, Y_bulge) found "
                      f"anywhere in bounds [0.1, 1.2]. Hit_Bounds value for this "
                      f"galaxy is not meaningful.")

            results.append({
                'Galaxy': galaxy_name,
                'Opt_Ups_Disk': best_ups_disk,
                'Opt_Ups_Bulge': best_ups_bulge,
                'Chi2_Final': res.fun,
                'Hit_Bounds': hit_bounds,
                'Success': res.success,
                'Genuine_Fit': genuine_fit,
            })
            
        except Exception as e:
            pass

    results_df = pd.DataFrame(results)
    out_path = os.path.join(project_root, "results", "tables", "optimized_ml_ratios_eg.csv")
    results_df.to_csv(out_path, index=False)

    n_total = len(results_df)
    n_genuine = int(results_df['Genuine_Fit'].sum())
    n_sentinel = n_total - n_genuine
    n_hit_bounds_clean = int(results_df.loc[results_df['Genuine_Fit'], 'Hit_Bounds'].sum())

    print(f"\n[SUMMARY]")
    print(f"Total galaxies processed: {n_total}")
    print(f"Genuine fits (chi2 below sentinel): {n_genuine}")
    print(f"Sentinel-trapped failures (no valid Y in bounds): {n_sentinel}")
    print(f"Hit_Bounds among genuine fits: {n_hit_bounds_clean} / {n_genuine} "
          f"({100*n_hit_bounds_clean/n_genuine:.1f}%)")
    print(f"✅ Optimization complete! Summary saved to {out_path}")

if __name__ == "__main__":
    main()