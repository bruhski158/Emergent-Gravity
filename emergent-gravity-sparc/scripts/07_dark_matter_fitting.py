import os
import glob
import sys
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.physics import model_nfw_acceleration, model_emergent_gravity_full

def objective_function_chi2_nfw(params, r_m_array, g_bar, g_obs, total_err):
    rho0, rs = params
    g_nfw = model_nfw_acceleration(r_m_array, rho0, rs)
    delta = np.log10(g_obs) - np.log10(g_bar + g_nfw)
    return np.sum((delta / total_err)**2)

def main():
    print("--- Starting Step 7: Batch NFW vs. Emergent Gravity Fitting ---")
    processed_dir = os.path.join(project_root, "data", "processed", "Rotmod_LTG")
    processed_files = glob.glob(os.path.join(processed_dir, "*.csv"))
    
    results = []
    KPC_TO_M = 3.085677581e19
    
    for filepath in processed_files:
        galaxy_name = os.path.basename(filepath).split('_')[0]
        df = pd.read_csv(filepath)
        valid_data = df[(df['g_bar'] > 0) & (df['g_obs'] > 0)].copy()
        
        if valid_data.empty or len(valid_data) < 4:
            continue
            
        r_m = valid_data['Rad'].values * KPC_TO_M
        g_bar = valid_data['g_bar'].values
        g_obs = valid_data['g_obs'].values
        
        err_log_g_obs = valid_data['err_log_g_obs'].values 
        err_log_g_bar = valid_data['err_log_g_bar'].values
        total_err = np.sqrt(err_log_g_obs**2 + err_log_g_bar**2)
        
        try:
            g_eg = model_emergent_gravity_full(r_m, g_bar)
            delta_eg = np.log10(g_obs) - np.log10(g_eg)
            mean_res_eg = np.nanmean(delta_eg)
            std_res_eg = np.nanstd(delta_eg)
        except Exception:
            mean_res_eg, std_res_eg = np.nan, np.nan
            
        bnds = ((1e-5, 0.1), (0.1, 500.0))
        
        try:
            res = differential_evolution(
                objective_function_chi2_nfw, bounds=bnds, 
                args=(r_m, g_bar, g_obs, total_err),
                strategy='best1bin', maxiter=100, tol=0.01, seed=42
            )
            best_rho0, best_rs = res.x
            
            nfw_failed = (best_rs >= 499.9) or (best_rho0 <= 1.1e-5)
            
            g_nfw_fit = model_nfw_acceleration(r_m, best_rho0, best_rs)
            delta_nfw = np.log10(g_obs) - np.log10(g_bar + g_nfw_fit)
            mean_res_nfw = np.nanmean(delta_nfw)
            std_res_nfw = np.nanstd(delta_nfw)
            
        except Exception:
            best_rho0, best_rs, mean_res_nfw, std_res_nfw, nfw_failed = np.nan, np.nan, np.nan, np.nan, True
            
        results.append({
            'Galaxy': galaxy_name,
            'NFW_rho0_fit': best_rho0,
            'NFW_Rs_fit': best_rs,
            'Mean_Residual_NFW': mean_res_nfw,
            'Scatter_NFW': std_res_nfw,
            'NFW_Hit_Bounds': nfw_failed,
            'Mean_Residual_EG': mean_res_eg,
            'Scatter_EG': std_res_eg
        })
        
    results_df = pd.DataFrame(results)
    
    valid_nfw = results_df[~results_df['NFW_Hit_Bounds']]
    print(f"\n[SUMMARY] valid weighted NFW fits: {len(valid_nfw)}/{len(results_df)}")
    
    print(f"Mean NFW Scatter (valid subset): {valid_nfw['Scatter_NFW'].mean():.3f} dex")
    print(f"Mean EG Scatter (valid subset): {valid_nfw['Scatter_EG'].mean():.3f} dex")
    
    out_dir = os.path.join(project_root, "results", "tables")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "nfw_vs_eg_summary.csv")
    results_df.to_csv(out_path, index=False)

if __name__ == "__main__":
    main()