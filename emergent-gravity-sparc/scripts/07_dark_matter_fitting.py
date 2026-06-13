import os
import glob
import sys
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# Standard path resolution for your VS Code architecture
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.physics import model_nfw_acceleration, model_emergent_gravity_full

def objective_function(r_m_array, rho0, rs, g_bar):
    """Objective function for the scipy optimizer."""
    g_nfw = model_nfw_acceleration(r_m_array, rho0, rs)
    return np.log10(g_bar + g_nfw)

def main():
    print("--- Starting Batch NFW vs. Emergent Gravity Fitting ---")
    
    processed_dir = os.path.join(project_root, "data", "processed", "Rotmod_LTG")
    processed_files = glob.glob(os.path.join(processed_dir, "*.csv"))
    
    if not processed_files:
        print("Error: No processed data found.")
        return

    results = []
    KPC_TO_M = 3.085677581e19
    
    for filepath in processed_files:
        filename = os.path.basename(filepath)
        galaxy_name = filename.split('_')[0]
        
        df = pd.read_csv(filepath)
        valid_data = df[(df['g_bar'] > 0) & (df['g_obs'] > 0)].copy()
        
        if valid_data.empty or len(valid_data) < 4:
            continue
            
        r_m = valid_data['Rad'].values * KPC_TO_M
        g_bar = valid_data['g_bar'].values
        g_obs = valid_data['g_obs'].values
        y_target = np.log10(g_obs)
        
        # 1. Evaluate Emergent Gravity (Zero Free Parameters)
        try:
            g_eg = model_emergent_gravity_full(r_m, g_bar)
            delta_eg = np.log10(g_obs) - np.log10(g_eg)
            mean_res_eg = np.nanmean(delta_eg)
            std_res_eg = np.nanstd(delta_eg)
        except Exception:
            mean_res_eg, std_res_eg = np.nan, np.nan
            
        # 2. Fit NFW Profile (Two Free Parameters)
        initial_guess = [0.01, 5.0]
        bounds = ([1e-5, 0.1], [1.0, 100.0]) # Logical bounds for halos
        
        try:
            # We use a lambda to pass g_bar directly into the objective function
            popt, _ = curve_fit(lambda r, rho0, rs: objective_function(r, rho0, rs, g_bar), 
                                r_m, y_target, p0=initial_guess, bounds=bounds)
            best_rho0, best_rs = popt
            
            g_nfw_fit = model_nfw_acceleration(r_m, best_rho0, best_rs)
            g_total_fit = g_bar + g_nfw_fit
            delta_nfw = np.log10(g_obs) - np.log10(g_total_fit)
            mean_res_nfw = np.nanmean(delta_nfw)
            std_res_nfw = np.nanstd(delta_nfw)
            
        except Exception:
            # If optimization fails (e.g., highly irregular galaxy)
            best_rho0, best_rs, mean_res_nfw, std_res_nfw = np.nan, np.nan, np.nan, np.nan
            
        # 3. Store Results
        results.append({
            'Galaxy': galaxy_name,
            'NFW_rho0_fit': best_rho0,
            'NFW_Rs_fit': best_rs,
            'Mean_Residual_NFW': mean_res_nfw,
            'Scatter_NFW': std_res_nfw,
            'Mean_Residual_EG': mean_res_eg,
            'Scatter_EG': std_res_eg
        })
        
        print(f"Processed {galaxy_name}...")
        
    # 4. Export to CSV
    results_df = pd.DataFrame(results)
    out_dir = os.path.join(project_root, "results", "tables")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "nfw_vs_eg_summary.csv")
    results_df.to_csv(out_path, index=False)
    
    print(f"\n✅ Batch processing complete! Summary saved to {out_path}")

if __name__ == "__main__":
    main()