import numpy as np
import pandas as pd
import os

KM_TO_M = 1.0e3
KPC_TO_M = 3.085677581e19

def load_and_convert_sparc(filepath, ups_disk=0.5, ups_bulge=0.7):
    col_names = ['Rad', 'Vobs', 'errV', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul']
    df = pd.read_csv(filepath, sep=r'\s+', comment='#', names=col_names)
    df = df[df['Rad'] > 0].copy()
    
    v_gas_sq = np.sign(df['Vgas']) * df['Vgas']**2
    v_disk_sq = np.sign(df['Vdisk']) * df['Vdisk']**2
    v_bulge_sq = np.sign(df['Vbul']) * df['Vbul']**2
    
    v_bar_sq = v_gas_sq + (ups_disk * v_disk_sq) + (ups_bulge * v_bulge_sq)
    
    radius_m = df['Rad'] * KPC_TO_M
    v_obs_m_s = df['Vobs'] * KM_TO_M
    err_v_m_s = df['errV'] * KM_TO_M
    
    df['g_obs'] = (v_obs_m_s**2) / radius_m
    df['g_bar'] = (v_bar_sq * (KM_TO_M**2)) / radius_m
    
    # NEW: Error propagation
    # sigma_g = (2 * v_obs * sigma_v) / r
    df['err_g_obs'] = (2.0 * v_obs_m_s * err_v_m_s) / radius_m
    
    # Convert to log10 space error for the chi-square objective function
    # sigma_log10(x) = sigma_x / (x * ln(10))
    df['err_log_g_obs'] = df['err_g_obs'] / (df['g_obs'] * np.log(10))
    
    return df


def save_processed_data(df, galaxy_name, output_dir="data/processed/"):
    """Saves the cleaned DataFrame to a CSV."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{galaxy_name}_processed.csv")
    df.to_csv(out_path, index=False)
    return out_path