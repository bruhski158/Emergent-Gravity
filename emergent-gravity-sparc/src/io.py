import numpy as np
import pandas as pd
import os

KM_TO_M = 1.0e3
KPC_TO_M = 3.085677581e19

def load_and_convert_sparc(filepath, ups_disk=0.5, ups_bulge=0.7):
    """
    Loads a raw SPARC .dat file, applies mass-to-light ratios, 
    and computes g_obs and g_bar in strict SI units (m/s^2).
    """
    # Explicitly define the column names because the file's header is hidden behind a '#'
    col_names = ['Rad', 'Vobs', 'errV', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul']
    
    # Read the file, ignore the '#' text lines, and enforce our column names
    df = pd.read_csv(filepath, sep='\s+', comment='#', names=col_names)
    
    # Drop rows where Radius is 0 to prevent division-by-zero
    df = df[df['Rad'] > 0].copy()
    
    # Reconstruct Baryonic Velocity Squared (Eq. 5)
    # Using np.sign to maintain the direction of the gravitational pull
    v_gas_sq = np.sign(df['Vgas']) * df['Vgas']**2
    v_disk_sq = np.sign(df['Vdisk']) * df['Vdisk']**2
    v_bulge_sq = np.sign(df['Vbul']) * df['Vbul']**2
    
    v_bar_sq = v_gas_sq + (ups_disk * v_disk_sq) + (ups_bulge * v_bulge_sq)
    
    # Convert to SI units (Eq. 6)
    radius_m = df['Rad'] * KPC_TO_M
    v_obs_m_s = df['Vobs'] * KM_TO_M
    
    # Calculate Accelerations
    df['g_obs'] = (v_obs_m_s**2) / radius_m
    df['g_bar'] = (v_bar_sq * (KM_TO_M**2)) / radius_m
    
    return df

def save_processed_data(df, galaxy_name, output_dir="data/processed/"):
    """Saves the cleaned DataFrame to a CSV."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{galaxy_name}_processed.csv")
    df.to_csv(out_path, index=False)
    return out_path