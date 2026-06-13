import os
import glob
import sys
import pandas as pd
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.plotting import plot_radial_acceleration_relation

def main():
    print("--- Starting Step 2: Reproducing the RAR ---")
    
    processed_dir = os.path.join(project_root, "data", "processed", "Rotmod_LTG")
    processed_files = glob.glob(os.path.join(processed_dir, "*.csv"))
    
    if not processed_files:
        print("Error: No processed CSV files found. Did you run Step 1?")
        return
        
    print(f"Loading data from {len(processed_files)} galaxies...")
    
    g_bar_all = []
    g_obs_all = []
    
    for filepath in processed_files:
        df = pd.read_csv(filepath)
        
        valid_data = df[(df['g_bar'] > 0) & (df['g_obs'] > 0)]
        
        g_bar_all.extend(valid_data['g_bar'].values)
        g_obs_all.extend(valid_data['g_obs'].values)
        
    g_bar_all = np.array(g_bar_all)
    g_obs_all = np.array(g_obs_all)
    
    print(f"Aggregated {len(g_bar_all)} individual radial data points.")
    print("Generating plot...")
    
    figure_path = os.path.join(project_root, "results", "figures", "01_empirical_RAR.png")
    
    plot_radial_acceleration_relation(g_bar_all, g_obs_all, save_path=figure_path)

if __name__ == "__main__":
    main()