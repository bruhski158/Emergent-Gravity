import os
import sys
import pandas as pd
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.physics import (model_empirical_rar, model_mond_simple, 
                         model_emergent_gravity_point_mass, model_emergent_gravity_full)
from src.plotting import plot_galaxy_accelerations

def main():
    print("--- Starting Step 3: Model Predictions on a Single Galaxy ---")
    
    galaxy_name = "NGC5055"
    filepath = os.path.join(project_root, "data", "processed", "Rotmod_LTG", f"{galaxy_name}_processed.csv")
    
    if not os.path.exists(filepath):
        print(f"Error: Could not find {filepath}. Did you run Step 1?")
        return
        
    df = pd.read_csv(filepath)
    
    KPC_TO_M = 3.085677581e19
    r_m = df['Rad'].values * KPC_TO_M  
    g_bar = df['g_bar'].values
    g_obs = df['g_obs'].values
    
    print(f"Calculating theoretical models for {galaxy_name}...")
    
    models = {
        'Empirical RAR': model_empirical_rar(g_bar),
        'MOND (Simple)': model_mond_simple(g_bar),
        'Emergent Gravity (Point Mass)': model_emergent_gravity_point_mass(g_bar),
        'Emergent Gravity (Full Extended)': model_emergent_gravity_full(r_m, g_bar)
    }
    
    print("Generating plot...")
    save_path = os.path.join(project_root, "results", "figures", f"02_{galaxy_name}_models.png")
    
    plot_galaxy_accelerations(r_m, g_obs, g_bar, models, galaxy_name, save_path=save_path)

if __name__ == "__main__":
    main()