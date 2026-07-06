import os
import sys
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.physics import (model_empirical_rar, model_mond_simple, 
                         model_emergent_gravity_point_mass, model_emergent_gravity_full)
from src.plotting import plot_diverse_galaxies

def main():
    print("--- Starting Step 3: Diverse Galaxy Comparison ---")
    targets = {
        "NGC5055": "High-Surface-Brightness Spiral",
        "UGC07524": "Low-Surface-Brightness",
        "DDO154": "Gas-Rich Dwarf"
    }
    
    galaxy_data_list = []
    KPC_TO_M = 3.085677581e19
    
    for name, gal_type in targets.items():
        filepath = os.path.join(project_root, "data", "processed", "Rotmod_LTG", f"{name}_processed.csv")
        df = pd.read_csv(filepath)
        r_m = df['Rad'].values * KPC_TO_M
        g_bar = df['g_bar'].values
        g_obs = df['g_obs'].values
        err_g_obs = df['err_g_obs'].values
        
        models = {
            'Empirical RAR': model_empirical_rar(g_bar),
            'MOND (Simple)': model_mond_simple(g_bar),
            'Emergent Gravity (Point Mass)': model_emergent_gravity_point_mass(g_bar),
            'Emergent Gravity (Full Extended)': model_emergent_gravity_full(r_m, g_bar)
        }
        
        galaxy_data_list.append({
            'name': name, 'type': gal_type, 'r_m': r_m,
            'g_obs': g_obs, 'err_g_obs': err_g_obs,
            'g_bar': g_bar, 'models': models
        })
        
    save_path = os.path.join(project_root, "results", "figures", "02_diverse_galaxies_comparison.png")
    plot_diverse_galaxies(galaxy_data_list, save_path=save_path)

if __name__ == "__main__":
    main()