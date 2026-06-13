import os
import sys
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.io import load_and_convert_sparc
from src.physics import model_emergent_gravity_full

def main():
    print("--- Starting Step 7: Mass-to-Light Ratio Scan ---")
    
    galaxy_name = "NGC5055"
    raw_filepath = os.path.join(project_root, "data", "raw", "Rotmod_LTG", f"{galaxy_name}_rotmod.dat")
    
    if not os.path.exists(raw_filepath):
        print(f"Error: Could not find raw file at {raw_filepath}.")
        return
        
    ml_ratios = {
        r"Low Mass ($\Upsilon_d=0.2, \Upsilon_b=0.5$)": (0.2, 0.5),
        r"Standard ($\Upsilon_d=0.5, \Upsilon_b=0.7$)": (0.5, 0.7),
        r"High Mass ($\Upsilon_d=0.8, \Upsilon_b=1.0$)": (0.8, 1.0)
    }
    
    colors = ['#1f77b4', '#2ca02c', '#d62728'] # Blue, Green, Red
    KPC_TO_M = 3.085677581e19
    
    plt.figure(figsize=(10, 6))
    
    for i, (label, (ups_d, ups_b)) in enumerate(ml_ratios.items()):
        df = load_and_convert_sparc(raw_filepath, ups_disk=ups_d, ups_bulge=ups_b)
        
        r_kpc = df['Rad'].values
        r_m = r_kpc * KPC_TO_M
        g_bar = df['g_bar'].values
        
        if i == 0:
            plt.plot(r_kpc, df['g_obs'].values, 'ko', label='Observed ($g_{\\rm obs}$)', markersize=5)
            
        g_eg = model_emergent_gravity_full(r_m, g_bar)
        
        plt.plot(r_kpc, g_eg, label=f'EG Full: {label}', color=colors[i], linewidth=2.5, alpha=0.8)

    plt.yscale('log')
    plt.xlabel('Radius (kpc)', fontsize=14)
    plt.ylabel(r'Acceleration ($\rm m/s^2$)', fontsize=14)
    plt.title(f'Emergent Gravity Sensitivity to Stellar Mass: {galaxy_name}', fontsize=16)
    plt.legend(loc='lower left', fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    
    plt.tight_layout()
    
    save_path = os.path.join(project_root, "results", "figures", "04_mass_to_light_scan.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Figure saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    main()