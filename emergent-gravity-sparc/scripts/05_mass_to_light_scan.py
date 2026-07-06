import os
import sys
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.io import load_and_convert_sparc
from src.physics import model_emergent_gravity_full, model_mond_simple

def main():
    print("--- Starting Step 7: Mass-to-Light Ratio Scan ---")
    galaxy_name = "NGC5055"
    raw_filepath = os.path.join(project_root, "data", "raw", "Rotmod_LTG", f"{galaxy_name}_rotmod.dat")
    
    ml_ratios = {
        r"Low Mass ($\Upsilon_d=0.2, \Upsilon_b=0.5$)": (0.2, 0.5),
        r"Standard ($\Upsilon_d=0.5, \Upsilon_b=0.7$)": (0.5, 0.7),
        r"High Mass ($\Upsilon_d=0.8, \Upsilon_b=1.0$)": (0.8, 1.0)
    }
    colors = ['#1f77b4', '#2ca02c', '#d62728'] 
    KPC_TO_M = 3.085677581e19
    
    plt.figure(figsize=(10, 6))
    for i, (label, (ups_d, ups_b)) in enumerate(ml_ratios.items()):
        df = load_and_convert_sparc(raw_filepath, ups_disk=ups_d, ups_bulge=ups_b)
        
        r_kpc = df['Rad'].values
        r_m = r_kpc * KPC_TO_M
        g_bar = df['g_bar'].values
        
        if i == 0:
            plt.errorbar(r_kpc, df['g_obs'].values, yerr=df['err_g_obs'].values, fmt='ko', label='Observed', markersize=5)
            
        g_eg = model_emergent_gravity_full(r_m, g_bar)
        g_mond = model_mond_simple(g_bar)
        
        plt.plot(r_kpc, g_eg, '-', label=f'EG Full: {label}', color=colors[i], linewidth=2.5, alpha=0.9)
        plt.plot(r_kpc, g_mond, '--', label=f'MOND: {label}', color=colors[i], linewidth=2.0, alpha=0.6)

    plt.yscale('log')
    plt.xlabel('Radius (kpc)', fontsize=14)
    plt.ylabel(r'Acceleration ($\rm m/s^2$)', fontsize=14)
    plt.title(f'Theoretical Sensitivity to Stellar Mass Choices: {galaxy_name}', fontsize=16)
    
    plt.legend(loc='upper right', fontsize=10, ncol=2)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    
    save_path = os.path.join(project_root, "results", "figures", "04_mass_to_light_scan.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()

if __name__ == "__main__":
    main()