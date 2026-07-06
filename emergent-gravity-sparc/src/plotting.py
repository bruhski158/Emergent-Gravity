import matplotlib.pyplot as plt
import numpy as np
import os
from src.physics import model_empirical_rar

def plot_radial_acceleration_relation(g_bar_array, g_obs_array, err_g_obs_array=None, save_path=None):
    plt.figure(figsize=(8, 6))
    
    err_g_bar_array = g_bar_array * (10**0.1 - 1.0)
    
    if err_g_obs_array is not None:
        plt.errorbar(g_bar_array, g_obs_array, xerr=err_g_bar_array, yerr=err_g_obs_array, fmt='o', alpha=0.15, 
                     markersize=3, color='blue', ecolor='gray', label='SPARC Data')
    else:
        plt.scatter(g_bar_array, g_obs_array, alpha=0.15, s=15, color='blue', label='SPARC Data')
    
    min_val = min(np.min(g_bar_array), np.min(g_obs_array))
    max_val = max(np.max(g_bar_array), np.max(g_obs_array))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='1:1 Line (Baryons Only)')
    
    g_bar_line = np.logspace(np.log10(min_val), np.log10(max_val), 100)
    g_obs_line = model_empirical_rar(g_bar_line)
    plt.plot(g_bar_line, g_obs_line, 'r-', linewidth=2.5, label='Empirical Fit (McGaugh 2016)')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$g_{\rm bar} \ (\rm m/s^2)$', fontsize=14)
    plt.ylabel(r'$g_{\rm obs} \ (\rm m/s^2)$', fontsize=14)
    plt.title('The Radial Acceleration Relation (RAR)', fontsize=16)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_diverse_galaxies(galaxy_data_list, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    KPC_TO_M = 3.085677581e19
    colors = ['blue', 'red', 'green', 'purple']
    
    for ax, gal in zip(axes, galaxy_data_list):
        r_kpc = gal['r_m'] / KPC_TO_M
        
        ax.errorbar(r_kpc, gal['g_obs'], yerr=gal['err_g_obs'], fmt='ko', label='Observed', markersize=4, capsize=2)
        ax.plot(r_kpc, gal['g_bar'], 'k--', label='Baryons Only', linewidth=2)
        
        for i, (model_name, g_model) in enumerate(gal['models'].items()):
            ax.plot(r_kpc, g_model, '.-', label=model_name, color=colors[i % len(colors)], linewidth=2, alpha=0.8)
            
        ax.set_yscale('log')
        ax.set_xlabel('Radius (kpc)', fontsize=14)
        ax.set_title(f"{gal['name']}\n({gal['type']})", fontsize=16)
        ax.grid(True, which="both", ls="--", alpha=0.4)
        
    axes[0].set_ylabel(r'Acceleration ($\rm m/s^2$)', fontsize=16)
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3, fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()