import matplotlib.pyplot as plt
import numpy as np
import os

def plot_radial_acceleration_relation(g_bar_array, g_obs_array, save_path=None):
    """Plots the empirical RAR: g_obs vs g_bar on a log-log scale."""
    plt.figure(figsize=(8, 6))
    
    # Scatter plot of all data points
    plt.scatter(g_bar_array, g_obs_array, alpha=0.15, s=15, color='blue', label='SPARC Data')
    
    # The 1:1 Line (Newtonian expectation without dark matter)
    min_val = min(np.min(g_bar_array), np.min(g_obs_array))
    max_val = max(np.max(g_bar_array), np.max(g_obs_array))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='1:1 Line (Baryons Only)')
    
    # Formatting the log-log plot
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$g_{\rm bar} \ (\rm m/s^2)$', fontsize=14)
    plt.ylabel(r'$g_{\rm obs} \ (\rm m/s^2)$', fontsize=14)
    plt.title('The Radial Acceleration Relation (RAR)', fontsize=16)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    
    # Save the figure if a path is provided (for your thesis document)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved to: {save_path}")
        
    plt.show()

def plot_galaxy_accelerations(r_m, g_obs, g_bar, models_dict, galaxy_name, save_path=None):
    """Plots the acceleration profiles for a single galaxy across different models."""
    plt.figure(figsize=(10, 6))
    
    # Convert radius back to kpc just for a cleaner x-axis on the plot
    KPC_TO_M = 3.085677581e19
    r_kpc = r_m / KPC_TO_M
    
    # Plot the baseline data
    plt.plot(r_kpc, g_obs, 'ko', label='Observed ($g_{\\rm obs}$)', markersize=5)
    plt.plot(r_kpc, g_bar, 'k--', label='Baryons Only ($g_{\\rm bar}$)', linewidth=2)
    
    # Plot each theoretical model
    colors = ['blue', 'red', 'green', 'purple']
    for i, (model_name, g_model) in enumerate(models_dict.items()):
        plt.plot(r_kpc, g_model, label=model_name, color=colors[i % len(colors)], linewidth=2, alpha=0.8)
    
    plt.yscale('log')
    plt.xlabel('Radius (kpc)', fontsize=14)
    plt.ylabel(r'Acceleration ($\rm m/s^2$)', fontsize=14)
    plt.title(f'Theoretical Models vs Observations: {galaxy_name}', fontsize=16)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved to: {save_path}")
        
    plt.show()

def plot_diverse_galaxies(galaxy_data_list, save_path=None):
    """Plots a 1x3 side-by-side comparison of diverse galaxy types."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    KPC_TO_M = 3.085677581e19
    colors = ['blue', 'red', 'green', 'purple']
    
    for ax, gal in zip(axes, galaxy_data_list):
        r_kpc = gal['r_m'] / KPC_TO_M
        
        # Plot data
        ax.plot(r_kpc, gal['g_obs'], 'ko', label='Observed', markersize=4)
        ax.plot(r_kpc, gal['g_bar'], 'k--', label='Baryons Only', linewidth=2)
        
        # Plot models
        for i, (model_name, g_model) in enumerate(gal['models'].items()):
            ax.plot(r_kpc, g_model, label=model_name, color=colors[i % len(colors)], linewidth=2, alpha=0.8)
            
        ax.set_yscale('log')
        ax.set_xlabel('Radius (kpc)', fontsize=14)
        ax.set_title(f"{gal['name']}\n({gal['type']})", fontsize=16)
        ax.grid(True, which="both", ls="--", alpha=0.4)
        
    axes[0].set_ylabel(r'Acceleration ($\rm m/s^2$)', fontsize=16)
    
    # Put a single legend below the middle plot
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3, fontsize=12)
    
    plt.tight_layout()
    # Adjust bottom margin to make room for the legend
    plt.subplots_adjust(bottom=0.25)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved to: {save_path}")
        
    plt.show()