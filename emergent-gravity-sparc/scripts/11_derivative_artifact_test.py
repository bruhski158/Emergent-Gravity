import os
import glob
import sys
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.physics import model_emergent_gravity_full, model_mond_simple
from src.calculus import smooth_derivative_log


def main():
    print("--- Starting Step 11: Derivative Artifact Test (corrected) ---")
    processed_dir = os.path.join(project_root, "data", "processed", "Rotmod_LTG")
    processed_files = glob.glob(os.path.join(processed_dir, "*.csv"))

    KPC_TO_M = 3.085677581e19

    all_log_gradient = []       
    all_abs_log_gradient = []
    all_abs_residual_eg = []
    all_wiggle_residual = []      
    galaxy_level_rs = []
    galaxy_names = []

    total_points = 0
    failed_files = []

    for filepath in processed_files:
        galaxy_name = os.path.splitext(os.path.basename(filepath))[0]
        df = pd.read_csv(filepath)
        valid_data = df[(df['g_bar'] > 0) & (df['g_obs'] > 0)].copy()

        if valid_data.empty or len(valid_data) < 4:
            continue

        r_m = valid_data['Rad'].values * KPC_TO_M
        g_bar = valid_data['g_bar'].values
        g_obs = valid_data['g_obs'].values

        total_points += len(g_obs)

        try:
            g_eg = model_emergent_gravity_full(r_m, g_bar)
            g_mond = model_mond_simple(g_bar)

            delta_eg = np.log10(g_obs) - np.log10(g_eg)

            wiggle_residual = np.log10(g_mond) - np.log10(g_eg)

            abs_residual_eg = np.abs(delta_eg)

            dgbar_dr = smooth_derivative_log(r_m, g_bar)

            log_gradient = (r_m / g_bar) * dgbar_dr
            abs_log_gradient = np.abs(log_gradient)

            finite_mask = (
                np.isfinite(abs_residual_eg)
                & np.isfinite(abs_log_gradient)
                & np.isfinite(wiggle_residual)
                & np.isfinite(log_gradient)
            )

            if not np.any(finite_mask):
                continue

            all_log_gradient.extend(log_gradient[finite_mask])
            all_abs_log_gradient.extend(abs_log_gradient[finite_mask])
            all_abs_residual_eg.extend(abs_residual_eg[finite_mask])
            all_wiggle_residual.extend(wiggle_residual[finite_mask])

            if finite_mask.sum() >= 5:
                r_s_gal, _ = spearmanr(
                    log_gradient[finite_mask], wiggle_residual[finite_mask]
                )
                if np.isfinite(r_s_gal):
                    galaxy_level_rs.append(r_s_gal)
                    galaxy_names.append(galaxy_name)

        except Exception as e:
            failed_files.append((galaxy_name, str(e)))

    x_abs = np.array(all_abs_log_gradient)
    y_abs = np.array(all_abs_residual_eg)
    x_signed = np.array(all_log_gradient)
    y_wiggle = np.array(all_wiggle_residual)
    galaxy_level_rs = np.array(galaxy_level_rs)

    print("\n[DIAGNOSTICS]")
    print(f"Data Points Analyzed: {len(x_abs)} / {total_points}")
    print(f"Galaxies that raised an exception: {len(failed_files)}")
    for name, err in failed_files:
        print(f"    - {name}: {err}")

    if len(x_abs) == 0:
        print("No valid points survived filtering -- nothing to correlate.")
        return

    spearman_abs, p_abs = spearmanr(x_abs, y_abs)
    spearman_signed, p_signed = spearmanr(x_signed, y_wiggle)

    print("\n--- Test A (original): |gradient| vs |EG residual|, pooled points ---")
    print(f"Spearman r_s: {spearman_abs:.4f} | p-value: {p_abs:.4e}")

    print("\n--- Test B (recommended): signed gradient vs signed wiggle residual, pooled points ---")
    print("NOTE: pooling points across galaxies as if independent makes this p-value")
    print("anti-conservative (too small). Treat Test C below as the more honest number.")
    print(f"Spearman r_s: {spearman_signed:.4f} | p-value: {p_signed:.4e}")

    print("\n--- Test C: per-galaxy correlation, one r_s per galaxy ---")
    print(f"Galaxies included: {len(galaxy_level_rs)}")
    if len(galaxy_level_rs) > 0:
        print(f"Mean r_s across galaxies:   {np.mean(galaxy_level_rs):.4f}")
        print(f"Median r_s across galaxies: {np.median(galaxy_level_rs):.4f}")
        print(f"Fraction of galaxies with r_s > 0: {np.mean(galaxy_level_rs > 0):.2%}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    hb0 = axes[0].hexbin(x_abs, y_abs, gridsize=30, cmap='Reds', mincnt=1, xscale='log')
    fig.colorbar(hb0, ax=axes[0], label='Number of Data Points')
    axes[0].text(0.05, 0.95, f"Spearman $r_s$ = {spearman_abs:.2f}\n$p$-value = {p_abs:.2e}",
                 transform=axes[0].transAxes, fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    axes[0].set_xlabel(r'$| d \log(g_{\rm bar}) / d \log(r) |$', fontsize=13)
    axes[0].set_ylabel(r'$|\Delta_{\rm EG}|$', fontsize=13)
    axes[0].set_title('Test A: magnitude-only (original)', fontsize=14)
    axes[0].grid(True, which="both", ls="--", alpha=0.3)

    hb1 = axes[1].hexbin(x_signed, y_wiggle, gridsize=30, cmap='Purples', mincnt=1)
    fig.colorbar(hb1, ax=axes[1], label='Number of Data Points')
    axes[1].axhline(0, color='black', lw=1, ls='--')
    axes[1].axvline(0, color='black', lw=1, ls='--')
    axes[1].text(0.05, 0.95, f"Spearman $r_s$ = {spearman_signed:.2f}\n$p$-value = {p_signed:.2e}",
                 transform=axes[1].transAxes, fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    axes[1].set_xlabel(r'Signed $d \log(g_{\rm bar}) / d \log(r)$', fontsize=13)
    axes[1].set_ylabel(r'Wiggle residual: $\log_{10}(g_{\rm mond}/g_{\rm eg})$', fontsize=13)
    axes[1].set_title('Test B: signed, wiggle-isolating', fontsize=14)
    axes[1].grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(project_root, "results", "figures", "11_derivative_artifact.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    main()