import os
import glob
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import ttest_1samp, ttest_rel, wilcoxon

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.physics import model_emergent_gravity_full, model_mond_simple


def safe_savgol_window(n_points, desired=5, min_window=3):
    w = min(desired, n_points)
    if w % 2 == 0:
        w -= 1
    if w < min_window:
        return None
    return w


def find_interior_peak(arr):
    idx = int(np.argmax(arr))
    if idx == 0 or idx == len(arr) - 1:
        return None
    return idx


def main():
    print("--- Starting Step 12: Kinematic Peak-Shift Test (corrected) ---")
    processed_dir = os.path.join(project_root, "data", "processed", "Rotmod_LTG")
    processed_files = glob.glob(os.path.join(processed_dir, "*.csv"))

    shift_eg = []
    shift_mond = []
    used_galaxies = []
    skipped_no_peak = 0
    skipped_nan_model = 0
    failed_files = []

    KPC_TO_M = 3.085677581e19

    for filepath in processed_files:
        galaxy_name = os.path.splitext(os.path.basename(filepath))[0]
        df = pd.read_csv(filepath)
        valid_data = df[(df['g_bar'] > 0) & (df['g_obs'] > 0)].copy()

        if valid_data.empty or len(valid_data) < 8:
            continue

        r_kpc = valid_data['Rad'].values
        r_m = r_kpc * KPC_TO_M
        g_bar = valid_data['g_bar'].values
        g_obs_raw = valid_data['g_obs'].values

        window = safe_savgol_window(len(g_obs_raw), desired=5)
        if window is None:
            continue

        try:
            g_eg = model_emergent_gravity_full(r_m, g_bar)
            g_mond = model_mond_simple(g_bar)

            if np.all(np.isnan(g_eg)) or np.all(np.isnan(g_mond)):
                continue

            if np.any(np.isnan(g_eg)) or np.any(np.isnan(g_mond)):
                skipped_nan_model += 1
                continue

            v_obs_raw = np.sqrt(r_m * g_obs_raw)
            v_eg = np.sqrt(r_m * g_eg)
            v_mond = np.sqrt(r_m * g_mond)

            v_obs_smooth = savgol_filter(v_obs_raw, window_length=window, polyorder=2)
            v_eg_smooth = savgol_filter(v_eg, window_length=window, polyorder=2)
            v_mond_smooth = savgol_filter(v_mond, window_length=window, polyorder=2)

            idx_peak_obs = find_interior_peak(v_obs_smooth)
            idx_peak_eg = find_interior_peak(v_eg_smooth)
            idx_peak_mond = find_interior_peak(v_mond_smooth)

            if idx_peak_obs is None or idx_peak_eg is None or idx_peak_mond is None:
                skipped_no_peak += 1
                continue

            r_peak_obs = r_kpc[idx_peak_obs]
            r_peak_eg = r_kpc[idx_peak_eg]
            r_peak_mond = r_kpc[idx_peak_mond]
            r_max = np.max(r_kpc)

            shift_eg.append((r_peak_obs - r_peak_eg) / r_max)
            shift_mond.append((r_peak_obs - r_peak_mond) / r_max)
            used_galaxies.append(galaxy_name)

        except Exception as e:
            failed_files.append((galaxy_name, str(e)))

    shift_eg = np.array(shift_eg)
    shift_mond = np.array(shift_mond)

    print("\n[DIAGNOSTICS]")
    print(f"Galaxies with a genuine interior V(r) peak in obs, EG, and MOND: {len(shift_eg)}")
    print(f"Galaxies skipped (no interior peak / monotonic V(r)): {skipped_no_peak}")
    print(f"Galaxies skipped (NaN in g_eg or g_mond): {skipped_nan_model}")
    print(f"Galaxies that raised an exception: {len(failed_files)}")
    for name, err in failed_files:
        print(f"    - {name}: {err}")

    if len(shift_eg) < 2:
        print("Not enough galaxies with genuine peaks to run statistics.")
        return

    t_stat_eg, p_val_eg = ttest_1samp(shift_eg, 0.0)
    t_stat_mond, p_val_mond = ttest_1samp(shift_mond, 0.0)

    print(f"\nEG Mean Shift:   {np.mean(shift_eg):.4f} (vs 0, p = {p_val_eg:.2e})")
    print(f"MOND Mean Shift: {np.mean(shift_mond):.4f} (vs 0, p = {p_val_mond:.2e})")

    t_stat_diff, p_val_diff = ttest_rel(shift_eg, shift_mond)
    try:
        w_stat, p_val_wilcoxon = wilcoxon(shift_eg, shift_mond)
    except ValueError:
        w_stat, p_val_wilcoxon = np.nan, np.nan

    print(f"\nPaired t-test (EG shift vs MOND shift):  t = {t_stat_diff:.4f}, p = {p_val_diff:.4e}")
    print(f"Wilcoxon signed-rank (EG vs MOND, robust): p = {p_val_wilcoxon:.4e}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    bins = np.linspace(-1.0, 1.0, 30)

    axes[0].hist(shift_mond, bins=bins, color='blue', alpha=0.7, edgecolor='black')
    axes[0].axvline(0, color='black', linestyle='--', linewidth=2)
    axes[0].axvline(np.mean(shift_mond), color='red', linestyle='-', linewidth=3,
                     label=f'Mean Shift: {np.mean(shift_mond):.2f}')
    axes[0].set_xlabel(r'Normalized Radial Shift $\Delta R / R_{\rm max}$', fontsize=14)
    axes[0].set_ylabel('Number of Galaxies', fontsize=14)
    axes[0].set_title('MOND Peak Alignment ($V(r)$)', fontsize=16)
    axes[0].legend(fontsize=12)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    axes[1].hist(shift_eg, bins=bins, color='purple', alpha=0.7, edgecolor='black')
    axes[1].axvline(0, color='black', linestyle='--', linewidth=2)
    axes[1].axvline(np.mean(shift_eg), color='red', linestyle='-', linewidth=3,
                     label=f'Mean Shift: {np.mean(shift_eg):.2f}')
    axes[1].set_xlabel(r'Normalized Radial Shift $\Delta R / R_{\rm max}$', fontsize=14)
    axes[1].set_title('Emergent Gravity Peak Alignment ($V(r)$)', fontsize=16)
    axes[1].legend(fontsize=12)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    save_path = os.path.join(project_root, "results", "figures", "12_peak_shift_test.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    main()