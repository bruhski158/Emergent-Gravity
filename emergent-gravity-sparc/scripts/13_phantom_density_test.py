import os
import glob
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.physics import model_emergent_gravity_full
from src.calculus import smooth_derivative_log


def compute_density(r_m, m_dark_enclosed, method="smooth"):
    if method == "smooth":
        dM_dr = smooth_derivative_log(r_m, m_dark_enclosed)
    elif method == "raw":
        dM_dr = np.gradient(m_dark_enclosed, r_m)
    else:
        raise ValueError(method)
    return (1.0 / (4.0 * np.pi * r_m ** 2)) * dM_dr


def main():
    print("--- Starting Step 13: Phantom Dark Density Test (Strict + Local Gradient) ---")
    processed_dir = os.path.join(project_root, "data", "processed", "Rotmod_LTG")
    processed_files = glob.glob(os.path.join(processed_dir, "*.csv"))

    G_NEWTON   = 6.67430e-11
    KPC_TO_M   = 3.085677581e19
    MSUN_TO_KG = 1.98847e30
    PC_TO_M    = KPC_TO_M / 1000.0

    total_attempted       = 0
    excluded_nan_eg       = 0
    clean_galaxies        = 0
    count_smooth_only     = 0
    count_raw_only        = 0
    count_genuine_failure = 0

    genuine_min_densities  = []
    genuine_rel_violations = []
    per_galaxy_data        = [] 

    failed_files = []

    example_success_name = "NGC5055"
    example_success_data = None

    example_fail_name        = None
    example_fail_data        = None
    worst_relative_violation = 0.0

    for filepath in processed_files:
        filename    = os.path.basename(filepath)
        galaxy_name = (filename
                       .replace('_processed.csv', '')
                       .replace('_rotmod.csv',    '')
                       .replace('.csv',           ''))

        df = pd.read_csv(filepath)
        valid_data = df[(df['g_bar'] > 0) & (df['g_obs'] > 0)].copy()

        if valid_data.empty or len(valid_data) < 5:
            continue

        r_kpc = valid_data['Rad'].values
        r_m   = r_kpc * KPC_TO_M
        g_bar = valid_data['g_bar'].values

        total_attempted += 1

        try:
            g_eg = model_emergent_gravity_full(r_m, g_bar)

            if np.any(np.isnan(g_eg)):
                excluded_nan_eg += 1
                continue

            clean_galaxies += 1

            g_dark          = g_eg - g_bar
            m_dark_enclosed = (r_m ** 2 * g_dark) / G_NEWTON

            rho_kg_m3 = {
                "smooth": compute_density(r_m, m_dark_enclosed, "smooth"),
                "raw":    compute_density(r_m, m_dark_enclosed, "raw"),
            }
            rho_msun_pc3 = {
                k: v * ((PC_TO_M ** 3) / MSUN_TO_KG)
                for k, v in rho_kg_m3.items()
            }

            trim        = 1
            core_smooth = rho_msun_pc3["smooth"][trim:-trim]
            core_raw    = rho_msun_pc3["raw"][trim:-trim]

            is_smooth_neg = len(core_smooth) > 0 and np.any(core_smooth < 0)
            is_raw_neg    = len(core_raw)    > 0 and np.any(core_raw    < 0)
            is_genuine    = is_smooth_neg and is_raw_neg

            if is_genuine:
                count_genuine_failure += 1

                min_smooth       = np.min(core_smooth)
                min_raw          = np.min(core_raw)
                conservative_min = max(min_smooth, min_raw)
                genuine_min_densities.append(conservative_min)

                pos_vals = core_smooth[core_smooth > 0]
                rel_violation = (
                    abs(conservative_min) / np.max(pos_vals)
                    if len(pos_vals) > 0 else np.inf
                )
                genuine_rel_violations.append(rel_violation)

                try:
                    dgbar_dr     = smooth_derivative_log(r_m, g_bar)
                    log_gradient = (r_m / g_bar) * dgbar_dr         
                    abs_log_grad = np.abs(log_gradient)

                    max_abs_gradient = np.max(abs_log_grad)

                    idx_min_core    = int(np.argmin(core_smooth))
                    idx_full        = idx_min_core + trim
                    gradient_at_min = abs_log_grad[idx_full]

                except Exception:
                    max_abs_gradient = np.nan
                    gradient_at_min  = np.nan

                per_galaxy_data.append({
                    "Galaxy":               galaxy_name,
                    "Genuine_Failure":      True,
                    "Min_Density_Msun_pc3": conservative_min,
                    "Rel_Violation":        rel_violation,
                    "Max_LogLog_Gradient":  max_abs_gradient,
                    "Gradient_At_Min":      gradient_at_min,
                })

                if galaxy_name == "NGC4013":
                    print(f"\n[NGC4013 DEBUG]")
                    print(f"  core_smooth: {core_smooth}")
                    print(f"  core_raw:    {core_raw}")
                    print(f"  argmin(core_smooth): {np.argmin(core_smooth)}")
                    print(f"  argmin(core_raw):    {np.argmin(core_raw)}")

                if rel_violation > worst_relative_violation:
                    worst_relative_violation = rel_violation
                    example_fail_name = galaxy_name
                    example_fail_data = (
                        r_kpc,
                        rho_msun_pc3["smooth"],
                        rho_msun_pc3["raw"]
                    )

            else:
                if is_smooth_neg:
                    count_smooth_only += 1
                elif is_raw_neg:
                    count_raw_only += 1

                per_galaxy_data.append({
                    "Galaxy":               galaxy_name,
                    "Genuine_Failure":      False,
                    "Min_Density_Msun_pc3": np.nan,
                    "Rel_Violation":        np.nan,
                    "Max_LogLog_Gradient":  np.nan,
                    "Gradient_At_Min":      np.nan,
                })

            if galaxy_name == example_success_name:
                example_success_data = (
                    r_kpc,
                    rho_msun_pc3["smooth"],
                    rho_msun_pc3["raw"]
                )

        except Exception as e:
            failed_files.append((galaxy_name, str(e)))

  
    print("\n[DIAGNOSTICS]")
    print(f"Total galaxies attempted:          {total_attempted}")
    print(f"Excluded (NaN in g_eg):            {excluded_nan_eg}")
    print(f"Clean galaxies evaluated:          {clean_galaxies}")
    print(f"Galaxies that raised an exception: {len(failed_files)}")
    for name, err in failed_files:
        print(f"    - {name}: {err}")

    if clean_galaxies == 0:
        print("No clean galaxies -- cannot compute failure rates.")
        return

    rate_smooth_only = (count_smooth_only     / clean_galaxies) * 100
    rate_raw_only    = (count_raw_only        / clean_galaxies) * 100
    rate_genuine     = (count_genuine_failure / clean_galaxies) * 100

    print("\n[STRICT PHANTOM DENSITY RESULTS]")
    print(f"Smooth-only negative (artifact suspect): {count_smooth_only:3d}  ({rate_smooth_only:.1f}%)")
    print(f"Raw-only negative (noise suspect):       {count_raw_only:3d}  ({rate_raw_only:.1f}%)")
    print(f"GENUINE FAILURE (both methods agree):    {count_genuine_failure:3d}  ({rate_genuine:.1f}%)")

    if len(genuine_min_densities) > 0:
        arr_abs = np.array(genuine_min_densities)
        arr_rel = np.array(genuine_rel_violations)

        print("\n[MAGNITUDE DISTRIBUTION -- genuine failures only]")
        print(f"  Median minimum density:        {np.median(arr_abs):.4f}  M_sun/pc^3")
        print(f"  Most extreme (absolute):       {np.min(arr_abs):.4f}  M_sun/pc^3")
        print(f"  Median relative violation:     {np.median(arr_rel)*100:.1f}%  of local peak density")
        print(f"  Most extreme (relative):       {np.max(arr_rel)*100:.1f}%  of local peak density")
        print(f"  Failures with |min| > 0.01:    {np.sum(arr_abs < -0.01)} / {len(arr_abs)}")
        print(f"  Failures with |min| > 0.05:    {np.sum(arr_abs < -0.05)} / {len(arr_abs)}")

    csv_dir = os.path.join(project_root, "results", "tables")
    os.makedirs(csv_dir, exist_ok=True)

    summary_df = pd.DataFrame({
        "Category":   ["Smooth-Only (Artifact)", "Raw-Only (Noise)", "Genuine (Intersection)"],
        "Count":      [count_smooth_only, count_raw_only, count_genuine_failure],
        "Percentage": [rate_smooth_only,  rate_raw_only,  rate_genuine],
    })
    summary_df.to_csv(os.path.join(csv_dir, "13_phantom_density_strict.csv"), index=False)

    per_galaxy_df = pd.DataFrame(per_galaxy_data)
    per_galaxy_path = os.path.join(csv_dir, "13_per_galaxy_diagnostics.csv")
    per_galaxy_df.to_csv(per_galaxy_path, index=False)
    print(f"\nSaved per-galaxy data to: {per_galaxy_path}")

    failing_subset = per_galaxy_df[
        per_galaxy_df["Genuine_Failure"] == True
    ].dropna(subset=["Rel_Violation", "Gradient_At_Min"])

    if len(failing_subset) > 1:
        corr, p_val = spearmanr(
            failing_subset["Rel_Violation"],
            failing_subset["Gradient_At_Min"]
        )
        print("\n[GRADIENT STEEPNESS TEST]")
        print(f"Spearman r_s (Rel Violation vs Gradient at Min): {corr:.3f}  (p = {p_val:.2e})")
        if corr > 0.4 and p_val < 0.05:
            print("  -> CLAIM VERIFIED: soften 'concentrated in' is fine as stated.")
        else:
            print("  -> CLAIM UNSUPPORTED: soften to 'plausibly linked to'.")

    if example_success_data is not None and example_fail_data is not None:
        print(f"\nPlotting genuine failure: {example_fail_name} "
              f"(relative violation: {worst_relative_violation*100:.1f}%)")

        fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

        def plot_density(ax, title, data):
            r, rho_smooth, rho_raw = data
            ax.axhline(0, color='red', linestyle='-', linewidth=2,
                       label='Physical Bound (Zero Mass)')
            ax.plot(r, rho_smooth, 'k.-', markersize=8, linewidth=2,
                    label='EG Density (smoothed)')
            ax.plot(r, rho_raw, 'b.--', markersize=6, linewidth=1.5,
                    alpha=0.7, label='EG Density (raw)')
            ax.fill_between(r, rho_smooth, 0,
                            where=(rho_smooth < 0),
                            color='red', alpha=0.3,
                            label='Negative Mass Region')
            ax.set_xlabel('Radius (kpc)', fontsize=14)
            ax.set_title(title, fontsize=16)
            ax.grid(True, which="both", ls="--", alpha=0.4)
            ax.legend(fontsize=10)

        plot_density(axes[0],
                     f'{example_success_name} (Physical Profile)',
                     example_success_data)
        axes[0].set_ylabel(
            r'Apparent Density $\rho_{\rm D} \ (M_\odot / {\rm pc}^3)$',
            fontsize=14
        )

        plot_density(axes[1],
                     f'{example_fail_name} (Genuine Viability Failure, '
                     f'rel. violation {worst_relative_violation*100:.0f}%)',
                     example_fail_data)

        plt.tight_layout()
        save_path = os.path.join(
            project_root, "results", "figures", "13_phantom_density.png"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.show()

    elif example_success_data is None:
        print(f"\nWarning: example success galaxy '{example_success_name}' not found in sample.")
    elif example_fail_data is None:
        print("\nWarning: no genuine failure found to plot.")


if __name__ == "__main__":
    main()