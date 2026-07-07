# Testing Emergent Gravity: A SPARC Dataset Pipeline

[![License](https://img.shields.io/badge/License-See_File-blue.svg)](LICENSE)
[![Data](https://img.shields.io/badge/Data-SPARC-orange.svg)](http://astroweb.cwru.edu/SPARC/)

This repository contains the complete, automated pipeline developed for my thesis. It is designed to evaluate Erik Verlinde's **Emergent Gravity (EG)** alongside **Modified Newtonian Dynamics (MOND)** and standard **Navarro-Frenk-White (NFW)** Dark Matter halos.

Using high-quality, near-infrared galaxy rotation curve data from the **SPARC (Spitzer Photometry and Accurate Rotation Curves)** dataset, this pipeline executes a sequence of physical and statistical stress tests—evaluating everything from the universality of the acceleration scale ($a_0$) to the physical viability of implied mass distributions.

---

## 🔬 Scientific Highlights

Based on the aggregate analysis of the SPARC dataset, this pipeline demonstrates several diagnostic findings regarding the viability of these gravitational models:

* **The "Phantom Density" Problem (Negative Mass):** A fundamental physical viability check calculates the implied "phantom dark matter" density required by EG. The pipeline isolates genuine physical failures (e.g., NGC4013) where EG requires an unphysical **negative mass density** to match observed kinematics.
* **Failure of $a_0$ Universality:** While MOND successfully maintains a globally flat, universal $a_0$ target, Emergent Gravity's implied local acceleration scale systematically drifts and curves upward as a function of radius.
* **Kinematic Peak Misalignment:** MOND accurately predicts the spatial center of maximum rotation velocity $V(r)$. EG systematically pushes the predicted velocity peak too far outward into the galaxy.
* **Derivative Artifacts:** EG's residuals are statistically correlated with the localized radial gradient of the baryonic acceleration, suggesting its deviations are partly driven by mathematical artifacts in its derivative formulation.

---

## 🚀 How to Reproduce the Results

The codebase is strictly sequential. Running the numbered scripts in order will take you from raw data ingestion all the way to the final figures and statistical CSVs.

### 1. Prerequisites & Setup

Ensure you have Python 3.8+ installed along with the standard scientific stack.

```bash
git clone https://github.com/bruhski158/Emergent-Gravity
cd Emergent-Gravity
```

### 2. Download the SPARC Data

1. Download the raw SPARC dataset (`Rotmod_LTG.zip`) from the [SPARC Database](http://astroweb.cwru.edu/SPARC/).
2. Extract the `.dat` files into the following directory structure:
   `data/raw/Rotmod_LTG/`

### 3. Run the Pipeline

Execute the scripts sequentially from the project root. The pipeline is highly automated.

**Phase 1: Data Processing & Visual Validation**

```bash
python 01_data_ingestion.py        # Cleans raw .dat files, calculates accelerations, converts to SI
python 02_reproduce_rar.py         # Reproduces the classic 1:1 RAR scatter plot
python 03_diverse_galaxies.py      # Plots theoretical models against diverse radial profiles
```

**Phase 2: Core Analysis & Fitting**

```bash
python 04_residual_diagnostics.py  # Computes global residuals across all valid data points
python 05_mass_to_light_scan.py    # Explores theoretical sensitivity to stellar M/L ratios
python 06_inner_vs_outer.py        # Splits residual performance across the a_0 boundary
python 07_dark_matter_fitting.py   # Automated batch fitting for NFW Dark Matter halos
```

**Phase 3: Falsification & Stress Tests**

```bash
python 08_statistical_correlations.py # Spearman rank tests for unphysical spatial trends
python 09_optimize_mass_to_light.py   # Differential evolution optimization for M/L bounds
python 10_mond_vs_eg_universality.py  # Tests the constancy of the acceleration parameter
python 11_derivative_artifact_test.py # Isolates derivative-driven mathematical artifacts
python 12_peak_shift_test.py          # Evaluates predictive alignment of velocity peaks
python 13_phantom_density_test.py     # Calculates EG's effective dark mass density
python 14_tests.py                    # Cross-references failures against SPARC Q-flags
```

All generated CSV tables and high-resolution figures will be saved automatically to the `results/tables/` and `results/figures/` directories.

---

## 📁 Repository Structure

```text
├── data/
│   ├── raw/Rotmod_LTG/         # Raw SPARC .dat files
│   └── processed/Rotmod_LTG/   # Cleaned, SI-unit CSVs
├── results/
│   ├── figures/                # Generated visual plots
│   └── tables/                 # Statistical summaries and diagnostic CSVs
├── src/
│   ├── calculus.py             # Numerical derivatives & Savitzky-Golay filters
│   ├── io.py                   # Data ingestion and error propagation
│   ├── physics.py              # Gravitational models (NFW, MOND, EG, RAR)
│   └── plotting.py             # Visualization tools
├── CITATION.cff                # Citation information
├── LICENSE                     # License terms
├── scripts/
    └── [01-14]_*.py                # Executable pipeline scripts

```

---

## 📓 Conceptual Notebooks

Alongside the main SPARC pipeline, this repository includes a set of standalone Mathematica notebooks used to generate the background/illustrative figures in the thesis. These notebooks are self-contained theoretical toy models and sanity checks—they don't consume SPARC data and aren't part of the numbered `scripts/` pipeline, but they underpin the theoretical motivation for the analysis.

* **`Random Walk of String bits and numerical verification of the square-root relationship.nb`**
  Simulates a highly-excited string as a 3D random walk of "string bits" and numerically verifies the holographic scaling relation $R_{rms} \propto \sqrt{M}$. Generates a sample 3D string path plus log-log and linear fits of RMS radius vs. bit number, recovering the expected $\alpha \approx 0.5$ scaling exponent with fitted error bars.

* **`Saturation of Bekenstein Bound.nb`**
  Plots the Bekenstein entropy bound $S_{Bek} = 2\pi E R$ against the Bekenstein-Hawking black hole entropy $S_{BH} = 4\pi E^2$ to illustrate that a Schwarzschild black hole saturates the bound. Includes a second plot at fixed radius showing sub-black-hole configurations approaching but never exceeding the maximum entropy, demonstrating black holes as maximal-entropy objects.

* **`Scaling of Entropy and Area in a Flat FRW Universe.nb`**
  Log-log plot comparing volume-scaling entropy ($S \propto \chi_0^3$), area-scaling entropy ($A \propto \chi_0^2$), and their ratio as a function of comoving radius in a flat FRW cosmology—motivating the entropic/holographic scaling arguments underlying Verlinde's Emergent Gravity.

* **`Thermodynamic Quantities.nb`**
  Computes and plots the standard Hawking temperature, Bekenstein-Hawking entropy, and (negative) heat capacity of a Schwarzschild black hole as functions of mass (in solar masses), using physical constants in SI units. Illustrates that larger black holes are colder, have quadratically larger entropy, and exhibit the characteristic negative heat capacity of black hole thermodynamics.

* **`Toy model: Emergent-Gravity Rotation Curve.nb`**
  A minimal analytic toy model of a galaxy rotation curve under Emergent Gravity: combines the standard Newtonian acceleration $g_b = GM_b/r^2$ with an EG "dark" contribution $g_d = \sqrt{a_0 g_b}$ to produce a flattened total velocity curve $v_{tot}(r)$, plotted alongside the purely Newtonian prediction $v_N(r)$ for comparison. Serves as the conceptual precursor to the full SPARC-based EG fitting done in the numbered pipeline.

---

## 📜 License

This project is licensed under the terms provided in the [LICENSE](LICENSE) file. Please review it before using or distributing this code.

---

## 📝 Citation

If you use this pipeline, codebase, or the results generated herein for your research, please cite this repository/thesis directly. A citation file (`CITATION.cff`) is included in the root of this repository for your convenience.

Additionally, this project relies on the SPARC database. **Any use of this data must also cite the original SPARC publication:**

> Lelli, F., McGaugh, S. S., & Schombert, J. M. (2016). *SPARC: Mass Models for 175 Disk Galaxies with Spitzer Photometry and Accurate Rotation Curves*. The Astronomical Journal, 152(6), 157. [arXiv:1606.09251](https://arxiv.org/abs/1606.09251)