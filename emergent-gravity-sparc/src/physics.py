import numpy as np
from src.calculus import smooth_derivative_log

A0_DEFAULT = 1.2e-10

G_NEWTON = 6.67430e-11  # m^3 kg^-1 s^-2
MSUN_TO_KG = 1.98847e30
KPC_TO_M = 3.085677581e19

def model_nfw_acceleration(r_m, rho0_msun_pc3, rs_kpc):
    """
    Calculates the NFW dark matter acceleration.
    r_m: radius in meters
    rho0_msun_pc3: characteristic density in M_sun / pc^3 (standard literature unit)
    rs_kpc: scale radius in kpc
    """
    # Convert inputs to SI units
    rs_m = rs_kpc * KPC_TO_M
    # Convert M_sun/pc^3 to kg/m^3
    rho0_kg_m3 = rho0_msun_pc3 * (MSUN_TO_KG / (3.086e16)**3) 
    
    x = r_m / rs_m
    
    # Enclosed NFW Mass (Eq from thesis)
    mass_term = np.log(1.0 + x) - (x / (1.0 + x))
    m_nfw_kg = 4.0 * np.pi * rho0_kg_m3 * (rs_m**3) * mass_term
    
    # NFW Acceleration: G * M / r^2
    g_nfw = (G_NEWTON * m_nfw_kg) / (r_m**2)
    
    return g_nfw

def model_baryons_only(g_bar):
    """Model 1: Pure Newtonian baryons."""
    return g_bar

def model_empirical_rar(g_bar, g_dagger=A0_DEFAULT):
    """Model 3: The standard Empirical RAR fit from McGaugh (2016)."""
    return g_bar / (1.0 - np.exp(-np.sqrt(g_bar / g_dagger)))

def model_mond_simple(g_bar, a0=A0_DEFAULT):
    """Model 4: MOND using the simple interpolation function mu(x) = x/(1+x)."""
    # Solved quadratic for total g given the simple mu function
    return 0.5 * (g_bar + np.sqrt(g_bar**2 + 4 * g_bar * a0))

def model_emergent_gravity_point_mass(g_bar, a0=A0_DEFAULT):
    """Verlinde EG: Point-mass approximation (Eq 18 in thesis)."""
    return g_bar + np.sqrt((a0 / 6.0) * g_bar)

def model_emergent_gravity_full(r, g_bar, a0=A0_DEFAULT):
    """
    Model 2: Verlinde-type Emergent Gravity (Finite-Size/Extended Mass).
    Uses the smoothed derivative to calculate the elastic response.
    """
    # Step 4: Compute the smooth derivative
    dgbar_dr = smooth_derivative_log(r, g_bar)
    
    # The term inside the square root
    inner_term = (a0 / 6.0) * (3.0 * g_bar + r * dgbar_dr)
    
    # Step 4.5: Check for positivity. If numerical noise pushes this negative, 
    # we flag it with np.nan as explicitly requested in the thesis workflow.
    gd2 = np.where(inner_term > 0.0, inner_term, np.nan)
    
    # Final total acceleration
    return g_bar + np.sqrt(gd2)