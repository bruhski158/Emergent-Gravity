import numpy as np
from src.calculus import smooth_derivative_log

A0_DEFAULT = 1.2e-10

G_NEWTON = 6.67430e-11
MSUN_TO_KG = 1.98847e30
KPC_TO_M = 3.085677581e19
PC_TO_M = KPC_TO_M / 1000.0  

def model_nfw_acceleration(r_m, rho0_msun_pc3, rs_kpc):
    rs_m = rs_kpc * KPC_TO_M
    rho0_kg_m3 = rho0_msun_pc3 * (MSUN_TO_KG / (PC_TO_M)**3) 
    
    x = r_m / rs_m
    mass_term = np.log(1.0 + x) - (x / (1.0 + x))
    m_nfw_kg = 4.0 * np.pi * rho0_kg_m3 * (rs_m**3) * mass_term
    
    g_nfw = (G_NEWTON * m_nfw_kg) / (r_m**2)
    return g_nfw

def model_baryons_only(g_bar):
    return g_bar

def model_empirical_rar(g_bar, g_dagger=A0_DEFAULT):
    return g_bar / (1.0 - np.exp(-np.sqrt(g_bar / g_dagger)))

def model_mond_simple(g_bar, a0=A0_DEFAULT):
    return 0.5 * (g_bar + np.sqrt(g_bar**2 + 4 * g_bar * a0))

def model_emergent_gravity_point_mass(g_bar, a0=A0_DEFAULT):
    return g_bar + np.sqrt((a0 / 6.0) * g_bar)

def model_emergent_gravity_full(r, g_bar, a0=A0_DEFAULT):
    dgbar_dr = smooth_derivative_log(r, g_bar)
    inner_term = (a0 / 6.0) * (3.0 * g_bar + r * dgbar_dr)
    gd2 = np.where(inner_term > 0.0, inner_term, np.nan)
    return g_bar + np.sqrt(gd2)