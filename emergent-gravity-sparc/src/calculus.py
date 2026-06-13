import numpy as np
from scipy.interpolate import UnivariateSpline

def smooth_derivative_log(r, g_bar, smoothing_factor=1.0):
    """
    Computes a smooth derivative dg_bar/dr by fitting a spline in logarithmic space.
    Follows the exact procedure from Step 4 of the thesis.
    """
    # 1. Work in logarithmic variables to handle large dynamic ranges
    log_r = np.log10(r)
    log_gbar = np.log10(g_bar)
    
    # Sort the arrays to ensure the spline fits correctly
    sort_idx = np.argsort(log_r)
    log_r_sorted = log_r[sort_idx]
    log_gbar_sorted = log_gbar[sort_idx]
    
    # 2. Fit an interpolating spline to avoid numerical noise
    # s is the smoothing factor. Adjust if the derivative is still too noisy.
    spline = UnivariateSpline(log_r_sorted, log_gbar_sorted, s=smoothing_factor)
    
    # 3. Compute the derivative of the spline: d(log_gbar)/d(log_r)
    spline_deriv = spline.derivative()
    dlog_gbar_dlog_r = spline_deriv(log_r)
    
    # 4. Chain rule to get back to linear space: 
    # d(log10 y) / d(log10 x) = (x/y) * (dy/dx)  =>  dy/dx = (y/x) * d(log_y)/d(log_x)
    dgbar_dr = (g_bar / r) * dlog_gbar_dlog_r
    
    return dgbar_dr