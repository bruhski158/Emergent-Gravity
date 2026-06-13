import numpy as np
from scipy.signal import savgol_filter

def smooth_derivative_log(r, g_bar):
    """
    Computes a smooth derivative dg_bar/dr using a Savitzky-Golay filter 
    in logarithmic space to suppress numerical noise.
    """
    log_r = np.log10(r)
    log_gbar = np.log10(g_bar)
    
    sort_idx = np.argsort(log_r)
    log_r_sorted = log_r[sort_idx]
    log_gbar_sorted = log_gbar[sort_idx]
    
    n_points = len(log_r_sorted)
    
    # Dynamic window sizing: Savitzky-Golay needs window_length > polyorder
    poly_order = 2
    if n_points < 5:
        # Fallback for extremely sparse data (simple finite difference)
        dlog_gbar_dlog_r = np.gradient(log_gbar_sorted, log_r_sorted)
    else:
        window_length = min(7, n_points if n_points % 2 != 0 else n_points - 1)
        if window_length <= poly_order:
            window_length = poly_order + 1 if (poly_order + 1) % 2 != 0 else poly_order + 2
            
        # Smooth the g_bar profile first
        smoothed_log_gbar = savgol_filter(log_gbar_sorted, window_length, poly_order)
        # Compute the derivative of the smoothed profile
        dlog_gbar_dlog_r = np.gradient(smoothed_log_gbar, log_r_sorted)
    
    # Chain rule to get back to linear space
    dgbar_dr = np.zeros_like(g_bar)
    dgbar_dr[sort_idx] = (10**log_gbar_sorted / 10**log_r_sorted) * dlog_gbar_dlog_r
    
    return dgbar_dr