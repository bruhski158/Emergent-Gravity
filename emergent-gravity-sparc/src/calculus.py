import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

def smooth_derivative_log(r, g_bar):
    log_r = np.log10(r)
    log_gbar = np.log10(g_bar)
    
    sort_idx = np.argsort(log_r)
    log_r_sorted = log_r[sort_idx]
    log_gbar_sorted = log_gbar[sort_idx]
    
    n_points = len(log_r_sorted)
    poly_order = 2
    
    if n_points < 10:
        dlog_gbar_dlog_r = np.gradient(log_gbar_sorted, log_r_sorted)
    else:
        window_length = min(7, n_points if n_points % 2 != 0 else n_points - 1)
        if window_length <= poly_order:
            window_length = poly_order + 1 if (poly_order + 1) % 2 != 0 else poly_order + 2
            
        pad_width = window_length // 2
        
        uniform_log_r = np.linspace(log_r_sorted.min(), log_r_sorted.max(), num=n_points*2)
        
        interp_func = interp1d(log_r_sorted, log_gbar_sorted, kind='cubic')
        uniform_log_gbar = interp_func(uniform_log_r)
        
        padded_log_gbar = np.pad(
            uniform_log_gbar, 
            pad_width, 
            mode='reflect', 
            reflect_type='odd'
        )
        
        smoothed_padded = savgol_filter(padded_log_gbar, window_length, poly_order)
        smoothed_uniform_log_gbar = smoothed_padded[pad_width:-pad_width]
        
        uniform_gradient = np.gradient(smoothed_uniform_log_gbar, uniform_log_r)
        
        grad_interp_func = interp1d(uniform_log_r, uniform_gradient, kind='cubic')
        dlog_gbar_dlog_r = grad_interp_func(log_r_sorted)
    
    dgbar_dr = np.zeros_like(g_bar)
    dgbar_dr[sort_idx] = (10**log_gbar_sorted / 10**log_r_sorted) * dlog_gbar_dlog_r
    
    return dgbar_dr