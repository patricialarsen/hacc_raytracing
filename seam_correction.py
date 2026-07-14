import numpy as np

def correct_density_sheet_y0(rho, signed_dist_pix, A, w, max_dist=20,):
    r = np.abs(signed_dist_pix)
    eps = A * np.exp(-r / w)
    eps[r > max_dist] = 0.0
    eps = np.clip(eps, 0.0, 0.5)
    rho_corr = rho / (1.0 - eps)

    return rho_corr, eps