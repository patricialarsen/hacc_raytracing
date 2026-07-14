from scipy.integrate import cumulative_trapezoid
from numpy.polynomial.legendre import legval
import numpy as np
import healpy as hp

def point_lens_theory(
    lmax,
    fwhm_deg,
    kappa_peak,
    chi_lens,
    chi_source,
    ntheta=None,
):
    if ntheta is None:
        ntheta = max(8192, 4 * lmax)

    ell = np.arange(lmax + 1)
    fwhm = np.deg2rad(fwhm_deg)
    beam = hp.gauss_beam(fwhm=fwhm, lmax=lmax)

    
    # Match the injected alm normalization: ell=0 excluded.
    norm = np.sum((2.0 * ell[1:] + 1.0) * beam[1:]) / (4.0 * np.pi)
    K = kappa_peak / norm

    theta = np.linspace(1.0e-8, np.pi - 1.0e-8, ntheta)

    # kappa_L(theta) = K/(4 pi) sum_l (2l+1) B_l P_l(cos theta)
    coeff = np.zeros(lmax + 1)
    coeff[1:] = K * (2.0 * ell[1:] + 1.0) * beam[1:] / (4.0 * np.pi)
    kappa_lens = legval(np.cos(theta), coeff)

    integral = cumulative_trapezoid(
        kappa_lens * np.sin(theta), theta, initial=0.0
    )
    alpha_lens = 2.0 * integral / np.sin(theta)

    gamma_t_lens = alpha_lens / np.tan(theta) - kappa_lens

    W = (chi_source - chi_lens) / chi_source

    return {
        "theta": theta,
        "kappa": W * kappa_lens,
        "gamma_t": W * gamma_t_lens,
        "alpha": W * alpha_lens,
        "beta_theta": theta - W * alpha_lens,
    }


def compute_ps_theory(lmax, fwhm_deg, kappa_peak, chi_source, chi_lens):
    ell = np.arange(lmax + 1)

    beam = hp.gauss_beam(np.deg2rad(fwhm_deg), lmax=lmax)
    norm = np.sum((2.0 * ell[1:] + 1.0) * beam[1:]) / (4.0 * np.pi)
    K = kappa_peak / norm

    W = (chi_source - chi_lens) / chi_source

    cl_kk_th = np.zeros(lmax + 1)
    cl_kk_th[1:] = W**2 * K**2 * beam[1:]**2 / (4.0 * np.pi)

    shear_factor = np.zeros(lmax + 1)
    shear_factor[2:] = (
        (ell[2:] + 2.0) * (ell[2:] - 1.0)
        / (ell[2:] * (ell[2:] + 1.0))
    )

    cl_ee_th = shear_factor * cl_kk_th
    cl_bb_th = np.zeros_like(cl_kk_th)
    cl_ww_th = np.zeros_like(cl_kk_th)
    return cl_ee_th, cl_bb_th, cl_ww_th