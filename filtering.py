import numpy as np
import healpy as hp
from scipy.ndimage import gaussian_filter1d

def smooth_cl_log_ell(cl, lmin=2, ngrid=4096, sigma_grid=25, floor=1e-300):
    cl = np.asarray(cl, dtype=np.float64)
    lmax = cl.size - 1
    ell = np.arange(lmax + 1)

    good = (ell >= lmin) & (cl > 0)
    x = np.log(ell[good])
    y = np.log(np.maximum(cl[good], floor))

    xgrid = np.linspace(np.log(lmin), np.log(lmax), ngrid)
    ygrid = np.interp(xgrid, x, y)

    ygrid_smooth = gaussian_filter1d(ygrid, sigma=sigma_grid, mode="reflect")

    out = np.zeros_like(cl)
    target = np.log(np.maximum(ell[lmin:], lmin))
    out[lmin:] = np.exp(np.interp(target, xgrid, ygrid_smooth))
    return out


def wiener_filter(alms_lens, n_per_steradian, lmax, nside, apply_pix_window=True, ell_cut=None, datapath=None):
    """ Wiener filter to filter out shot noise """
    cl_lens = hp.alm2cl(alms_lens, lmax=lmax)
    shot_noise = 1.0/n_per_steradian

    cl_signal_obs = np.maximum(cl_lens - shot_noise, 0.0)
    cl_sig_smooth = smooth_cl_log_ell(cl_signal_obs)

    filt = np.zeros_like(cl_lens)
    
    if apply_pix_window:
        window = hp.pixwin(nside, lmax=lmax, datapath=datapath)
    else:
        window = np.ones_like(filt)
    
    good = window > 1e-8
    filt[good] = (cl_sig_smooth[good] / window[good]) / np.maximum(cl_sig_smooth[good] + shot_noise, 1.e-30) 
    
    if ell_cut is not None:
        ell = np.arange(filt.size)
        filt[ell > ell_cut] = 0.0

    alms_filtered = hp.almxfl(alms_lens,filt, mmax=lmax)
    return alms_filtered

def wiener_filter_withtaper(alms_lens, n_per_steradian, lmax, nside, sn_taper=10.0, sn_end=5.0, min_taper_width=0.0, ell_cut=None, apply_pix_window=True, datapath=None):
    """ Wiener filter to filter out shot noise, with an additional signal to noise tapering, 
    cutting out scales where the noise strongly dominates the signal and the recovered Cl gets
    noisy """

    cl_lens = hp.alm2cl(alms_lens, lmax=lmax)
    shot_noise = 1.0/n_per_steradian

    cl_signal_obs = np.maximum(cl_lens - shot_noise, 0.0)
    cl_sig_smooth = smooth_cl_log_ell(cl_signal_obs)

    filt = np.zeros_like(cl_lens)
    
    if apply_pix_window:
        window = hp.pixwin(nside, lmax=lmax, datapath=datapath)
    else:
        window = np.ones_like(filt)

    sn = cl_sig_smooth / shot_noise
    ell = np.arange(sn.size)

    # here, we're making sure that the signal to noise drop is past the peak, so we're 
    # not finding noisy large-scales 
    search_min_ell = 100
    ell_peak = np.argmax(np.where(ell >= search_min_ell, sn, -np.inf))
    
    taper = np.ones_like(sn, dtype=np.float64)
    
    # If the final multipole is still above the zero threshold, keep everything.
    if sn[-1] < sn_end:
        above_keep = np.where(sn >= sn_taper)[0]
        above_zero = np.where(sn >= sn_end)[0]

        if above_keep.size > 0 and above_zero.size > 0:
            ell_start = above_keep[-1]
            ell_end_raw = above_zero[-1]

            ell_end_raw = max(ell_end_raw, ell_start)
            ell_end = max(ell_end_raw, ell_start + min_taper_width)
            ell_end = min(ell_end, sn.size - 1)

            if ell_start < sn.size - 1:
                x = (ell - ell_start) / max(ell_end - ell_start, 1)
                mid = (ell >= ell_start) & (ell < ell_end)

                taper[mid] = 0.5 * (1.0 + np.cos(np.pi * x[mid]))
                taper[ell >= ell_end] = 0.0

    # Build the filter from the smoothed signal estimate.
    good = window > 1e-8
    filt[good] = (
        (cl_sig_smooth[good] / window[good])
        / np.maximum(cl_sig_smooth[good] + shot_noise, 1.e-30)
    )

    filt *= taper
    if ell_cut is not None:
        ell = np.arange(filt.size)
        filt[ell > ell_cut] = 0.0

    alms_filtered = hp.almxfl(alms_lens,filt, mmax=lmax)

    return alms_filtered

def window_filter(alms_lens,  lmax, nside, ell_cut=None, datapath=None):
    """ filter accounting for the pixel window function only """
    window = hp.pixwin(nside, lmax=lmax, datapath=datapath)
    good = window > 1e-8

    filt = np.ones_like(window)
    filt[good] = 1.0/window[good]    
    
    if ell_cut is not None:
        ell = np.arange(filt.size)
        filt[ell > ell_cut] = 0.0

    alms_filtered = hp.almxfl(alms_lens,filt, mmax=lmax)
    return alms_filtered

