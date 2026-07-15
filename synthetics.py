#!/bin/bash/python

import os
import sys
import numpy as np
import healpy as hp
import numpy as np
from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy import units as u
import pyccl as ccl
from filtering import wiener_filter, window_filter, wiener_filter_withtaper
import lenspyx
import math 
from scipy.special import sph_harm_y
from seam_correction import correct_density_sheet_y0
from simulation import get_chi_step



# constants 
vc = 2.998e5 #km/s
G = 4.3011790220362e-09 # Mpc/h (Msun/h)^-1 (km/s)^2
z_cmb = 1089.0


def get_synthetic_alms_CCL(step_idx, sim, lmax, nthreads,  filter='window', z_source=None, seed0=12345, use_map=False, use_pixel_weights=True, sn_taper=10.0, sn_end=5.0, min_taper_width=500, ell_cut=None, apply_pix_window=True, add_noise=False, use_n_pers=False):
    """
    """
    chi_min, chi_max, chi_av = get_chi_step(step_idx, sim)
    z_av = z_at_value(sim['cosmo'].comoving_distance, chi_av/sim['h']*u.Mpc, method='bounded',zmin=0,zmax=1100.)

    if z_source is None:
        z_source = sim["z_cmb"]

    # create pyccl overdensity field power spectrum 
    ells = np.arange(lmax + 1)
    chi = np.linspace(chi_min/sim['h'], chi_max/sim['h'], 10000)
    pchi = chi**2
    pchi /= np.trapezoid(pchi, chi)
    tr_shell = ccl.Tracer()
    tr_shell.add_tracer( sim["cosmo_ccl"], kernel=(chi, pchi), der_bessel=0, der_angles=0,)
    cl = ccl.angular_cl(sim["cosmo_ccl"], tr_shell, tr_shell, ells)
    cl = np.asarray(cl, dtype=np.float64)
    cl[:2] = 0.0

    # create healpy synthetic alm 
    rng = np.random.default_rng(seed0 + step_idx)
    alm = hp.synalm(cl, lmax=lmax, mmax=lmax, new=True,)

    # determine multiplicative factor to go from overdensity to convergence
    kappa_fac = 4.0*np.pi*G/vc**2*(1.+z_av)/chi_av * sim['rho_m0'] * (chi_max**3 - chi_min**3) / 3.0

    if use_n_pers:
        # have stored n_per_steradian for first few maps to better mimic shot noise and density field for tests
        n_per_steradian = sim['nperst'][step_idx]
        kappa_fac = 4.0*np.pi*G/vc**2*(1.+z_av)/chi_av * n_per_steradian *sim['mpp']
    else:
        # otherwise estimate that from average density field and particle mass 
        n_per_steradian = sim['rho_m0'] * (chi_max**3 - chi_min**3) / 3.0/ sim['mpp']
    
    if use_map:   
        # synthesize map from alms
        map_syn = hp.alm2map(alm, nside=sim["nside"], lmax=lmax, pol=False, mmax=lmax, pixwin=True)
        npix = hp.nside2npix(sim['nside'])
        afac = 4.*np.pi/npix
        if add_noise:
            # add gaussian random noise at the map level 
            npix = hp.nside2npix(sim['nside'])
            afac = 4.*np.pi/npix
            sigma_pix = np.sqrt((1.0 / n_per_steradian) / afac)
            map_syn += rng.normal(scale=sigma_pix, size=map_syn.size)

        # go back to alm space and apply filter 
        alms_lens = hp.map2alm(map_syn, lmax=lmax, mmax=lmax, iter=3, use_pixel_weights=use_pixel_weights)
        if filter=='wiener':
            alms_filtered = wiener_filter(alms_lens, n_per_steradian, lmax, sim['nside'], apply_pix_window=apply_pix_window, ell_cut=ell_cut, datapath=sim['pixwin_datapath'])
        elif filter=='wiener_tapered':
            alms_filtered = wiener_filter_withtaper(alms_lens, n_per_steradian, lmax, sim['nside'], sn_taper=sn_taper, sn_end=sn_end, min_taper_width=min_taper_width, ell_cut=ell_cut, apply_pix_window=apply_pix_window, datapath=sim['pixwin_datapath'])
        elif filter=='window':
            alms_filtered = window_filter(alms_lens,  lmax, sim['nside'], ell_cut=ell_cut, datapath=sim['pixwin_datapath'])
        else:
            alms_filtered = alms_lens

        return chi_av, alms_filtered*kappa_fac


    return chi_av, alm*kappa_fac




def get_synthetic_alms_gaussian_point(nside, mass, ra, dec, fwhm_deg, scale_to_kappa=True, step_idx=None, sim=None):
    """ Apply a point source mass in map-space at given angular position, and smooth with a gaussian with given fwhm.
    For testing. """
    pix_ps = hp.ang2pix(nside, ra, dec, lonlat=True)
    omega_pix = 4.0 * np.pi / hp.nside2npix(nside)

    ps_map = np.zeros(hp.nside2npix(nside))
    ps_map[pix_ps] = mass / omega_pix
    ps_map= hp.smoothing(ps_map, fwhm=fwhm_deg * np.pi/180)
    ps_map = (ps_map-np.mean(ps_map))/np.mean(ps_map) # convert to overdensity  
    lmax = 3 * nside - 1
    alms_ps = hp.map2alm(ps_map, lmax=lmax, mmax=lmax, iter=3)

    if scale_to_kappa:
        chi_min, chi_max, chi_av = get_chi_step(step_idx, sim)
        z_av = z_at_value(sim['cosmo'].comoving_distance, chi_av/sim['h']*u.Mpc, method='bounded',zmin=0,zmax=1100.)
        kappa_fac = 4.0*np.pi*G/vc**2*(1.+z_av)/chi_av * sim['rho_m0'] * (chi_max**3 - chi_min**3) / 3.0
        return chi_av, alms_ps*kappa_fac

    return alms_ps


def get_synthetic_alms_gaussian_point_alm_north(lmax,fwhm_deg, kappa_peak=0.01, step_idx=None, sim=None):
    
    if step_idx is None or sim is None:
        raise ValueError("step_idx and sim are required to return chi_av")

    beam = hp.gauss_beam(fwhm= np.deg2rad(fwhm_deg), lmax=lmax)
    ell = np.arange(lmax + 1)

    # Choose the integrated amplitude K so that kappa(theta=0)=kappa_peak.
    norm = np.sum((2.0 * ell[1:] + 1.0) * beam[1:]) / (4.0 * np.pi)
    K = kappa_peak / norm

    alm = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)

    # kappa_l0 = K B_l Y_l0*(north pole)
    alm[hp.Alm.getidx(lmax, ell[1:], 0)] = (
        K
        * beam[1:]
        * np.sqrt((2.0 * ell[1:] + 1.0) / (4.0 * np.pi))
    )
    chi_min, chi_max, chi_av = get_chi_step(step_idx, sim)

    return chi_av, alm

def rotate_north_lens_alm(alm_north, lmax, theta_lens, phi_lens):
    # HEALPix Y convention: Z-Y-Z rotations.
    # (psi, theta, phi) = (0, theta_lens, phi_lens)
    rot = hp.Rotator(
        rot=[0.0, theta_lens, -phi_lens],
        deg=False,
        eulertype="Y",
        inv=False,
    )

    theta_check, phi_check = rot(0.0, 0.0)
    print("north pole mapped to:", theta_check, phi_check)

    return rot.rotate_alm(
        alm_north.copy(),
        lmax=lmax,
        mmax=lmax,
        inplace=False,
    )

def get_empty_alms(nside, step_idx=None, sim=None):
    """ Apply a point source mass in map-space at given angular position, and smooth with a gaussian with given fwhm.
    For testing. """
    lmax = nside*3-1
    kappa_alm = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
    chi_min, chi_max, chi_av = get_chi_step(step_idx, sim)

    return chi_av, kappa_alm




def get_synthetic_alms_quadrupole(nside, eps, output_chi=False, step_idx=None, sim=None):
    """ Quadrupole signal for testing """
    lmax = nside*3-1
    kappa_alm = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
    kappa_alm[hp.Alm.getidx(lmax, 2, 0)] = -3 * eps * np.sqrt(4.0 * np.pi / 5.0)
    if output_chi:
        chi_min, chi_max, chi_av = get_chi_step(step_idx, sim)
        return chi_av, kappa_alm
        
    return kappa_alm


