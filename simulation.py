#!/bin/bash/python

import os
import sys
import numpy as np
import healpy as hp
import numpy as np
import h5py
from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy import units as u
import pyccl as ccl
from filtering import wiener_filter, window_filter, wiener_filter_withtaper
import lenspyx
import math 
from hacc_sims import LJ_simulation as sim
from hacc_sims import step2z
from seam_correction import correct_density_sheet_y0
from sht_utils import map2alm_ducc_lsmr, map2alm_ducc_lsmr_from_weighted_adjoint

from contextlib import contextmanager
import time

@contextmanager
def timed_rank(label, comm):
    t0 = time.perf_counter()
    rank = comm.Get_rank()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"[timer] {label}: {dt:.2f} s, on rank {rank:d}", flush=True)


# constants 
vc = 2.998e5 #km/s
G = 4.3011790220362e-09 # Mpc/h (Msun/h)^-1 (km/s)^2
z_cmb = 1089.0


def get_chi_step(step_idx, sim):
    """Covoming distance bounds and average of step in Mpc/h"""
    if step_idx>sim['nplanes']-1:
        step_idx_new = step_idx - sim['nplanes']
        if step_idx_new >= sim["n_steps_cmb"]:
            raise IndexError("CMB step index out of range")
            
        chi_steps = np.linspace(sim['sim_chi_max'], sim['chi_cmb'], sim['n_steps_cmb']+1)
        chi_avs = (chi_steps[1:]+chi_steps[:-1])/2. #Mpc/h
        return chi_steps[step_idx_new], chi_steps[step_idx_new+1], chi_avs[step_idx_new]
    else:
        z_min = step2z(sim['step_list_min'][step_idx]-1, sim['zfin'], sim['zinit'], sim['nsteps'])
        z_max = step2z(sim['step_list_max'][step_idx]-1, sim['zfin'], sim['zinit'], sim['nsteps'])
        chi_min = ccl.background.comoving_radial_distance(sim['cosmo_ccl'],a=1./(z_min+1))*sim['h']  
        chi_max = ccl.background.comoving_radial_distance(sim['cosmo_ccl'],a=1./(z_max+1))*sim['h'] 
        chi_av = (chi_min+chi_max)/2.
    return chi_min, chi_max, chi_av

def randomize_alm_phases_preserve_power(alm, lmax, seed=12345, ell_min=2):
    """
    Preserve |a_lm| mode-by-mode, randomize phases.
    This preserves measured C_l exactly, but destroys phase correlations.
    """
    rng = np.random.default_rng(seed)
    out = np.empty_like(alm)

    # m = 0 modes must remain real for a real map.
    for ell in range(lmax + 1):
        idx = hp.Alm.getidx(lmax, ell, 0)
        if ell < ell_min:
            out[idx] = 0.0
        else:
            sign = 1.0 if rng.random() < 0.5 else -1.0
            out[idx] = sign * abs(alm[idx])

    # m > 0 complex modes: keep amplitude, randomize phase.
    for m in range(1, lmax + 1):
        n_this_m = lmax - m + 1
        idx0 = hp.Alm.getidx(lmax, m, m)
        idx = idx0 + np.arange(n_this_m)

        ell = np.arange(m, lmax + 1)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=n_this_m)

        vals = np.abs(alm[idx]) * np.exp(1j * phase)
        vals[ell < ell_min] = 0.0
        out[idx] = vals

    return out

def downgrade_nested_surface_density(map_nest, nside_out):
    nside_in = hp.npix2nside(map_nest.size)
    ratio = nside_in // nside_out

    if nside_in % nside_out != 0:
        raise ValueError("nside_out must divide nside_in")

    nchild = ratio * ratio
    npix_out = hp.nside2npix(nside_out)

    return map_nest.reshape(npix_out, nchild).mean(axis=1)



def get_input_alm_chi_z(step_idx, sim, comm=None):
    """ Read input density map and scale to convergence"""

    if not sim['has_alms']:
        raise ValueError("No known alm outputs, yet get_input_alm is called")
        
    chi_min, chi_max, chi_av = get_chi_step(step_idx, sim)
    z_av = z_at_value(sim['cosmo'].comoving_distance, chi_av/sim['h']*u.Mpc, method='bounded',zmin=0,zmax=10)

    with timed_rank("Reading alm data", comm):
        alm_read = hp.read_alm(sim['path_alms'] + '/alm_map_'+str(step_idx)+'.fits')

    if sim['name']=="Frontier-E":
        alm_read  /= sim["mpp"]
        alm_read[0] = 0.0
    elif sim['name']=="Frontier-E (hydro)":
        fb_fact = (1-sim['fb'])**2 + sim['fb']**2
        alm_read  /= sim["mpp"]
        alm_read /= fb_fact
        alm_read[0] = 0.0

    n_per_steradian = float(np.loadtxt(sim['output_path_alms'] +'n_perst_'+str(step_idx)+'.txt'))
    #n_per_steradian = sim['nperst'][step_idx]
    if sim['name']=="Frontier-E (hydro)":
        # n_per_steradian = mean()/ mpp / fb_fact
        fb_fact = (1-sim['fb'])**2 + sim['fb']**2
        kappa_fac = 4.0*np.pi*G/vc**2*(1.+z_av)/chi_av *sim['mpp'] * n_per_steradian  * fb_fact
    else:
        kappa_fac = 4.0*np.pi*G/vc**2*(1.+z_av)/chi_av *sim['mpp'] * n_per_steradian
    
    return alm_read, chi_av, kappa_fac, n_per_steradian

def precompute_n_perst(step_idx,  sim, comm=None):
    """ Read input density map and scale to convergence"""

    chi_min, chi_max, chi_av = get_chi_step(step_idx, sim)
    z_av = z_at_value(sim['cosmo'].comoving_distance, chi_av/sim['h']*u.Mpc, method='bounded',zmin=0,zmax=10)

    if sim['name']=="Last Journey":
        map_read = np.fromfile(sim['path_maps']+'density_'+str(sim['step_list_max'][step_idx])+'_'+str(sim['step_list_min'][step_idx])+'.bin','<f8')
        dist_pix = sim['signed_dist_pix']
        map_read, eps = correct_density_sheet_y0(map_read, dist_pix, sim['A_array'][step_idx], sim['w_array'][step_idx], max_dist=20)

    if sim['name']=="Frontier-E":
        with timed_rank("Reading in data", comm):
            map_read = h5py.File(sim['path_maps']+'GO_map_' +str(sim['step_list_max'][step_idx])+'_'+str(int(sim['step_list_min'][step_idx]-1))+'_dens.hdf5','r')
            map_read = map_read['rho'][:]

    if sim['name']=="Frontier-E (hydro)":
        with timed_rank("Reading in data", comm):
            map_read = h5py.File(sim['path_maps']+'hydro_map_' +str(sim['step_list_max'][step_idx])+'_'+str(int(sim['step_list_min'][step_idx]-1))+'_dens.hdf5','r')
            map_read = map_read['rho'][:]


    if sim['name']=="Last Journey":
        n_per_steradian = np.mean(map_read)
    if sim['name']== "Frontier-E (hydro)":
        fb_fact = (1-sim['fb'])**2 + sim['fb']**2
        n_per_steradian = np.mean(map_read)/ fb_fact /sim['mpp'] #(1-f_b)^2 + fb^2
    else:
        n_per_steradian = np.mean(map_read)/sim['mpp']

    np.savetxt(sim['output_path_alms'] +'n_perst_'+str(step_idx)+'.txt', np.array([n_per_steradian]))

    return 

    
def get_input_map_chi_z(step_idx,  sim, comm=None):
    """ Read input density map and scale to convergence"""

    chi_min, chi_max, chi_av = get_chi_step(step_idx, sim)
    z_av = z_at_value(sim['cosmo'].comoving_distance, chi_av/sim['h']*u.Mpc, method='bounded',zmin=0,zmax=10)
    
    if sim['name']=="Last Journey":
        map_read = np.fromfile(sim['path_maps']+'density_'+str(sim['step_list_max'][step_idx])+'_'+str(sim['step_list_min'][step_idx])+'.bin','<f8')
        dist_pix = sim['signed_dist_pix'] 
        map_read, eps = correct_density_sheet_y0(map_read, dist_pix, sim['A_array'][step_idx], sim['w_array'][step_idx], max_dist=20)
        
    if sim['name']=="Frontier-E":
        with timed_rank("Reading in data", comm):
            map_read = h5py.File(sim['path_maps']+'GO_map_' +str(sim['step_list_max'][step_idx])+'_'+str(int(sim['step_list_min'][step_idx]-1))+'_dens.hdf5','r')
            map_read = map_read['rho'][:]
        
    if sim['name']=="Frontier-E (hydro)": 
        with timed_rank("Reading in data", comm):
            map_read = h5py.File(sim['path_maps']+'hydro_map_' +str(sim['step_list_max'][step_idx])+'_'+str(int(sim['step_list_min'][step_idx]-1))+'_dens.hdf5','r')
            map_read = map_read['rho'][:]


    #if sim['nside']
    #map_read = downgrade_nested_surface_density(map_read, sim['nside'])    # safe for nside >8192
    with timed_rank("Reordering data", comm):
        map_read = hp.reorder(map_read,n2r=True)    # safe for nside >8192


    if sim['name']=="Last Journey":
        n_per_steradian = np.mean(map_read)
        kappa_fac = 4.0*np.pi*G/vc**2*(1.+z_av)/chi_av *sim['mpp'] * n_per_steradian
    else:
        # otherwise maps are in units of Msun/h/steradian
        if sim['name']== "Frontier-E (hydro)":
            fb_fact = (1-sim['fb'])**2 + sim['fb']**2
            n_per_steradian = np.mean(map_read)/ fb_fact /sim['mpp'] #(1-f_b)^2 + fb^2
        else:
            n_per_steradian = np.mean(map_read)/sim['mpp'] 
        kappa_fac = 4.0*np.pi*G/vc**2*(1.+z_av)/chi_av * np.mean(map_read)
        
    return map_read, chi_av, kappa_fac, n_per_steradian



def precompute_input_map_alms(step_idx, sim, lmax, nthreads, comm=None):
    """ Read input density map and scale to convergence"""
    map_read,  chi_av, kappa_fac, n_per_steradian = get_input_map_chi_z(step_idx, sim, comm=comm)

    np.savetxt(sim['output_path_alms'] +'n_perst_'+str(step_idx)+'.txt', np.array([n_per_steradian]))
    with timed_rank("Computing alms", comm):
        map_read = map_read/np.mean(map_read)
        map_read = map_read -1 
        alms_lens = map2alm_ducc_lsmr_from_weighted_adjoint(map_read, sim['nside'], lmax, nthreads)
    with timed_rank("Writing alms", comm):
        hp.write_alm(sim['output_path_alms'] + '/alm_map_'+str(step_idx)+'.fits', alms_lens, overwrite=True)

    return alms_lens


def get_input_map_alms(step_idx, sim, lmax, nthreads, filter='wiener', save_cls=False, use_pixel_weights=True, sn_taper=10.0, sn_end=5.0, min_taper_width=500, ell_cut=None, apply_pix_window=True, test_randomize_phase=False, test_synthetic_alms=False, save_alms=False, comm=None):
    """ Read input density map and scale to convergence"""
    if sim['has_alms'] and step_idx<sim['nalms_stored']:
        alms_lens, chi_av, kappa_fac, n_per_steradian = get_input_alm_chi_z(step_idx, sim, comm=comm)
    else:
        map_read,  chi_av, kappa_fac, n_per_steradian = get_input_map_chi_z(step_idx, sim, comm=comm)
        if sim['nside']<=8192:
            alms_lens = hp.map2alm((map_read/n_per_steradian - 1),lmax=lmax, mmax=lmax, iter=3, use_pixel_weights=use_pixel_weights, ) 
        else:
            with timed_rank("Computing alms", comm):
                map_read = map_read/np.mean(map_read)
                map_read = map_read -1 
                alms_lens = map2alm_ducc_lsmr_from_weighted_adjoint(map_read, sim['nside'], lmax, nthreads)

    cl_lens = hp.alm2cl(alms_lens)
    if save_cls:
        np.savetxt(sim['output_path_cls']+'cl_file_'+str(step_idx)+'.txt', cl_lens)
    if save_alms:
        if (not sim['has_alms']) or (step_idx>=sim['nalms_stored']):
            with timed_rank("Writing alms", comm):
                hp.write_alm(sim['output_path_alms'] + '/alm_map_'+str(step_idx)+'.fits', alms_lens, overwrite=True)

    if test_synthetic_alms:
        alms_lens = hp.synalm(cl_lens, lmax=lmax, mmax=lmax, new=True,)
        filter=None

    
    if filter=='wiener':
        alms_filtered = wiener_filter(alms_lens, n_per_steradian, lmax, sim['nside'], apply_pix_window=apply_pix_window, ell_cut=ell_cut, datapath=sim['pixwin_datapath'])
    elif filter=='wiener_tapered':
        alms_filtered = wiener_filter_withtaper(alms_lens, n_per_steradian, lmax, sim['nside'], sn_taper=sn_taper, sn_end=sn_end, min_taper_width=min_taper_width, ell_cut=ell_cut, apply_pix_window=apply_pix_window, datapath=sim['pixwin_datapath'])
    elif filter=='window':
        alms_filtered = window_filter(alms_lens,  lmax, sim['nside'], ell_cut=ell_cut, datapath=sim['pixwin_datapath'])
    else:
        alms_filtered = alms_lens

    alms_filtered = kappa_fac * alms_filtered

    if test_randomize_phase: 
        alms_filtered = randomize_alm_phases_preserve_power( alms_filtered,lmax=lmax,seed=12345 + step_idx,)


    return chi_av, alms_filtered




def create_gauss_map_chi_z(read_path, run_fresh=False, nside_out = 8192):
    """ Read in, or optionally create high redshift plane """ 
    # note - need to update this to allow for multi-plane lensing at high z 
    # edit this - 46 is 9.5 so check the exact bounds

    if not run_fresh:
        map_hz = hp.read_map(read_path)
    else:
        max_step_bound = sim['step_list_max'][-1]-1

        # simulation dependent 
        z_min = step2z(max_step_bound, sim['zfin'], sim['zinit'], sim['nsteps'])

        #Last Journey/ Frontier-E cosmology
        chi_cmb = ccl.background.comoving_radial_distance(sim['cosmo_ccl'],a=1./(z_cmb+1))
        chi_max = ccl.background.comoving_radial_distance(sim['cosmo_ccl'],a=1./(z_min+1))
        cmbk = ccl.CMBLensingTracer(sim['cosmo_ccl'], z_source=z_cmb)

        chi = np.linspace(0.0, chi_cmb, 10000)
        kern = cmbk.get_kernel(chi = chi)

        # 3* nside - 1
        lmax = int(sim['nside']*3-1)
        ells = np.arange(lmax)
        
        cmb_tracer_hz = ccl.Tracer()
        cmb_tracer_hz.add_tracer(sim['cosmo_ccl'], kernel = (chi[chi>=chi_max], kern[0][chi>=chi_max]),der_bessel=-1,der_angles=1)
        cl_vals_hz = ccl.angular_cl(sim['cosmo_ccl'],cmb_tracer_hz, cmb_tracer_hz, ells)

        map_hz = hp.sphtfunc.synfast(cl_vals_hz,pol=False, nside=nside_out, pixwin=False)
        hp.write_map(read_path, map_hz, overwrite=False)
        
    return map_hz

