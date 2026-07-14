#!/bin/bash/python

import os
import sys
os.environ["OMP_NUM_THREADS"] = "8"
import time
import numpy as np
import healpy as hp

import matplotlib.pyplot as plt

from utils import initialize_ray_state, maps_from_jacobian, timed
from utils import rotate_state_to_observer_basis, advance_ray_and_matrix_state, advance_chi_values

from write_funcs import  write_outputs
from angle_updates import warmup_numba_kernels_beta
from transport_xyz import warmup_numba_kernels_transport

from synthetics import get_synthetic_alms_gaussian_point, get_empty_alms, get_synthetic_alms_gaussian_point_alm_north, rotate_north_lens_alm
from hacc_sims import test_simulation  
from iterate_arrays import iterate_arrays

from ref_profiles import point_lens_theory, compute_ps_theory


sim = test_simulation()
nthreads = 8
nside = sim['nside']
ra = 0.0; dec = 0.0;
chi_source = sim['chi_source']
fwhm_deg = 0.1
amp = 0.01

theta_lens = np.pi / 2.0
phi_lens = 0.7

# geometry definitions 
lmax = 3*nside-1

print('Setting up')

warmup_numba_kernels_beta()
warmup_numba_kernels_transport()

state = initialize_ray_state(nside, lmax)
theta0 = state.theta.copy()
phi0 = state.phi.copy()

print("Running single point source test at step 1", flush=True)    

while state.step_idx<sim['nplanes']:
    print('step_idx',state.step_idx, flush=True)
    
    step_t0 = time.perf_counter()
   
    with timed(f"step {state.step_idx} read/input map"):
        if state.step_idx==1:
            
            #chi_av, alms_filtered = get_synthetic_alms_gaussian_point_alm(lmax,fwhm_deg, amp, theta_lens, phi_lens, step_idx=state.step_idx, sim=sim)
            chi_av, alms_filtered = get_synthetic_alms_gaussian_point_alm_north(lmax,fwhm_deg, kappa_peak=amp, step_idx=state.step_idx, sim=sim)
            alms_filtered = rotate_north_lens_alm(alms_filtered, lmax, theta_lens, phi_lens)
            chi_lens = chi_av
        else:
            chi_av, alms_filtered = get_empty_alms(nside, step_idx=state.step_idx, sim=sim)
        
        chi_kp1 = chi_av
        
    if state.step_idx>0:
        with timed(f"step {state.step_idx} update beta and matrix"):
            advance_ray_and_matrix_state( state, gtheta2, gphi2, U11, U12, U22, chi_kp1 )

    with timed(f"step {state.step_idx} iterate arrays / SHTs"):
        gtheta2, gphi2, U11, U12, U22 = iterate_arrays(state.step_idx, alms_filtered,lmax, state.theta, state.phi, nside, nthreads)
        del alms_filtered

    advance_chi_values(state, chi_kp1)
    print(f"[timer] step {state.step_idx} total: {time.perf_counter() - step_t0:.2f} s", flush=True)

    state.step_idx +=1 

advance_ray_and_matrix_state( state, gtheta2, gphi2, U11, U12, U22, chi_source )
kappa_map, shear1_map, shear2_map, w_map = rotate_state_to_observer_basis(state, theta0, phi0)






theory = point_lens_theory( nside*3-1, fwhm_deg, amp, chi_lens, chi_source,)





n_lens = np.asarray(hp.ang2vec(theta_lens, phi_lens))
nx, ny, nz = hp.pix2vec(nside, np.arange(hp.nside2npix(nside)))
n_obs = np.stack([nx, ny, nz], axis=0)

sep0 = np.arccos(np.clip(
    n_lens[0] * nx + n_lens[1] * ny + n_lens[2] * nz,
    -1.0,
    1.0,
))

kappa_th = np.interp(sep0, theory["theta"], theory["kappa"])
gamma_t_th = np.interp(sep0, theory["theta"], theory["gamma_t"])
alpha_th = np.interp(sep0, theory["theta"], theory["alpha"])
#beta_theta_th = np.interp(sep0, theory["theta"], theory["beta_theta"])


sin_sep = np.maximum(np.sin(sep0), 1e-14)
toward_lens = (n_lens[:, None] - np.cos(sep0)[None, :] * n_obs) / sin_sep

n_beta_th = (
    np.cos(alpha_th)[None, :] * n_obs
    + np.sin(alpha_th)[None, :] * toward_lens
)

# n_beta_th has shape (3, npix)
norm = np.sqrt(np.sum(n_beta_th**2, axis=0))
n_beta_th /= norm

theta_beta_th = np.arccos(np.clip(n_beta_th[2], -1.0, 1.0))
phi_beta_th = np.mod(
    np.arctan2(n_beta_th[1], n_beta_th[0]),
    2.0 * np.pi,
)

dtheta = state.theta - theta_beta_th
dphi = (state.phi - phi_beta_th + np.pi) % (2.0 * np.pi) - np.pi

pix_size = np.sqrt(4.0 * np.pi / hp.nside2npix(nside))
good = (
    (sep0 > 3.0 * pix_size)
    & (sep0 < np.deg2rad(5.0))
)

print("theta source-map rms", np.std(dtheta[good]))
print("phi source-map rms", np.std(dphi[good]))
print("theta source-map max", np.max(np.abs(dtheta[good])))
print("phi source-map max", np.max(np.abs(dphi[good])))




ct = np.cos(theta0)
st = np.sin(theta0)
cp = np.cos(phi0)
sp = np.sin(phi0)

e_theta = np.stack([ct * cp, ct * sp, -st], axis=0)
e_phi = np.stack([-sp, cp, np.zeros_like(theta0)], axis=0)

cosvar = np.sum(toward_lens * e_theta, axis=0)
sinvar = np.sum(toward_lens * e_phi, axis=0)
varphi = np.arctan2(sinvar, cosvar)

shear1_th = -gamma_t_th * np.cos(2.0 * varphi)
shear2_th = gamma_t_th * np.sin(2.0 * varphi)
w_th = np.zeros_like(shear1_th)

c2 = np.cos(2.0 * varphi)
s2 = np.sin(2.0 * varphi)

gamma_t_num = -shear1_map * c2 + shear2_map * s2
gamma_x_num =  shear1_map * s2 + shear2_map * c2

plt.figure()
hp.mollview(gamma_t_num)
plt.savefig('gamma_t_num.png')

plt.figure()
hp.mollview(gamma_t_th)
plt.savefig('gamma_t_th.png')

print("kappa relative rms",
      np.std(kappa_map[good] - kappa_th[good]) / np.std(kappa_th[good]))
print("shear1 relative rms",
      np.std(shear1_map[good] - shear1_th[good]) / np.std(shear1_th[good]))
print("shear2 relative rms",
      np.std(shear2_map[good] - shear2_th[good]) / np.std(shear2_th[good]))
print("gammat relative rms",
      np.std(gamma_t_num[good] - gamma_t_th[good]) / np.std(gamma_t_th[good]))

print("gammax rms", np.std(gamma_x_num[good]))

print("w rms", np.std(w_map[good]))


analysis_lmax = min(lmax, int(2.5 * nside))

# Scalar convergence and rotation.
kappa_alm_out = hp.map2alm(
    kappa_map,
    lmax=analysis_lmax,
    mmax=analysis_lmax,
    iter=3,
    use_pixel_weights=True,
)

w_alm_out = hp.map2alm(
    w_map,
    lmax=analysis_lmax,
    mmax=analysis_lmax,
    iter=3,
    use_pixel_weights=True,
)

# Your stored shear2 is -gamma2 in Healpy's spin convention.
e_alm_out, b_alm_out = hp.map2alm_spin(
    [shear1_map, -shear2_map],
    spin=2,
    lmax=analysis_lmax,
    mmax=analysis_lmax,
)

cl_kk_out = hp.alm2cl(kappa_alm_out)
cl_ee_out = hp.alm2cl(e_alm_out)
cl_bb_out = hp.alm2cl(b_alm_out)
cl_ww_out = hp.alm2cl(w_alm_out)

cl_ee_th, cl_bb_th, cl_ww_th = compute_ps_theory(
    analysis_lmax,
    fwhm_deg,
    amp,
    chi_source,
    chi_lens,
)

ell = np.arange(analysis_lmax + 1)

# Avoid meaningless ratios after the Gaussian beam has made the theory tiny.
good = (
    (ell >= 2)
    & (cl_ee_th > 1.0e-12 * np.max(cl_ee_th))
)

plt.figure(figsize=(8, 5))
plt.loglog(ell[good], cl_ee_out[good], label="EE measured")
plt.loglog(ell[good], cl_ee_th[good], "--", label="EE theory")
plt.loglog(ell[good], cl_bb_out[good], label="BB measured")
plt.loglog(ell[good], cl_ww_out[good], label="ww measured")
plt.xlabel(r"$\ell$")
plt.ylabel(r"$C_\ell$")
plt.legend()
plt.tight_layout()
plt.savefig('test1.png')

ee_frac_resid = cl_ee_out[good] / cl_ee_th[good] - 1.0
bb_frac = cl_bb_out[good] / cl_ee_th[good]
ww_frac = cl_ww_out[good] / cl_ee_th[good]

plt.figure(figsize=(8, 4))
plt.semilogx(ell[good], ee_frac_resid)
plt.axhline(0.0, color="k", ls="--", lw=1)
plt.xlabel(r"$\ell$")
plt.ylabel(r"$C_\ell^{EE}/C_{\ell,\rm th}^{EE} - 1$")
plt.tight_layout()
#plt.ylim([-1e-5,1e-5])
plt.savefig('test2.png')

plt.figure(figsize=(8, 4))
plt.semilogx(ell[good], bb_frac, label="BB / EE theory")
plt.semilogx(ell[good], ww_frac, label="ww / EE theory")
plt.axhline(0.0, color="k", ls="--", lw=1)
plt.xlabel(r"$\ell$")
plt.ylabel("null power / EE theory")
#plt.ylim([-1e-5,1e-5])
plt.legend()
plt.tight_layout()
plt.savefig('test3.png')

