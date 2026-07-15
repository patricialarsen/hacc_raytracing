#!/bin/bash/python

import os
import sys
os.environ["OMP_NUM_THREADS"] = "128"
import time
import numpy as np
import healpy as hp
import numpy as np

from write_funcs import write_checkpoint_dir, write_outputs, restart_from_checkpoint, write_native_state

from angle_updates import warmup_numba_kernels_beta
from transport_xyz import warmup_numba_kernels_transport

from simulation import get_input_map_alms
from synthetics import get_synthetic_alms_CCL

from iterate_arrays import iterate_arrays

from utils import initialize_ray_state, maps_from_jacobian, timed, initialize_ray_state_restart
from utils import rotate_state_to_observer_basis, advance_ray_and_matrix_state, advance_chi_values

from hacc_sims import LJ_simulation, FrontierE_simulation, FrontierE_simulation_hydro
sim = LJ_simulation()


# additional born output at z=1, or separate born script?

idx_start = int(sys.argv[1])
restart = idx_start==1

write_checkpoints = True
write_all_rotated_steps = False
write_all_unrotated_steps = True
write_unrotated_step = []
write_rotated_step = [6, 13,23,36, 44, 63] 
nthreads = 128

# constants 
vc = 2.998e5 #km/s
G = 4.3011790220362e-09 # Mpc/h (Msun/h)^-1 (km/s)^2

chi_cmb = sim['chi_cmb']
nside = sim['nside']

# geometry definitions 
lmax = 3*nside-1


print('Setting up')

warmup_numba_kernels_beta()
warmup_numba_kernels_transport()

# # if starting from scratch, we initialize state 
theta0, phi0 = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))


# if restarting from previously stalled run, read in previous state 
if restart:
    state = initialize_ray_state_restart(sim)
    
    # Need U/gtheta/gphi from the completed current step, for updating to step_idx.
    if state.step_idx>sim['nplanes']:
        chi_av, alms_filtered = get_synthetic_alms_CCL(state.step_idx-1,  sim, lmax, nthreads)
    else:
        chi_av, alms_filtered = get_input_map_alms(state.step_idx-1, sim, lmax, nthreads, filter='wiener', ell_cut=int(2.5*nside), use_pixel_weights=False)
        
    gtheta2, gphi2, U11, U12, U22 = iterate_arrays(state.step_idx - 1, alms_filtered, lmax, state.theta, state.phi, nside, nthreads)
    del alms_filtered

else:
    state = initialize_ray_state(nside, lmax)
    theta0 = state.theta.copy()
    phi0 = state.phi.copy()


print("Ray tracing over ", sim['nplanes'], " planes, starting with step ", state.step_idx, flush=True)    
while state.step_idx<sim['nplanes'] + sim['n_steps_cmb']:
    print('step_idx',state.step_idx, flush=True)
    
    step_t0 = time.perf_counter()
   
    with timed(f"step {state.step_idx} read/input map"):
        if state.step_idx>sim['nplanes']-1:
            chi_av, alms_filtered = get_synthetic_alms_CCL(state.step_idx, sim['path_in'], sim, lmax, nthreads)
        else:
            chi_av, alms_filtered = get_input_map_alms(state.step_idx, sim, lmax, nthreads, filter='wiener', ell_cut=int(2.5*nside), use_pixel_weights=False)
    chi_kp1 = chi_av

    if state.step_idx>0:
        with timed(f"step {state.step_idx} update beta and matrix"):
            advance_ray_and_matrix_state( state, gtheta2, gphi2, U11, U12, U22, chi_kp1 )

    w_shell = (chi_cmb - chi_av)/ chi_cmb
    state.kappa_born_alm += w_shell * alms_filtered

    print('iterating arrays')
    with timed(f"step {state.step_idx} iterate arrays / SHTs"):
        gtheta2, gphi2, U11, U12, U22 = iterate_arrays(state.step_idx, alms_filtered,lmax, state.theta, state.phi, nside, nthreads)
        del alms_filtered

    advance_chi_values(state, chi_kp1)


    if write_checkpoints:
        with timed(f"step {state.step_idx} checkpoint writes"):
            kappa_map, shear1_map, shear2_map, w_map = maps_from_jacobian(state.A_11, state.A_12, state.A_21, state.A_22)    
            write_checkpoint_dir( state.step_idx, sim, kappa_map, shear1_map, shear2_map, w_map, state.theta, state.phi, state.psi, state.kappa_born_alm)
            
    if write_all_unrotated_steps or (state.step_idx in write_unrotated_step):
        with timed(f"step {state.step_idx} write native ray state"):
            write_native_state(
                state.step_idx, sim,
                state.A_11, state.A_12, state.A_21, state.A_22,
                state.theta, state.phi, state.psi,)

    
    if write_all_rotated_steps or (state.step_idx in write_rotated_step):
 
        with timed(f"step {state.step_idx} output-basis rotation"):
            kappa_map, shear1_map, shear2_map, w_map = rotate_state_to_observer_basis(state, theta0, phi0)
            kappa_born = hp.alm2map(state.kappa_born_alm, nside=nside, lmax=lmax, mmax=lmax, pixwin=True)
            write_outputs(state.step_idx, sim, kappa_map, shear1_map, shear2_map, w_map, kappa_born=kappa_born)

    print(f"[timer] step {state.step_idx} total: {time.perf_counter() - step_t0:.2f} s", flush=True)

    print('finished writing stuff')
    state.step_idx +=1 

cmb_convert=True
if cmb_convert:
    advance_ray_and_matrix_state( state, gtheta2, gphi2, U11, U12, U22, chi_cmb )
    kappa_map, shear1_map, shear2_map, w_map = rotate_state_to_observer_basis(state, theta0, phi0)

    kappa_born_cmb = hp.alm2map(state.kappa_born_alm, nside=nside, lmax=lmax, mmax=lmax, pixwin=True)

    write_outputs(state.step_idx, sim, kappa_map, shear1_map, shear2_map, w_map, CMB=True, kappa_born=kappa_born_cmb) # this one should use CMB=True

    
