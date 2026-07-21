#!/bin/bash/python

import os
import sys
os.environ["OMP_NUM_THREADS"] = "128"
import time
import numpy as np
import healpy as hp
import numpy as np

from write_funcs_mpi import write_checkpoint_dir, write_outputs, restart_from_checkpoint, write_native_state, write_outputs_parallel, write_native_state_parallel,write_checkpoint_dir_parallel

from angle_updates import warmup_numba_kernels_beta
from transport_xyz import warmup_numba_kernels_transport

from simulation import get_input_map_alms
from synthetics import get_synthetic_alms_CCL

from iterate_arrays import iterate_arrays

from utils import  maps_from_jacobian, timed, timed_rank
from utils import rotate_state_to_observer_basis, advance_ray_and_matrix_state, advance_chi_values

from utils_mpi import initialize_ray_state_restart, initialize_ray_state_chunked, print_mem_summary, print_mem_summary_rank

from hacc_sims import LJ_simulation, FrontierE_simulation, FrontierE_simulation_hydro


try:
    from mpi4py import MPI
except ImportError as exc:
    raise RuntimeError(
        "Optional mpi4py library not installed. "
        "Install mpi4py or run serial version. "
        ) from exc
    
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#sim = LJ_simulation()
sim = FrontierE_simulation()


# additional born output at z=1, or separate born script?

idx_start = int(sys.argv[1])
restart = idx_start==1

write_checkpoints = False
write_all_rotated_steps = False
write_all_unrotated_steps = False
write_unrotated_step = [6, 13, 23, 36, 44, 53]
write_rotated_step = [6, 13, 23, 36, 44, 53] 
nthreads = 128

output_born=False


# constants 
vc = 2.998e5 #km/s
G = 4.3011790220362e-09 # Mpc/h (Msun/h)^-1 (km/s)^2

chi_cmb = sim['chi_cmb']
nside = sim['nside']

# geometry definitions 
#lmax = 3*nside-1

lmax = int(2.0*nside)

npix = hp.nside2npix(nside)
n_lms = int((lmax+2)*(lmax+1)/2) # size of alm array
pix_min = rank * npix // size
pix_max = (rank + 1) * npix // size
npix_reduced = pix_max - pix_min


print('Setting up')
warmup_numba_kernels_beta()
warmup_numba_kernels_transport()

# # if starting from scratch, we initialize state 
theta0, phi0 = hp.pix2ang(nside, np.arange(pix_min, pix_max))





# if restarting from previously stalled run, read in previous state 
if restart:
    state = initialize_ray_state_restart(sim, pix_min, pix_max, comm)
    
    # Need U/gtheta/gphi from the completed current step, for updating to step_idx.
    if rank==0:
        if state.step_idx>sim['nplanes']:
            chi_av, alms_filtered = get_synthetic_alms_CCL(state.step_idx-1, sim, lmax, nthreads)
        else:
            chi_av, alms_filtered = get_input_map_alms(state.step_idx-1, sim, lmax, nthreads, filter='wiener', ell_cut=int(2.0*nside), use_pixel_weights=False, save_alms=True, comm=comm)
    else:
        alms_filtered = np.empty(n_lms, dtype=np.complex128)
        chi_av = np.empty((), dtype=np.float64)
        
    with timed_rank(f"communicaton",comm):
        chi_av = np.array(chi_av, dtype=np.float64)
        comm.Bcast(chi_av, root=0)
        comm.Bcast(alms_filtered, root=0)  
        chi_av = float(chi_av)
                
    gtheta2, gphi2, U11, U12, U22 = iterate_arrays(state.step_idx - 1, alms_filtered, lmax, state.theta, state.phi, nside, nthreads)
    del alms_filtered

else:
    with timed_rank(f"initialize",comm):
        state = initialize_ray_state_chunked(nside, pix_min, pix_max, lmax, output_born)
        theta0 = state.theta.copy()
        phi0 = state.phi.copy()

print_mem_summary_rank("after initialize", comm)


print("Ray tracing over ", sim['nplanes'], " planes, starting with step ", state.step_idx, flush=True)    
while state.step_idx<sim['nplanes'] + sim['n_steps_cmb']:
    print('step_idx',state.step_idx, flush=True)
    
    step_t0 = time.perf_counter()
   
    with timed_rank(f"step {state.step_idx} read/input map",comm):
        if rank==0:
            if state.step_idx>sim['nplanes']-1:
                chi_av, alms_filtered = get_synthetic_alms_CCL(state.step_idx, sim, lmax, nthreads)
            else:
                chi_av, alms_filtered = get_input_map_alms(state.step_idx, sim, lmax, nthreads, filter='wiener', ell_cut=int(2.0*nside), use_pixel_weights=False, save_alms=False, comm=comm)
        else:
            alms_filtered = np.empty(n_lms, dtype=np.complex128)
            chi_av = np.empty((), dtype=np.float64)
            
        chi_av = np.array(chi_av, dtype=np.float64)
        comm.Bcast(chi_av, root=0)
        comm.Bcast(alms_filtered, root=0)  
        chi_av = float(chi_av)

                
    chi_kp1 = chi_av

    if state.step_idx>0:
        with timed_rank(f"step {state.step_idx} update beta and matrix",comm):
            advance_ray_and_matrix_state( state, gtheta2, gphi2, U11, U12, U22, chi_kp1 )
        del gtheta2, gphi2, U11, U12, U22

    w_shell = (chi_cmb - chi_av)/ chi_cmb
    if rank==0 and output_born:
        state.kappa_born_alm += w_shell * alms_filtered

    print('iterating arrays')
    with timed_rank(f"step {state.step_idx} iterate arrays / SHTs", comm):
        gtheta2, gphi2, U11, U12, U22 = iterate_arrays(state.step_idx, alms_filtered,lmax, state.theta, state.phi, nside, nthreads)
        del alms_filtered

    advance_chi_values(state, chi_kp1)

    print_mem_summary_rank(f"step {state.step_idx} after iterate_arrays", comm)

    if write_checkpoints:
        with timed_rank(f"step {state.step_idx} checkpoint writes",comm):
            kappa_map, shear1_map, shear2_map, w_map = maps_from_jacobian(state.A_11, state.A_12, state.A_21, state.A_22)    
            write_checkpoint_dir_parallel( state.step_idx, sim, kappa_map, shear1_map, shear2_map, w_map, state.theta, state.phi,  
                                  pix_min, pix_max, comm, state.psi, state.kappa_born_alm)
        del kappa_map
        del shear1_map
        del shear2_map
        del w_map

    
    if write_all_unrotated_steps or (state.step_idx in write_unrotated_step):
        with timed_rank(f"step {state.step_idx} write native ray state",comm):
            write_native_state_parallel(
                state.step_idx, sim,
                state.A_11, state.A_12, state.A_21, state.A_22,
                state.theta, state.phi, state.psi, pix_min, pix_max, comm)

    
    if write_all_rotated_steps or (state.step_idx in write_rotated_step):
 
        with timed_rank(f"step {state.step_idx} output-basis rotation", comm):
            kappa_map, shear1_map, shear2_map, w_map = rotate_state_to_observer_basis(state, theta0, phi0)
            if output_born:
                if rank==0:
                    kappa_born = hp.alm2map(state.kappa_born_alm, nside=nside, lmax=lmax, mmax=lmax, pixwin=True)
                else:
                    kappa_born = None
            else:
                kappa_born=None
            write_outputs_parallel(state.step_idx, sim, pix_min, pix_max, comm, kappa_map, shear1_map, shear2_map, w_map, kappa_born=kappa_born)
        del kappa_map
        del shear1_map
        del shear2_map
        del w_map


    print(f"[timer] step {state.step_idx} total: {time.perf_counter() - step_t0:.2f} s", flush=True)

    print('finished writing stuff')
    state.step_idx +=1 

cmb_convert=True
if cmb_convert:
    advance_ray_and_matrix_state( state, gtheta2, gphi2, U11, U12, U22, chi_cmb )
    del gtheta2, gphi2, U11, U12, U22
    kappa_map, shear1_map, shear2_map, w_map = rotate_state_to_observer_basis(state, theta0, phi0)

    if rank==0:
        if output_born:
            kappa_born_cmb = hp.alm2map(state.kappa_born_alm, nside=nside, lmax=lmax, mmax=lmax, pixwin=True)
        else:
            kappa_born_cmb = None        
    else:
        kappa_born_cmb = None

    write_outputs_parallel(state.step_idx, sim, pix_min, pix_max, comm, kappa_map, shear1_map, shear2_map, w_map, CMB=True, kappa_born=kappa_born_cmb ) # this one should use CMB=True

    
