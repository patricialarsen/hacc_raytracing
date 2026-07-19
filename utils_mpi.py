#!/bin/bash/python
from dataclasses import dataclass
import numpy as np
import healpy as hp
from angle_updates import  update_beta_xyz_ray_numba
from transport_xyz import update_matrix_transport_xyz_basis_numba, transport_A_rows_xyz_basis_numba
from contextlib import contextmanager
from write_funcs_mpi import restart_from_checkpoint
import time

from utils import RayState

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



def initialize_ray_state_chunked(nside, pix_min, pix_max,  lmax, output_born):
    dtype = np.float64

    npix_chunked = int(pix_max-pix_min)
    
    def identity_jacobian():
        return (
            np.ones(npix_chunked, dtype=dtype),
            np.zeros(npix_chunked, dtype=dtype),
            np.zeros(npix_chunked, dtype=dtype),
            np.ones(npix_chunked, dtype=dtype),
        )

    theta, phi = hp.pix2ang(nside, np.arange(pix_min, pix_max, dtype=np.int64))

    A_11_m1, A_12_m1, A_21_m1, A_22_m1 = identity_jacobian()
    A_11, A_12, A_21, A_22 = identity_jacobian()

    if rank==0:
        if output_born:
            alm_init = np.zeros(np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128))
        else:
            alm_init = np.zeros(0, dtype=np.complex128)
        return RayState(
            A_11_m1, A_12_m1, A_21_m1, A_22_m1,
            A_11, A_12, A_21, A_22,
            theta, phi,
            theta.copy(), phi.copy(),
            np.zeros(npix_chunked, dtype=dtype),
            np.zeros(npix_chunked, dtype=dtype),
            alm_init,
        )
    else:
        return RayState(
            A_11_m1, A_12_m1, A_21_m1, A_22_m1,
            A_11, A_12, A_21, A_22,
            theta, phi,
            theta.copy(), phi.copy(),
            np.zeros(npix_chunked, dtype=dtype),
            np.zeros(npix_chunked, dtype=dtype),
            np.empty((), dtype=np.complex128),
        )
        
    

def initialize_ray_state_restart(sim, pix_min, pix_max, comm,  add_psi=True):
    (step_idx, theta, phi, A_11, A_22, A_12, A_21, psi, theta_m1, 
     phi_m1, A_11_m1, A_22_m1, A_12_m1, A_21_m1, psi_m1, kappa_born_alm, 
     chi_km1, chi_k,)= restart_from_checkpoint(sim, pix_min, pix_max, comm, add_psi=add_psi)
    return RayState(
        A_11_m1, A_12_m1, A_21_m1, A_22_m1,
        A_11, A_12, A_21, A_22, theta, phi,
        theta_m1, phi_m1, psi, psi_m1,
        kappa_born_alm, chi_km1, chi_k, step_idx)



def print_mem_summary(label, comm):
    import os
    import resource
    import psutil

    proc = psutil.Process(os.getpid())
    rss_gb = proc.memory_info().rss / 1024**3
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2

    vals = np.array([rss_gb, peak_gb], dtype=np.float64)
    gathered = comm.gather(vals, root=0)

    if comm.Get_rank() == 0:
        arr = np.vstack(gathered)
        print(
            f"[mem] {label}: "
            f"rss min/mean/max={arr[:,0].min():.2f}/"
            f"{arr[:,0].mean():.2f}/{arr[:,0].max():.2f} GB, "
            f"peak min/mean/max={arr[:,1].min():.2f}/"
            f"{arr[:,1].mean():.2f}/{arr[:,1].max():.2f} GB",
            flush=True,
        )

def print_mem_summary_rank(label, comm):
    import os
    import resource
    import psutil

    rank = comm.Get_rank()

    proc = psutil.Process(os.getpid())
    rss_gb = proc.memory_info().rss / 1024**3
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2

    vals = np.array([rank, rss_gb, peak_gb], dtype=np.float64)
    gathered = comm.gather(vals, root=0)

    if rank == 0:
        arr = np.vstack(gathered)

        ranks = arr[:, 0].astype(int)
        rss = arr[:, 1]
        peak = arr[:, 2]

        rss_min_i = np.argmin(rss)
        rss_max_i = np.argmax(rss)
        peak_min_i = np.argmin(peak)
        peak_max_i = np.argmax(peak)

        print(
            f"[mem] {label}: "
            f"rss min/mean/max="
            f"{rss[rss_min_i]:.2f}(r{ranks[rss_min_i]})/"
            f"{rss.mean():.2f}/"
            f"{rss[rss_max_i]:.2f}(r{ranks[rss_max_i]}) GB, "
            f"peak min/mean/max="
            f"{peak[peak_min_i]:.2f}(r{ranks[peak_min_i]})/"
            f"{peak.mean():.2f}/"
            f"{peak[peak_max_i]:.2f}(r{ranks[peak_max_i]}) GB",
            flush=True,
        )