#!/bin/bash/python
from dataclasses import dataclass
import numpy as np
import healpy as hp
from angle_updates import  update_beta_xyz_ray_numba
from transport_xyz import update_matrix_transport_xyz_basis_numba, transport_A_rows_xyz_basis_numba
from contextlib import contextmanager
from write_funcs import restart_from_checkpoint
import time
import ducc0.healpix
import ducc0.sht as dsht


@contextmanager
def timed(label):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"[timer] {label}: {dt:.2f} s", flush=True)


@dataclass
class RayState:
    A_11_m1: np.ndarray
    A_12_m1: np.ndarray
    A_21_m1: np.ndarray
    A_22_m1: np.ndarray
    A_11: np.ndarray
    A_12: np.ndarray
    A_21: np.ndarray
    A_22: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    theta_m1: np.ndarray
    phi_m1: np.ndarray
    psi: np.ndarray
    psi_m1: np.ndarray
    kappa_born_alm: np.ndarray
    chi_km1: float = 0.0
    chi_k: float = 0.0
    step_idx: int = 0


def initialize_ray_state(nside, lmax):
    npix = hp.nside2npix(nside)
    dtype = np.float64

    def identity_jacobian():
        return (
            np.ones(npix, dtype=dtype),
            np.zeros(npix, dtype=dtype),
            np.zeros(npix, dtype=dtype),
            np.ones(npix, dtype=dtype),
        )

    theta, phi = hp.pix2ang(nside, np.arange(npix, dtype=np.int64))

    A_11_m1, A_12_m1, A_21_m1, A_22_m1 = identity_jacobian()
    A_11, A_12, A_21, A_22 = identity_jacobian()

    return RayState(
        A_11_m1, A_12_m1, A_21_m1, A_22_m1,
        A_11, A_12, A_21, A_22,
        theta, phi,
        theta.copy(), phi.copy(),
        np.zeros(npix, dtype=dtype),
        np.zeros(npix, dtype=dtype),
        np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128),
    )



def maps_from_jacobian(A_11, A_12, A_21, A_22):
    kappa_map = 1.0 - (A_11 + A_22)/2.
    shear1_map = -(A_11-A_22)/2.
    shear2_map = (A_12 +A_21)/2.
    w_map = (A_12 - A_21)/2.
    return kappa_map, shear1_map, shear2_map, w_map
        

def advance_ray_and_matrix_state(
    state,
    gtheta2,
    gphi2,
    U11,
    U12,
    U22,
    chi_kp1,
):
    theta_m1_old = state.theta_m1
    phi_m1_old = state.phi_m1
    theta_old = state.theta
    phi_old = state.phi
    psi_m1_old = state.psi_m1
    psi_old = state.psi

    (
        state.theta,
        state.phi,
        state.theta_m1,
        state.phi_m1,
    ) = update_beta_xyz_ray_numba(
        state.theta_m1, state.phi_m1,
        state.theta, state.phi,
        gtheta2, gphi2,
        state.chi_km1, state.chi_k, chi_kp1,
    )

    (
        state.A_11, state.A_12, state.A_21, state.A_22,
        state.A_11_m1, state.A_12_m1,
        state.A_21_m1, state.A_22_m1,
        state.psi, state.psi_m1,
    ) = update_matrix_transport_xyz_basis_numba(
        state.A_11_m1, state.A_12_m1,
        state.A_21_m1, state.A_22_m1,
        state.A_11, state.A_12,
        state.A_21, state.A_22,
        U11, U12, U22,
        state.chi_km1, state.chi_k, chi_kp1,
        theta_m1_old, phi_m1_old,
        theta_old, phi_old,
        state.theta, state.phi,
        psi_m1_old, psi_old,
    )


def rotate_state_to_observer_basis(state, theta0, phi0):
    A_obs = transport_A_rows_xyz_basis_numba(
        state.A_11,
        state.A_12,
        state.A_21,
        state.A_22,
        state.theta,
        state.phi,
        theta0,
        phi0,
        state.psi,
    )
    return maps_from_jacobian(*A_obs)

def advance_chi_values(state, chi_kp1):
    state.chi_km1 = state.chi_k
    state.chi_k = chi_kp1



def initialize_ray_state_restart(sim,  add_psi=True):
    (step_idx, theta, phi, A_11, A_22, A_12, A_21, psi, theta_m1, 
     phi_m1, A_11_m1, A_22_m1, A_12_m1, A_21_m1, psi_m1, kappa_born_alm, 
     chi_km1, chi_k,)= restart_from_checkpoint(sim, add_psi=add_psi)
    return RayState(
        A_11_m1, A_12_m1, A_21_m1, A_22_m1,
        A_11, A_12, A_21, A_22, theta, phi,
        theta_m1, phi_m1, psi, psi_m1,
        kappa_born_alm, chi_km1, chi_k, step_idx)


def map2alm_ducc_lsmr(delta_map, nside, lmax, nthreads, eps=1e-6, maxiter=3,):

    m = np.asarray(delta_map, dtype=np.float64).reshape(1, -1)
    ginfo = ducc0.healpix.Healpix_Base(int(nside), "RING").sht_info()

    kwargs = dict(
        map=m,
        lmax=int(lmax),
        mmax=int(lmax),
        spin=0,
        maxiter=int(maxiter),
        epsilon=float(eps),
        nthreads=int(nthreads),
        **ginfo,
    )

    res = dsht.pseudo_analysis(**kwargs)

    alm = res[0][0]

    return alm
