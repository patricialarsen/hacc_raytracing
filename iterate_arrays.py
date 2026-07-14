import time 
import numpy as np
import healpy as hp
import ducc0
import ducc0.sht as dsht
from contextlib import contextmanager


@contextmanager
def timed(label):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"[timer] {label}: {dt:.2f} s", flush=True)


def iterate_arrays(step_idx, alms_filtered, lmax, theta_new, phi_new, nside, nthreads, eps =1e-5):
    ls = np.arange(lmax + 1)
    fl_phi = np.zeros_like(ls, dtype=np.float64)
    fl_phi[1:] = -2.0 / (ls[1:] * (ls[1:] + 1))  # this converts kappa_lm to phi_lm
    phi_alm = hp.almxfl(alms_filtered, fl_phi, mmax=lmax)

    fl = np.zeros_like(ls, dtype=np.float64)
    fl[1:] = np.sqrt(ls[1:] * (ls[1:] + 1))  # l=0 excluded, this is phi 
    fl_2 = np.zeros_like(ls, dtype=np.float64)
    fl_2[2:]  =  np.sqrt((ls[2:] + 2)*(ls[2:] + 1)*ls[2:]*(ls[2:] - 1))/2. 

    if step_idx==0:
        with timed("iterate almxfl grad"):
            grad_alm = hp.sphtfunc.almxfl(phi_alm, fl, mmax=lmax)
            grad_alm_spin = np.stack([grad_alm, np.zeros_like(grad_alm)])
            grad_alm_second = hp.sphtfunc.almxfl(phi_alm, fl_2, mmax=lmax)
            grad_alm_spin_second = np.stack([grad_alm_second, np.zeros_like(grad_alm_second)])

        
        c = time.time()
        base = ducc0.healpix.Healpix_Base(nside, "RING")
        geom = base.sht_info()

        with timed("iterate synth grad spin1"):
            gtheta2, gphi2 = dsht.synthesis(
                alm=grad_alm_spin,#grad_alm_spin,
                lmax=lmax,
                mmax=lmax,
                spin=1,
                nthreads=nthreads,
                **geom
            )

        with timed("iterate synth kappa spin0"):
            kappa_map = dsht.synthesis(
                alm=alms_filtered[None,:],
                lmax=lmax,
                mmax=lmax,
                spin=0,
                nthreads=nthreads,
                **geom
            ) 
            
        with timed("iterate synth second deriv spin2"):
            second_deriv_maps = dsht.synthesis(
                alm=grad_alm_spin_second,
                lmax=lmax,
                mmax=lmax,
                spin=2,
                nthreads=nthreads,
                **geom
            ) 

        U11 = kappa_map - second_deriv_maps[0] 
        U12 = -second_deriv_maps[1] 
        U22 = kappa_map + second_deriv_maps[0] 

        print('kappa pre trace', np.min((U11+U22)/2.),np.max((U11+U22)/2.))
        print('shear2', np.min(U12),np.max(U12))
        return gtheta2, gphi2, U11, U12, U22
    else:
        with timed("iterate almxfl grad"):
            grad_alm = hp.sphtfunc.almxfl(phi_alm, fl, mmax=lmax)
            #grad_alm_spin = np.stack([grad_alm, np.zeros_like(grad_alm)])
            grad_alm_second = hp.sphtfunc.almxfl(phi_alm, fl_2, mmax=lmax)
            #grad_alm_spin_second = np.stack([grad_alm_second, np.zeros_like(grad_alm_second)])


        with timed("create loc"):
            grad_spin = np.empty((2, grad_alm.size), dtype=grad_alm.dtype)
            grad_spin[0] = grad_alm
            grad_spin[1].fill(0.0)
            loc = np.empty((theta_new.size, 2), dtype=np.float64)
            loc[:, 0] = theta_new
            loc[:, 1] = phi_new


        with timed("iterate synth grad spin1"):
            gtheta2, gphi2 = dsht.synthesis_general(
                alm=grad_spin,
                loc = loc,
                epsilon=eps,
                lmax=lmax,
                mmax=lmax,
                spin=1,
                nthreads=nthreads,
            )
        with timed("iterate synth kappa spin0"):
            kappa_map = dsht.synthesis_general(
                alm=alms_filtered[None,:],
                loc = loc,
                epsilon=eps,
                lmax=lmax,
                mmax=lmax,
                spin=0,
                nthreads=nthreads,
            )
        
        with timed("filled loc"):
            grad_spin[0] = grad_alm_second
            grad_spin[1].fill(0.0)

        with timed("iterate synth second deriv spin2"):
            second_deriv_maps = dsht.synthesis_general(
                alm=grad_spin,
                loc = loc,
                epsilon=eps,
                lmax=lmax,
                mmax=lmax,
                spin=2,
                nthreads=nthreads,
            ) 
        del loc
        del grad_spin
        del grad_alm_second        
        

        U11 = kappa_map - second_deriv_maps[0] 
        U12 = -second_deriv_maps[1] 
        U22 = kappa_map + second_deriv_maps[0] 

        print('kappa pre trace', np.min((U11+U22)/2.),np.max((U11+U22)/2.))
        print('shear2', np.min(U12),np.max(U12))
        
        return gtheta2, gphi2, U11, U12, U22