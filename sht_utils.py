import ducc0.healpix
import ducc0.sht as dsht
import numpy as np


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


def map2alm_ducc_lsmr_from_weighted_adjoint(
    delta_map,
    nside,
    lmax,
    nthreads,
    mmax=None,
    maxiter=1,
    eps=1e-6,
):
    if mmax is None:
        mmax = lmax

    ginfo = ducc0.healpix.Healpix_Base(int(nside), "RING").sht_info()

    m = np.asarray(delta_map, dtype=np.float64)
    omega_pix = 4.0 * np.pi / m.size

    guess = dsht.adjoint_synthesis(
        map=(omega_pix * m).reshape(1, -1),
        spin=0,
        lmax=lmax,
        mmax=mmax,
        nthreads=nthreads,
        **ginfo,
    )

    res = dsht.pseudo_analysis(
        map=m.reshape(1, -1),
        alm=guess.copy(),
        alm_contains_initial_guess=True,
        spin=0,
        lmax=lmax,
        mmax=mmax,
        maxiter=maxiter,
        epsilon=eps,
        nthreads=nthreads,
        **ginfo,
    )

    alm = res[0][0]


    print(
        "istop", res[1],
        "niter", res[2],
        "relres", res[3] / np.linalg.norm(m),
        flush=True,)
    
    return alm