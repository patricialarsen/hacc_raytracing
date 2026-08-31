import numpy as np
import math 
from numba import njit, prange

def wrap_angles(theta, phi):
    pole_mask = (theta <= 1e-12) | (theta >= np.pi - 1e-12)
    theta = np.clip(theta, 1e-12, np.pi - 1e-12)
    phi = np.mod(phi, 2 * np.pi)
    phi = np.where(pole_mask, 0.0, phi)
    return theta, phi

def wrap_angles_ra_dec(ra, dec):
    """ 
    Wrap RA and declination values to their expected bounds
    """ 
    dec = np.clip(dec, -90.0 + 1e-8, 90.0 - 1e-8)
    ra = np.mod(ra, 360.0)
    return ra, dec


def update_beta(theta_m1, phi_m1, theta, phi, alpha_theta, alpha_phi, chi_km1, chi_k, chi_kp1):
    chi_km1_kp1 = chi_kp1 - chi_km1
    chi_km1_k = chi_k - chi_km1
    chi_k_kp1 = chi_kp1 - chi_k
    fact2 = chi_k/chi_kp1 *chi_km1_kp1/chi_km1_k
    fact1 = 1. - fact2
    fact3 = -chi_k_kp1/chi_kp1
    polar = (theta <= 1e-4) | (theta >= np.pi - 1e-4)
    alpha_phi = np.where(polar, 0.0, alpha_phi)
    theta_kp1 = fact1 * theta_m1 + fact2 * theta + fact3 * alpha_theta
    phi_kp1 = fact1 * phi_m1 + fact2 * phi + fact3 * alpha_phi
    theta_kp1, phi_kp1 = wrap_angles(theta_kp1,phi_kp1)
    return theta_kp1, phi_kp1, theta, phi # return plane and m1 plane 


@njit(parallel=True, fastmath=True)
def update_beta_xyz_ray_kernel(
    theta_m1, phi_m1,
    theta, phi,
    alpha_theta, alpha_phi,
    fact1, fact3,
    theta_out, phi_out,
):
    n = theta.size
    twopi = 2.0 * math.pi
    eps = 1e-12
    pole_eps = 1e-4

    for i in prange(n):
        th0 = theta[i]
        ph0 = phi[i]
        sth = math.sin(th0)
        cth = math.cos(th0)
        sph = math.sin(ph0)
        cph = math.cos(ph0)

        nx = sth * cph
        ny = sth * sph
        nz = cth

        ethx = cth * cph
        ethy = cth * sph
        ethz = -sth
        ephx = -sph
        ephy = cph
        ephz = 0.0

        thm = theta_m1[i]
        phm = phi_m1[i]
        stm = math.sin(thm)
        ctm = math.cos(thm)
        spm = math.sin(phm)
        cpm = math.cos(phm)
        nmx = stm * cpm
        nmy = stm * spm
        nmz = ctm

        dot = nx * nmx + ny * nmy + nz * nmz
        if dot > 1.0:
            dot = 1.0
        elif dot < -1.0:
            dot = -1.0

        ang = math.acos(dot)
        sang = math.sin(ang)
        if sang > 1e-14:
            scal = ang / sang
            logx = scal * (nmx - dot * nx)
            logy = scal * (nmy - dot * ny)
            logz = scal * (nmz - dot * nz)
        else:
            logx = 0.0
            logy = 0.0
            logz = 0.0

        aphi = alpha_phi[i]
        if th0 <= pole_eps or th0 >= math.pi - pole_eps:
            aphi = 0.0

        alphax = alpha_theta[i] * ethx + aphi * ephx
        alphay = alpha_theta[i] * ethy + aphi * ephy
        alphaz = alpha_theta[i] * ethz + aphi * ephz

        vx = fact1 * logx + fact3 * alphax
        vy = fact1 * logy + fact3 * alphay
        vz = fact1 * logz + fact3 * alphaz

        vnorm = math.sqrt(vx * vx + vy * vy + vz * vz)
        if vnorm > 1e-14:
            cv = math.cos(vnorm)
            sv = math.sin(vnorm) / vnorm
            outx = cv * nx + sv * vx
            outy = cv * ny + sv * vy
            outz = cv * nz + sv * vz
        else:
            outx = nx + vx
            outy = ny + vy
            outz = nz + vz

        norm = math.sqrt(outx * outx + outy * outy + outz * outz)
        outx /= norm
        outy /= norm
        outz /= norm

        if outz > 1.0:
            outz = 1.0
        elif outz < -1.0:
            outz = -1.0

        th = math.acos(outz)
        if th < eps:
            th = eps
        elif th > math.pi - eps:
            th = math.pi - eps

        ph = math.atan2(outy, outx)
        if ph < 0.0:
            ph += twopi

        theta_out[i] = th
        phi_out[i] = ph


def update_beta_xyz_ray_numba(
    theta_m1, phi_m1,
    theta, phi,
    alpha_theta, alpha_phi,
    chi_km1, chi_k, chi_kp1,
):
    chi_km1_kp1 = chi_kp1 - chi_km1
    chi_km1_k = chi_k - chi_km1
    chi_k_kp1 = chi_kp1 - chi_k

    fact2 = chi_k / chi_kp1 * chi_km1_kp1 / chi_km1_k
    fact1 = 1.0 - fact2
    fact3 = -chi_k_kp1 / chi_kp1

    theta_m1 = np.asarray(theta_m1).reshape(-1)
    phi_m1 = np.asarray(phi_m1).reshape(-1)
    theta = np.asarray(theta).reshape(-1)
    phi = np.asarray(phi).reshape(-1)
    alpha_theta = np.asarray(alpha_theta).reshape(-1)
    alpha_phi = np.asarray(alpha_phi).reshape(-1)

    theta_out = np.empty_like(theta)
    phi_out = np.empty_like(phi)

    update_beta_xyz_ray_kernel(
        theta_m1, phi_m1,
        theta, phi,
        alpha_theta, alpha_phi,
        fact1, fact3,
        theta_out, phi_out,
    )

    return theta_out, phi_out, theta, phi


def update_beta_state(state):
    (
        state.theta,
        state.phi,
        state.theta_m1,
        state.phi_m1,
    ) = update_beta_xyz_ray_numba(
        state.theta_m1,
        state.phi_m1,
        state.theta,
        state.phi,
        state.gtheta2,
        state.gphi2,
        state.chi_km1,
        state.chi_k,
        state.chi_kp1,
    )

def warmup_numba_kernels_beta():
    print("warming up numba kernels", flush=True)

    n = 2048

    theta0 = np.linspace(0.05, np.pi - 0.05, n, dtype=np.float64)
    phi0 = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float64)

    theta1 = theta0 + 1.0e-5 * np.sin(phi0)
    phi1 = np.mod(phi0 + 1.0e-5 * np.cos(theta0), 2.0 * np.pi)

    theta_m1 = theta0.copy()
    phi_m1 = phi0.copy()
    theta_old = theta0.copy()
    phi_old = phi0.copy()

    alpha_theta = 1.0e-5 * np.sin(phi0)
    alpha_phi = 1.0e-5 * np.cos(phi0)

    chi_km1 = 10.0
    chi_k = 50.0
    chi_kp1 = 150.0

    _ = update_beta_xyz_ray_numba(
        theta_m1, phi_m1,
        theta_old, phi_old,
        alpha_theta, alpha_phi,
        chi_km1, chi_k, chi_kp1,
    )

    print("finished numba warmup", flush=True)
