import numpy as np
import math 
from numba import njit, prange


@njit(fastmath=True)
def wrap_pm_pi(x):
    return (x + math.pi) % (2.0 * math.pi) - math.pi    

@njit(fastmath=True)
def exact_transport_cs_one(theta_from, phi_from, theta_to, phi_to):
    st0 = math.sin(theta_from)
    ct0 = math.cos(theta_from)
    sp0 = math.sin(phi_from)
    cp0 = math.cos(phi_from)

    st1 = math.sin(theta_to)
    ct1 = math.cos(theta_to)
    sp1 = math.sin(phi_to)
    cp1 = math.cos(phi_to)

    # n0, eth0
    n0x = st0 * cp0
    n0y = st0 * sp0
    n0z = ct0

    eth0x = ct0 * cp0
    eth0y = ct0 * sp0
    eth0z = -st0

    # n1, eth1, eph1
    n1x = st1 * cp1
    n1y = st1 * sp1
    n1z = ct1

    eth1x = ct1 * cp1
    eth1y = ct1 * sp1
    eth1z = -st1

    eph1x = -sp1
    eph1y = cp1
    eph1z = 0.0

    # axis = cross(n0, n1)
    ax = n0y * n1z - n0z * n1y
    ay = n0z * n1x - n0x * n1z
    az = n0x * n1y - n0y * n1x

    sinang = math.sqrt(ax * ax + ay * ay + az * az)
    cosang = n0x * n1x + n0y * n1y + n0z * n1z

    if sinang < 1e-14:
        # same point: no transport rotation
        return 1.0, 0.0

    inv = 1.0 / sinang
    ax *= inv
    ay *= inv
    az *= inv

    # Rodrigues rotate eth0 about axis by angle:
    # eth0_t = eth0*cosang + cross(axis, eth0)*sinang + axis*dot(axis, eth0)*(1-cosang)
    cx = ay * eth0z - az * eth0y
    cy = az * eth0x - ax * eth0z
    cz = ax * eth0y - ay * eth0x

    adot = ax * eth0x + ay * eth0y + az * eth0z
    one_minus_c = 1.0 - cosang

    eth0tx = eth0x * cosang + cx * sinang + ax * adot * one_minus_c
    eth0ty = eth0y * cosang + cy * sinang + ay * adot * one_minus_c
    eth0tz = eth0z * cosang + cz * sinang + az * adot * one_minus_c

    c = eth1x * eth0tx + eth1y * eth0ty + eth1z * eth0tz
    s = eph1x * eth0tx + eph1y * eth0ty + eph1z * eth0tz

    return c, s


@njit(parallel=True, fastmath=True)
def update_matrix_transport_xyz_basis_kernel(
    A_11_m1, A_12_m1, A_21_m1, A_22_m1,
    A_11, A_12, A_21, A_22,
    U11, U12, U22,
    theta_m1_old, phi_m1_old,
    theta_old, phi_old,
    theta_new, phi_new,
    psi_m1_old, psi_old,
    fact1, fact2, fact3,
    A_11_out, A_12_out, A_21_out, A_22_out,
    psi_out,
):
    n = A_11.size

    for i in prange(n):
        c_m1_local, s_m1_local = exact_transport_cs_one(
            theta_m1_old[i], phi_m1_old[i],
            theta_new[i], phi_new[i],
        )
        gamma_m1 = math.atan2(s_m1_local, c_m1_local)

        c_k_local, s_k_local = exact_transport_cs_one(
            theta_old[i], phi_old[i],
            theta_new[i], phi_new[i],
        )
        gamma_k = math.atan2(s_k_local, c_k_local)

        # Carry the current basis forward by exact parallel transport and store A_out in that basis.
        psi_new = wrap_pm_pi(psi_old[i] + gamma_k)
        psi_out[i] = psi_new

        # Previous-plane basis, transported to the new point, relative to the carried current basis.
        delta_m1 = wrap_pm_pi(psi_m1_old[i] + gamma_m1 - psi_new)
        c_m1 = math.cos(delta_m1)
        s_m1 = math.sin(delta_m1)

        Tm1_11 = c_m1 * A_11_m1[i] - s_m1 * A_21_m1[i]
        Tm1_21 = s_m1 * A_11_m1[i] + c_m1 * A_21_m1[i]
        Tm1_12 = c_m1 * A_12_m1[i] - s_m1 * A_22_m1[i]
        Tm1_22 = s_m1 * A_12_m1[i] + c_m1 * A_22_m1[i]

        # A_k is already expressed in the carried current basis, which defines the new basis after transport.
        # U is evaluated in the local angular basis, so rotate U into the carried basis before applying it.
        cb = math.cos(psi_old[i])
        sb = math.sin(psi_old[i])

        m11 = U11[i] * cb + U12[i] * sb
        m12 = -U11[i] * sb + U12[i] * cb
        m21 = U12[i] * cb + U22[i] * sb
        m22 = -U12[i] * sb + U22[i] * cb

        UB_11 = cb * m11 + sb * m21
        UB_12 = cb * m12 + sb * m22
        UB_21 = -sb * m11 + cb * m21
        UB_22 = -sb * m12 + cb * m22

        UA_11 = UB_11 * A_11[i] + UB_12 * A_21[i]
        UA_21 = UB_21 * A_11[i] + UB_22 * A_21[i]
        UA_12 = UB_11 * A_12[i] + UB_12 * A_22[i]
        UA_22 = UB_21 * A_12[i] + UB_22 * A_22[i]

        A_11_out[i] = fact1 * Tm1_11 + fact2 * A_11[i] - fact3 * UA_11
        A_12_out[i] = fact1 * Tm1_12 + fact2 * A_12[i] - fact3 * UA_12
        A_21_out[i] = fact1 * Tm1_21 + fact2 * A_21[i] - fact3 * UA_21
        A_22_out[i] = fact1 * Tm1_22 + fact2 * A_22[i] - fact3 * UA_22


def update_matrix_transport_xyz_basis_numba(
    A_11_m1, A_12_m1, A_21_m1, A_22_m1,
    A_11, A_12, A_21, A_22,
    U11, U12, U22,
    chi_km1, chi_k, chi_kp1,
    theta_m1_old, phi_m1_old,
    theta_old, phi_old,
    theta_new, phi_new,
    psi_m1_old, psi_old,
):
    chi_km1_kp1 = chi_kp1 - chi_km1
    chi_km1_k = chi_k - chi_km1
    chi_k_kp1 = chi_kp1 - chi_k

    fact2 = chi_k / chi_kp1 * chi_km1_kp1 / chi_km1_k
    fact1 = 1.0 - fact2
    fact3 = chi_k_kp1 / chi_kp1

    A_11 = np.asarray(A_11).reshape(-1)
    A_12 = np.asarray(A_12).reshape(-1)
    A_21 = np.asarray(A_21).reshape(-1)
    A_22 = np.asarray(A_22).reshape(-1)
    A_11_m1 = np.asarray(A_11_m1).reshape(-1)
    A_12_m1 = np.asarray(A_12_m1).reshape(-1)
    A_21_m1 = np.asarray(A_21_m1).reshape(-1)
    A_22_m1 = np.asarray(A_22_m1).reshape(-1)
    U11 = np.asarray(U11).reshape(-1)
    U12 = np.asarray(U12).reshape(-1)
    U22 = np.asarray(U22).reshape(-1)
    theta_m1_old = np.asarray(theta_m1_old).reshape(-1)
    phi_m1_old = np.asarray(phi_m1_old).reshape(-1)
    theta_old = np.asarray(theta_old).reshape(-1)
    phi_old = np.asarray(phi_old).reshape(-1)
    theta_new = np.asarray(theta_new).reshape(-1)
    phi_new = np.asarray(phi_new).reshape(-1)
    psi_m1_old = np.asarray(psi_m1_old).reshape(-1)
    psi_old = np.asarray(psi_old).reshape(-1)

    A_11_out = np.empty_like(A_11)
    A_12_out = np.empty_like(A_12)
    A_21_out = np.empty_like(A_21)
    A_22_out = np.empty_like(A_22)
    psi_out = np.empty_like(psi_old)

    update_matrix_transport_xyz_basis_kernel(
        A_11_m1, A_12_m1, A_21_m1, A_22_m1,
        A_11, A_12, A_21, A_22,
        U11, U12, U22,
        theta_m1_old, phi_m1_old,
        theta_old, phi_old,
        theta_new, phi_new,
        psi_m1_old, psi_old,
        fact1, fact2, fact3,
        A_11_out, A_12_out, A_21_out, A_22_out,
        psi_out,
    )

    return A_11_out, A_12_out, A_21_out, A_22_out, A_11, A_12, A_21, A_22, psi_out, psi_old


@njit(parallel=True, fastmath=True)
def transport_A_rows_xyz_basis_kernel(
    A_11, A_12, A_21, A_22,
    theta_from, phi_from,
    theta_to, phi_to,
    psi_from,
    A_11_t, A_12_t, A_21_t, A_22_t,
):
    n = A_11.size
    for i in prange(n):
        c_local, s_local = exact_transport_cs_one(
            theta_from[i], phi_from[i],
            theta_to[i], phi_to[i],
        )
        gamma_local = math.atan2(s_local, c_local)
        delta = wrap_pm_pi(psi_from[i] + gamma_local)
        c = math.cos(delta)
        s = math.sin(delta)

        A_11_t[i] = c * A_11[i] - s * A_21[i]
        A_21_t[i] = s * A_11[i] + c * A_21[i]
        A_12_t[i] = c * A_12[i] - s * A_22[i]
        A_22_t[i] = s * A_12[i] + c * A_22[i]


def transport_A_rows_xyz_basis_numba(
    A_11, A_12, A_21, A_22,
    theta_from, phi_from,
    theta_to, phi_to,
    psi_from,
):
    A_11 = np.asarray(A_11).reshape(-1)
    A_12 = np.asarray(A_12).reshape(-1)
    A_21 = np.asarray(A_21).reshape(-1)
    A_22 = np.asarray(A_22).reshape(-1)
    theta_from = np.asarray(theta_from).reshape(-1)
    phi_from = np.asarray(phi_from).reshape(-1)
    theta_to = np.asarray(theta_to).reshape(-1)
    phi_to = np.asarray(phi_to).reshape(-1)
    psi_from = np.asarray(psi_from).reshape(-1)

    A_11_t = np.empty_like(A_11)
    A_12_t = np.empty_like(A_12)
    A_21_t = np.empty_like(A_21)
    A_22_t = np.empty_like(A_22)

    transport_A_rows_xyz_basis_kernel(
        A_11, A_12, A_21, A_22,
        theta_from, phi_from,
        theta_to, phi_to,
        psi_from,
        A_11_t, A_12_t, A_21_t, A_22_t,
    )

    return A_11_t, A_12_t, A_21_t, A_22_t


def warmup_numba_kernels_transport():
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
    theta_new = theta1.copy()
    phi_new = phi1.copy()

    A_11_m1 = np.ones(n, dtype=np.float64)
    A_12_m1 = np.zeros(n, dtype=np.float64)
    A_21_m1 = np.zeros(n, dtype=np.float64)
    A_22_m1 = np.ones(n, dtype=np.float64)

    A_11 = np.ones(n, dtype=np.float64)
    A_12 = np.zeros(n, dtype=np.float64)
    A_21 = np.zeros(n, dtype=np.float64)
    A_22 = np.ones(n, dtype=np.float64)

    U11 = 1.0e-4 * np.sin(phi0)
    U12 = 1.0e-5 * np.cos(phi0)
    U22 = 1.0e-4 * np.cos(theta0)

    alpha_theta = 1.0e-5 * np.sin(phi0)
    alpha_phi = 1.0e-5 * np.cos(phi0)

    chi_km1 = 10.0
    chi_k = 50.0
    chi_kp1 = 150.0

    psi_m1 = np.zeros(n, dtype=np.float64)
    psi = np.zeros(n, dtype=np.float64)
    xyz_res = update_matrix_transport_xyz_basis_numba(
        A_11_m1, A_12_m1, A_21_m1, A_22_m1,
        A_11, A_12, A_21, A_22,
        U11, U12, U22,
        chi_km1, chi_k, chi_kp1,
        theta_m1, phi_m1,
        theta_old, phi_old,
        theta_new, phi_new,
        psi_m1, psi,
    )

    _ = transport_A_rows_xyz_basis_numba(
        A_11, A_12, A_21, A_22,
        theta_old, phi_old,
        theta_new, phi_new,
        xyz_res[8],
    )

    print("finished numba warmup", flush=True)

