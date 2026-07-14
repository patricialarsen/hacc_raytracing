import numpy as np
import math 
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def transport_A_rows_approx_kernel(
    A_11, A_12, A_21, A_22,
    theta_from, phi_from,
    theta_to, phi_to,
    A_11_t, A_12_t, A_21_t, A_22_t,
):
    n = A_11.size
    twopi = 2.0 * math.pi

    for i in prange(n):
        dphi = (phi_to[i] - phi_from[i] + math.pi) % twopi - math.pi
        theta_mid = 0.5 * (theta_from[i] + theta_to[i])

        gamma = -math.cos(theta_mid) * dphi
        if abs(dphi) > 0.01:
            gamma = 0.0

        c = math.cos(gamma)
        s = math.sin(gamma)

        A_11_t[i] = c * A_11[i] - s * A_21[i]
        A_21_t[i] = s * A_11[i] + c * A_21[i]
        A_12_t[i] = c * A_12[i] - s * A_22[i]
        A_22_t[i] = s * A_12[i] + c * A_22[i]

def transport_A_rows_approx_numba(
    A_11, A_12, A_21, A_22,
    theta_from, phi_from,
    theta_to, phi_to,
):
    A_11 = np.asarray(A_11).reshape(-1)
    A_12 = np.asarray(A_12).reshape(-1)
    A_21 = np.asarray(A_21).reshape(-1)
    A_22 = np.asarray(A_22).reshape(-1)

    theta_from = np.asarray(theta_from).reshape(-1)
    phi_from = np.asarray(phi_from).reshape(-1)
    theta_to = np.asarray(theta_to).reshape(-1)
    phi_to = np.asarray(phi_to).reshape(-1)

    A_11_t = np.empty_like(A_11)
    A_12_t = np.empty_like(A_12)
    A_21_t = np.empty_like(A_21)
    A_22_t = np.empty_like(A_22)

    transport_A_rows_approx_kernel(
        A_11, A_12, A_21, A_22,
        theta_from, phi_from,
        theta_to, phi_to,
        A_11_t, A_12_t, A_21_t, A_22_t,
    )

    return A_11_t, A_12_t, A_21_t, A_22_t



@njit(parallel=True, fastmath=True)
def update_matrix_transport_approx_kernel(
    A_11_m1, A_12_m1, A_21_m1, A_22_m1,
    A_11, A_12, A_21, A_22,
    U11, U12, U22,
    theta_m1_old, phi_m1_old,
    theta_old, phi_old,
    theta_new, phi_new,
    fact1, fact2, fact3,
    A_11_out, A_12_out, A_21_out, A_22_out,
):
    n = A_11.size
    twopi = 2.0 * math.pi

    for i in prange(n):
        # Transport m1 -> new
        dphi_m1 = (phi_new[i] - phi_m1_old[i] + math.pi) % twopi - math.pi
        theta_mid_m1 = 0.5 * (theta_m1_old[i] + theta_new[i])
        gamma_m1 = -math.cos(theta_mid_m1) * dphi_m1
        if abs(dphi_m1) > 0.01:
            gamma_m1 = 0.0
        c_m1 = math.cos(gamma_m1)
        s_m1 = math.sin(gamma_m1)

        # Transport k -> new
        dphi_k = (phi_new[i] - phi_old[i] + math.pi) % twopi - math.pi
        theta_mid_k = 0.5 * (theta_old[i] + theta_new[i])
        gamma_k = -math.cos(theta_mid_k) * dphi_k
        if abs(dphi_k) > 0.01:
            gamma_k = 0.0
        c_k = math.cos(gamma_k)
        s_k = math.sin(gamma_k)

        # T A_m1
        Tm1_11 = c_m1 * A_11_m1[i] - s_m1 * A_21_m1[i]
        Tm1_21 = s_m1 * A_11_m1[i] + c_m1 * A_21_m1[i]
        Tm1_12 = c_m1 * A_12_m1[i] - s_m1 * A_22_m1[i]
        Tm1_22 = s_m1 * A_12_m1[i] + c_m1 * A_22_m1[i]

        # T A_k
        Tk_11 = c_k * A_11[i] - s_k * A_21[i]
        Tk_21 = s_k * A_11[i] + c_k * A_21[i]
        Tk_12 = c_k * A_12[i] - s_k * A_22[i]
        Tk_22 = s_k * A_12[i] + c_k * A_22[i]

        # U @ A_k, in old k basis
        UA_11 = U11[i] * A_11[i] + U12[i] * A_21[i]
        UA_21 = U12[i] * A_11[i] + U22[i] * A_21[i]
        UA_12 = U11[i] * A_12[i] + U12[i] * A_22[i]
        UA_22 = U12[i] * A_12[i] + U22[i] * A_22[i]

        # T(U @ A_k)
        TUA_11 = c_k * UA_11 - s_k * UA_21
        TUA_21 = s_k * UA_11 + c_k * UA_21
        TUA_12 = c_k * UA_12 - s_k * UA_22
        TUA_22 = s_k * UA_12 + c_k * UA_22

        A_11_out[i] = fact1 * Tm1_11 + fact2 * Tk_11 - fact3 * TUA_11
        A_12_out[i] = fact1 * Tm1_12 + fact2 * Tk_12 - fact3 * TUA_12
        A_21_out[i] = fact1 * Tm1_21 + fact2 * Tk_21 - fact3 * TUA_21
        A_22_out[i] = fact1 * Tm1_22 + fact2 * Tk_22 - fact3 * TUA_22


def update_matrix_transport(
    A_11_m1, A_12_m1, A_21_m1, A_22_m1,
    A_11, A_12, A_21, A_22,
    U11, U12, U22,
    chi_km1, chi_k, chi_kp1,
    theta_m1_old, phi_m1_old,
    theta_old, phi_old,
    theta_new, phi_new,
    nthreads=128,
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

    A_11_out = np.empty_like(A_11)
    A_12_out = np.empty_like(A_12)
    A_21_out = np.empty_like(A_21)
    A_22_out = np.empty_like(A_22)

    update_matrix_transport_approx_kernel(
        A_11_m1, A_12_m1, A_21_m1, A_22_m1,
        A_11, A_12, A_21, A_22,
        U11, U12, U22,
        theta_m1_old, phi_m1_old,
        theta_old, phi_old,
        theta_new, phi_new,
        fact1, fact2, fact3,
        A_11_out, A_12_out, A_21_out, A_22_out,
    )

    return A_11_out, A_12_out, A_21_out, A_22_out, A_11, A_12, A_21, A_22