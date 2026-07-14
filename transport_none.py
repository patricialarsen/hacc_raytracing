
def update_matrix(A_11_m1, A_12_m1, A_21_m1, A_22_m1, A_11, A_12, A_21, A_22, U11, U12, U22, chi_km1, chi_k, chi_kp1):
    
    chi_km1_kp1 = chi_kp1 - chi_km1
    chi_km1_k = chi_k - chi_km1
    chi_k_kp1 = chi_kp1 - chi_k
    fact2 = chi_k/chi_kp1 *chi_km1_kp1/chi_km1_k
    fact1 = 1. - fact2
    fact3 = chi_k_kp1/chi_kp1


    A_11_kp1 = fact1 * A_11_m1 + fact2 * A_11 - fact3 * (U11 * A_11 + U12 * A_21)
    A_12_kp1 = fact1 * A_12_m1 + fact2 * A_12 - fact3 * (U11 * A_12 + U12 * A_22)
    A_21_kp1 = fact1 * A_21_m1 + fact2 * A_21 - fact3 * (U12 * A_11 + U22 * A_21)
    A_22_kp1 = fact1 * A_22_m1 + fact2 * A_22 - fact3 * (U12 * A_12 + U22 * A_22)

    return A_11_kp1, A_12_kp1, A_21_kp1, A_22_kp1, A_11, A_12, A_21, A_22 