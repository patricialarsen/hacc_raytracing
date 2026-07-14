import os
os.environ["OMP_NUM_THREADS"] = "128"

import time
import numpy as np
import healpy as hp

from utils import timed
from write_funcs import write_outputs
from hacc_sims import LJ_simulation

from simulation import get_chi_step, get_input_map_alms
from synthetics import get_synthetic_alms_CCL

sim = LJ_simulation()
nside = sim["nside"]
nthreads = 128
lmax = 3 * nside - 1
n_lms = hp.Alm.getsize(lmax, lmax)

write_rotated_step = [6, 13, 23, 36]

source_chi = {
    s: get_chi_step(s, sim)[2]
    for s in write_rotated_step
}

born_accum = {
    s: np.zeros(n_lms, dtype=np.complex128)
    for s in write_rotated_step
}

n_total = int(write_rotated_step[-1]+1)#sim["nplanes"] + sim.get("n_steps_cmb", 0)

for step_idx in range(n_total):
    print("step_idx", step_idx, flush=True)
    step_t0 = time.perf_counter()

    with timed(f"step {step_idx} read/input map"):
        if step_idx > sim["nplanes"] - 1:
            chi_av, alms_filtered = get_synthetic_alms_CCL(
                step_idx,
                sim["path_maps"],
                sim,
                lmax,
                nthreads,
            )
        else:
            chi_av, alms_filtered = get_input_map_alms(
                step_idx,
                sim,
                lmax,
                nthreads,
                filter="wiener",
                ell_cut=None, #int(2.5 * nside),
                use_pixel_weights=False,
            )

    chi_lens = chi_av

    for source_step, chi_source in source_chi.items():
        if chi_lens < chi_source:
            w_shell = (chi_source - chi_lens) / chi_source
            born_accum[source_step] += w_shell * alms_filtered

    if step_idx in write_rotated_step:
        with timed(f"step {step_idx} born alm2map/write"):
            kappa_born = hp.alm2map(
                born_accum[step_idx],
                nside=nside,
                lmax=lmax,
                mmax=lmax,
                pixwin=False,
            )

            write_outputs(
                step_idx,
                sim,
                born_only=True,
                kappa_born=kappa_born,
            )

    del alms_filtered

    print(
        f"[timer] step {step_idx} total: "
        f"{time.perf_counter() - step_t0:.2f} s",
        flush=True,
    )