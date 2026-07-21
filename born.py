import os
os.environ["OMP_NUM_THREADS"] = "128"

import time
import numpy as np
import healpy as hp

from utils import timed
from hacc_sims import LJ_simulation, FrontierE_simulation

from simulation import get_chi_step, get_input_map_alms
from synthetics import get_synthetic_alms_CCL
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


sim = FrontierE_simulation()
nside = sim["nside"]
nthreads = 128

lmax = int(2.5*nside)# * nside - 1
n_lms = hp.Alm.getsize(lmax, lmax)

write_rotated_step = [6, 13, 23]

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
                ell_cut=int(2.5*nside), #int(2.5 * nside),
                use_pixel_weights=False, comm=comm, save_alms=True
            )

    chi_lens = chi_av

    for source_step, chi_source in source_chi.items():
        if chi_lens < chi_source:
            w_shell = (chi_source - chi_lens) / chi_source
            born_accum[source_step] += w_shell * alms_filtered

    del alms_filtered
    if step_idx in write_rotated_step:
        hp.write_alm(sim['output_path_rt']+'born_'+str(step_idx)+'.fits', born_accum[step_idx], overwrite=True)


    print(
        f"[timer] step {step_idx} total: "
        f"{time.perf_counter() - step_t0:.2f} s",
        flush=True,
    )