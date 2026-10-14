#!/bin/bash/python

import os
import sys
import time
os.environ["OMP_NUM_THREADS"] = "128"
import healpy as hp
import numpy as np

from hacc_sims import LJ_simulation
from utils import initialize_singleplane_state_from_outputs, rotate_state_to_observer_basis_inplace
from transport_xyz import warmup_numba_kernels_transport
from diffsky_routines import add_lensing, galaxy_steps_overlapping_chi_interval, galaxy_steps_overlapping_chi_interval_z0, parse_patch_list


nthreads = 128

print('Setting up', flush=True)
warmup_numba_kernels_transport()


step_list_gals = np.array([
    "97", "100", "102", "105", "107", "110", "113", "116", "119",
    "121", "124", "127", "131", "134", "137", "141", "144", "148",
    "151", "155", "159", "163", "167", "171", "176", "180", "184",
    "189", "194", "198", "203", "208", "213", "219", "224", "230",
    "235", "241", "247", "253", "259", "266", "272", "279", "286",
    "293", "300", "307", "315", "323", "331", "338", "347", "355",
    "365", "373", "382", "392", "401", "411", "421", "432", "442",
    "453", "464", "475", "487",
])[::-1]

mag_cols = []
for band in ["g", "r", "i", "z", "y", "u"]:
    for comp in ["", "_bulge", "_disk", "_knots"]:
        mag_cols.append("lsst_" + band + comp)
for band in [
    "F062", "F087", "F106", "F129", "F146", "F158", "F184", "F213",
    "Grism_0thOrder", "Grism_1stOrder", "Prism",
]:
    for comp in ["", "_bulge", "_disk", "_knots"]:
        mag_cols.append("roman_" + band + comp)

flux_cols = [
    "Ba_alpha_6563", "Ba_beta_4861",
    "O_III_4363", "O_III_4959", "O_III_5007",
    "O_II_3726", "O_II_3729",
    "Pa_alpha_1p8751um",
    "S_III_6312", "S_III_9069", "S_III_9532",
]


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) < 3:
        raise SystemExit(
            "usage: python run_interpolation.py PATH_GALS SYNTHETIC_FLAG PATCH_LIST "
        )

    path_gals = argv[0]
    synthetic = argv[1]
    patch_list = parse_patch_list(argv[2])

    if path_gals and not path_gals.endswith("/"):
        path_gals = path_gals + "/"

    if synthetic == "S":
        path_end = ".diffsky_gals.synthetic_halos.hdf5"
    else:
        path_end = ".diffsky_gals.hdf5"

    sim = LJ_simulation()
    nside = sim['nside']
    npix = hp.nside2npix(nside)

    step_idx = 0; step_idx2 = 1;
    theta0, phi0 = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))


    while(step_idx<sim['nplanes']):
        plane_start = time.perf_counter()
        print(f"[plane {step_idx + 1}/{sim['nplanes']}] Loading and rotating source states {step_idx}, {step_idx2}", flush=True)
        # initialize state and rotate values to observer basis 
        state = initialize_singleplane_state_from_outputs(sim, step_idx)
        rotate_state_to_observer_basis_inplace(state, theta0, phi0)

        state2 = initialize_singleplane_state_from_outputs(sim, step_idx2)
        rotate_state_to_observer_basis_inplace(state2, theta0, phi0)

        gal_steps, chi_min, chi_max = galaxy_steps_overlapping_chi_interval(step_idx, step_idx2, step_list_gals, sim)
        print(f"[plane {step_idx + 1}/{sim['nplanes']}] States ready; chi=[{chi_min:.6g}, {chi_max:.6g}) Mpc/h; galaxy steps={list(gal_steps)}", flush=True)

        candidate_files = []
        if len(gal_steps)>0:
            candidate_files = []
            for step in gal_steps:
                for patch in patch_list:
                    filename = path_gals + "lc_cores-" + str(step) + "." + str(patch) + path_end
                    # will need to update each file separately
                    candidate_files.append(filename)
                
        add_lensing(candidate_files, state, state2, nside, chi_min, chi_max, mag_cols, flux_cols, sim)

        # adding the set of galaxies which are just beyond one bound 
        if step_idx==0:
            gal_steps, chi_max = galaxy_steps_overlapping_chi_interval_z0(step_idx, step_list_gals, sim)
            print(f"[plane 1/{sim['nplanes']}] Processing galaxies below the first source plane: chi=[0, {chi_max:.6g}) Mpc/h", flush=True)
            if len(gal_steps)>0:
                candidate_files = []
                for step in gal_steps:
                    for patch in patch_list:
                        filename = path_gals + "lc_cores-" + str(step) + "." + str(patch) + path_end
                        # will need to update each file separately
                        candidate_files.append(filename)

            add_lensing(candidate_files, state, state2, nside, 0, chi_max, mag_cols, flux_cols, sim, first_plane=True)



        print(f"[plane {step_idx + 1}/{sim['nplanes']}] Complete in {time.perf_counter() - plane_start:.1f} s", flush=True)
        step_idx +=1
        step_idx2 +=1 

    print("Interpolation complete: all configured plane intervals processed.", flush=True)


if __name__ == "__main__":
    main()

