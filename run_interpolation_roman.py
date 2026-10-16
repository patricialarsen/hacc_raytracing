#!/bin/bash/python

import os
import sys
import time
import argparse
from contextlib import nullcontext
os.environ["OMP_NUM_THREADS"] = "128"
import healpy as hp
import numpy as np

from hacc_sims import LJ_simulation
from interpolation_checkpoint import (InterpolationCheckpoint, checkpoint_metadata,
                                      default_checkpoint_path)
from utils import initialize_singleplane_state_from_outputs, rotate_state_to_observer_basis_inplace
from transport_xyz import warmup_numba_kernels_transport
from diffsky_routines import add_lensing, galaxy_steps_overlapping_chi_interval, galaxy_steps_overlapping_chi_interval_z0, parse_patch_list


nthreads = 128

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
    parser = argparse.ArgumentParser(description="Interpolate lensing into Diffsky catalogs, with automatic restart.")
    parser.add_argument('path_gals')
    parser.add_argument('synthetic_flag', help="S for synthetic halos; N for regular galaxies")
    parser.add_argument('patch_list', help="For example 0:20 (stop is excluded)")
    parser.add_argument('--start-plane', type=int, default=1,
                        help="First logged plane to process, 1-based (not a galaxy step). Use to bootstrap old runs.")
    parser.add_argument('--checkpoint', help="Progress JSON path; default is a deterministic name in the working directory")
    parser.add_argument('--no-checkpoint', action='store_true', help="Disable progress tracking and automatic resume")
    parser.add_argument('--no-checkpoint-lock', action='store_true',
                        help="Skip flock on filesystems that do not support it; requires exclusive use of this checkpoint and its catalogs")
    args = parser.parse_args(argv)
    patches = list(dict.fromkeys(parse_patch_list(args.patch_list)))
    if not patches:
        parser.error('Patch list is empty')
    path_end = '.diffsky_gals.synthetic_halos.hdf5' if args.synthetic_flag == 'S' else '.diffsky_gals.hdf5'
    sim = LJ_simulation()
    if not 1 <= args.start_plane <= sim['nplanes']:
        parser.error(f"--start-plane must be between 1 and {sim['nplanes']}")
    if args.no_checkpoint and args.checkpoint:
        parser.error('--checkpoint and --no-checkpoint cannot be combined')
    if args.no_checkpoint:
        manager = nullcontext(None)
    else:
        path = args.checkpoint or default_checkpoint_path(args.path_gals, path_end, patches)
        metadata = checkpoint_metadata(sim, args.path_gals, path_end, patches,
                                       mag_cols, flux_cols, step_list_gals)
        manager = InterpolationCheckpoint(path, metadata, args.start_plane,
                                          use_lock=not args.no_checkpoint_lock)
    with manager as checkpoint:
        start_plane = checkpoint.start_plane if checkpoint is not None else args.start_plane
        run_interpolation(sim, args.path_gals, path_end, patches, start_plane, checkpoint)


def run_interpolation(sim, path_gals, path_end, patch_list, start_plane, checkpoint):
    def catalog_files(steps):
        return [os.path.join(path_gals, f"lc_cores-{step}.{patch}{path_end}")
                for step in steps for patch in patch_list]

    nside = sim['nside']
    theta0 = phi0 = None
    for step_idx in range(start_plane - 1, sim['nplanes']):
        step_idx2 = step_idx + 1
        plane_start = time.perf_counter()
        gal_steps, chi_min, chi_max = galaxy_steps_overlapping_chi_interval(
            step_idx, step_idx2, step_list_gals, sim)
        jobs = [(catalog_files(gal_steps), chi_min, chi_max, False)]
        if step_idx == 0:
            low_steps, low_chi_max = galaxy_steps_overlapping_chi_interval_z0(
                step_idx, step_list_gals, sim)
            jobs.append((catalog_files(low_steps), 0, low_chi_max, True))
        pending_jobs = []
        for filenames, lower, upper, first_plane in jobs:
            pending = (checkpoint.pending_files(filenames, step_idx, first_plane)
                       if checkpoint is not None else filenames)
            if len(pending) < len(filenames):
                print(f"[plane {step_idx + 1}/{sim['nplanes']}] Restart: skipping {len(filenames) - len(pending)} completed catalog intervals (first_plane={first_plane})", flush=True)
            if pending:
                pending_jobs.append((pending, lower, upper, first_plane))
        if not pending_jobs:
            print(f"[plane {step_idx + 1}/{sim['nplanes']}] No pending catalog updates; skipping map reads", flush=True)
            continue
        if theta0 is None:
            print('Setting up', flush=True)
            warmup_numba_kernels_transport()
            theta0, phi0 = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
        print(f"[plane {step_idx + 1}/{sim['nplanes']}] Loading and rotating source states {step_idx}, {step_idx2}", flush=True)
        state = initialize_singleplane_state_from_outputs(sim, step_idx)
        rotate_state_to_observer_basis_inplace(state, theta0, phi0)
        state2 = initialize_singleplane_state_from_outputs(sim, step_idx2)
        rotate_state_to_observer_basis_inplace(state2, theta0, phi0)
        print(f"[plane {step_idx + 1}/{sim['nplanes']}] States ready; chi=[{chi_min:.6g}, {chi_max:.6g}) Mpc/h; galaxy steps={list(gal_steps)}", flush=True)
        for filenames, lower, upper, first_plane in pending_jobs:
            if first_plane:
                print(f"[plane 1/{sim['nplanes']}] Processing galaxies below the first source plane: chi=[0, {upper:.6g}) Mpc/h", flush=True)
            add_lensing(filenames, state, state2, nside, lower, upper,
                        mag_cols, flux_cols, sim, first_plane=first_plane,
                        checkpoint=checkpoint)
        del state, state2
        print(f"[plane {step_idx + 1}/{sim['nplanes']}] Complete in {time.perf_counter() - plane_start:.1f} s", flush=True)
    print(f"Interpolation complete: configured plane intervals {start_plane}..{sim['nplanes']} processed.", flush=True)


if __name__ == "__main__":
    main()
