import sys

import healpy as hp
import h5py
import numpy as np
import pyccl as ccl

from hacc_sims import LJ_simulation, step2z
from transport_xyz import transport_A_rows_xyz_basis_numba


DEFAULT_PATH_MAPS = "/pscratch/sd/p/plarsen/ray_tracing_tests/map_8192_2p5_wiener_seamfix_alloutputs/"

STEP_LIST_GALS = np.array([
    "121", "124", "127", "131", "134", "137", "141", "144", "148",
    "151", "155", "159", "163", "167", "171", "176", "180", "184",
    "189", "194", "198", "203", "208", "213", "219", "224", "230",
    "235", "241", "247", "253", "259", "266", "272", "279", "286",
    "293", "300", "307", "315", "323", "331", "338", "347", "355",
    "365", "373", "382", "392", "401", "411", "421", "432", "442",
    "453", "464", "475", "487",
])[::-1]

MAG_COLS = []
for band in ["g", "r", "i", "z", "y", "u"]:
    for comp in ["", "_bulge", "_disk", "_knots"]:
        MAG_COLS.append("lsst_" + band + comp)
for band in [
    "F062", "F087", "F106", "F129", "F146", "F158", "F184", "F213",
    "Grism_0thOrder", "Grism_1stOrder", "Prism",
]:
    for comp in ["", "_bulge", "_disk", "_knots"]:
        MAG_COLS.append("roman_" + band + comp)

FLUX_COLS = ["Halpha", "OII", "OIII"]


def wrap_angles_ra_dec(ra, dec):
    dec = np.clip(dec, -90.0 + 1e-8, 90.0 - 1e-8)
    ra = np.mod(ra, 360.0)
    ra = np.where((dec < -90.0 + 1e-8) | (dec > 90.0 - 1e-8), 0.0, ra)
    return ra, dec


def linear_interp(x, x0, x1, y0, y1):
    """Linearly interpolate y(x), allowing x/y to be arrays."""
    if x1 == x0:
        return 0.5 * (y0 + y1)
    w = (x - x0) / (x1 - x0)
    return (1.0 - w) * y0 + w * y1


def parse_patch_list(raw):
    """Accept '0,1,2', '0:4', or a plain string of one-character patch ids."""
    if "," in raw:
        return [x.strip() for x in raw.split(",") if x.strip()]
    if ":" in raw:
        start, stop = raw.split(":", 1)
        return [str(i) for i in range(int(start), int(stop))]
    if raw.isdigit() and len(raw) > 1:
        return [raw]
    return list(raw)


def source_basis_suffix(step_idx, sim):
    if step_idx > sim["nplanes"] - 1:
        return "tail_plane_" + str(step_idx - sim["nplanes"]) + "_source_basis.fits"

    step_high = sim["step_list_max"][step_idx]
    step_low = sim["step_list_min"][step_idx]
    return str(step_high) + "_" + str(step_low) + "_source_basis.fits"


def plane_redshift(step_idx, sim):
    if step_idx > sim["nplanes"] - 1:
        raise ValueError("galaxy interpolation currently expects simulation source planes")
    step = sim["step_list_max"][step_idx] - 1
    return step2z(step, sim["zfin"], sim["zinit"], sim["nsteps"])


def chi_from_redshift(z, sim):
    z = np.asarray(z, dtype=np.float64)
    return ccl.background.comoving_radial_distance(
        sim["cosmo_ccl"],
        a=1.0 / (1.0 + z),
    ) * sim["h"]


def plane_chi(step_idx, sim):
    return chi_from_redshift(plane_redshift(step_idx, sim), sim)


def find_bounding_plane_indices(snapshot_step, sim):
    """Return source-plane indices bracketing a galaxy lightcone snapshot step."""
    step_max = np.asarray(sim["step_list_max"], dtype=np.int64)
    step_min = np.asarray(sim["step_list_min"], dtype=np.int64)
    snapshot_step = int(snapshot_step)

    exact = np.where(step_max == snapshot_step)[0]
    if exact.size:
        idx = int(exact[0])
        return idx, idx

    if snapshot_step > step_max[0]:
        return 0, 0

    if snapshot_step < step_max[-1]:
        idx = step_max.size - 1
        return idx, idx

    lower_candidates = step_min[step_min >= snapshot_step]
    upper_candidates = step_max[step_max <= snapshot_step]
    if lower_candidates.size == 0 or upper_candidates.size == 0:
        idx = int(np.argmin(np.abs(step_max - snapshot_step)))
        return idx, idx

    step_low = int(lower_candidates[-1])
    step_high = int(upper_candidates[0])

    idx_low = int(np.where(step_max == step_low)[0][0])
    idx_high = int(np.where(step_max == step_high)[0][0])
    return idx_low, idx_high


def neighbor_candidates_fixed_depth(nside, pix0, depth=1, nest=False):
    pix0 = np.asarray(pix0, dtype=np.int64).ravel()
    candidates = [pix0]
    frontier = pix0[:, None]

    for _ in range(depth):
        nb = hp.get_all_neighbours(nside, frontier.ravel(), nest=nest).T
        nb = nb.reshape(pix0.size, -1)
        nb = np.where(nb >= 0, nb, pix0[:, None])
        candidates.append(nb)
        frontier = nb

    return np.concatenate(
        [c[:, None] if c.ndim == 1 else c for c in candidates],
        axis=1,
    )


def local_basis(theta, phi):
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)

    n = np.stack([st * cp, st * sp, ct], axis=0)
    e_theta = np.stack([ct * cp, ct * sp, -st], axis=0)
    e_phi = np.stack([-sp, cp, np.zeros_like(theta)], axis=0)
    return n, e_theta, e_phi


def interpolate_gals_to_source_basis(
    step_idx,
    sim,
    path_maps,
    dec_gal,
    ra_gal,
    depth,
    chunk_size=100_000,
):
    """Interpolate native ray-tracing outputs to catalog source positions.

    Native maps are indexed by observed HEALPix pixel. ``theta`` and ``phi`` are
    the source-plane beta position reached by each observed ray, and A is stored
    in the carried/source basis. This routine inverts beta(theta_obs) locally and
    returns observer-basis lensing quantities at the inferred observed position.
    """
    nside = int(sim["nside"])
    dec_gal = np.asarray(dec_gal, dtype=np.float64).ravel()
    ra_gal = np.asarray(ra_gal, dtype=np.float64).ravel()

    if dec_gal.shape != ra_gal.shape:
        raise ValueError("dec_gal and ra_gal must have the same shape")

    suffix = source_basis_suffix(step_idx, sim)
    if path_maps and not path_maps.endswith("/"):
        path_maps = path_maps + "/"

    theta_beta_map = hp.read_map(path_maps + "theta_" + suffix, dtype=np.float32)
    phi_beta_map = hp.read_map(path_maps + "phi_" + suffix, dtype=np.float32)

    theta_gal = np.deg2rad(90.0 - dec_gal)
    phi_gal = np.mod(np.deg2rad(ra_gal), 2.0 * np.pi)
    n_gal = np.asarray(hp.ang2vec(theta_gal, phi_gal))

    pix_guess = hp.ang2pix(nside, theta_gal, phi_gal, nest=False)
    best_pix = np.empty(theta_gal.size, dtype=np.int64)

    for start in range(0, theta_gal.size, chunk_size):
        stop = min(start + chunk_size, theta_gal.size)
        cand = neighbor_candidates_fixed_depth(
            nside,
            pix_guess[start:stop],
            depth=depth,
            nest=False,
        )
        gx, gy, gz = hp.ang2vec(theta_gal[start:stop], phi_gal[start:stop])
        bx, by, bz = hp.ang2vec(theta_beta_map[cand], phi_beta_map[cand])
        dots = bx * gx[:, None] + by * gy[:, None] + bz * gz[:, None]
        best = np.argmax(dots, axis=1)
        best_pix[start:stop] = cand[np.arange(stop - start), best]

    theta_obs0, phi_obs0 = hp.pix2ang(nside, best_pix, nest=False)
    theta_beta0 = theta_beta_map[best_pix]
    phi_beta0 = phi_beta_map[best_pix]

    dA11_map = hp.read_map(path_maps + "dA_11_" + suffix, dtype=np.float32)
    dA22_map = hp.read_map(path_maps + "dA_22_" + suffix, dtype=np.float32)
    A12_map = hp.read_map(path_maps + "A_12_" + suffix, dtype=np.float32)
    A21_map = hp.read_map(path_maps + "A_21_" + suffix, dtype=np.float32)
    psi_map = hp.read_map(path_maps + "psi_" + suffix, dtype=np.float32)

    A11 = 1.0 + dA11_map[best_pix]
    A22 = 1.0 + dA22_map[best_pix]
    A12 = A12_map[best_pix]
    A21 = A21_map[best_pix]
    psi = psi_map[best_pix]

    n_beta0, e_th_beta0, e_ph_beta0 = local_basis(theta_beta0, phi_beta0)
    dot = np.sum(n_gal * n_beta0, axis=0)
    r3 = n_gal - dot[None, :] * n_beta0
    d_beta_theta = np.sum(r3 * e_th_beta0, axis=0)
    d_beta_phi = np.sum(r3 * e_ph_beta0, axis=0)

    c = np.cos(psi)
    s = np.sin(psi)
    dbt = d_beta_theta.copy()
    dbp = d_beta_phi.copy()
    d_beta_theta = c * dbt + s * dbp
    d_beta_phi = -s * dbt + c * dbp

    det = A11 * A22 - A12 * A21
    if np.any(np.abs(det) < 1e-12):
        raise RuntimeError("Singular or nearly singular local Jacobian in interpolation")

    d_obs_theta = (A22 * d_beta_theta - A12 * d_beta_phi) / det
    d_obs_phi = (-A21 * d_beta_theta + A11 * d_beta_phi) / det

    theta_obs = theta_obs0 + d_obs_theta
    phi_obs = phi_obs0 + d_obs_phi / np.maximum(np.sin(theta_obs0), 1e-12)
    theta_obs = np.clip(theta_obs, 1e-12, np.pi - 1e-12)
    phi_obs = np.mod(phi_obs, 2.0 * np.pi)

    theta_beta_gal = hp.get_interp_val(theta_beta_map, theta_obs, phi_obs, nest=False)
    phi_beta_gal = hp.get_interp_val(phi_beta_map, theta_obs, phi_obs, nest=False)
    A11_gal = 1.0 + hp.get_interp_val(dA11_map, theta_obs, phi_obs, nest=False)
    A22_gal = 1.0 + hp.get_interp_val(dA22_map, theta_obs, phi_obs, nest=False)
    A12_gal = hp.get_interp_val(A12_map, theta_obs, phi_obs, nest=False)
    A21_gal = hp.get_interp_val(A21_map, theta_obs, phi_obs, nest=False)

    psi_sin = hp.get_interp_val(np.sin(psi_map), theta_obs, phi_obs, nest=False)
    psi_cos = hp.get_interp_val(np.cos(psi_map), theta_obs, phi_obs, nest=False)
    psi_gal = np.arctan2(psi_sin, psi_cos)

    del theta_beta_map, phi_beta_map, dA11_map, dA22_map, A12_map, A21_map, psi_map

    A11_obs, A12_obs, A21_obs, A22_obs = transport_A_rows_xyz_basis_numba(
        A11_gal,
        A12_gal,
        A21_gal,
        A22_gal,
        theta_beta_gal,
        phi_beta_gal,
        theta_obs,
        phi_obs,
        psi_gal,
    )

    kappa = 1.0 - 0.5 * (A11_obs + A22_obs)
    shear1 = -0.5 * (A11_obs - A22_obs)
    shear2 = 0.5 * (A12_obs + A21_obs)
    w = 0.5 * (A12_obs - A21_obs)
    mu = 1.0 / (A11_obs * A22_obs - A12_obs * A21_obs)

    ra_obs = np.degrees(phi_obs)
    dec_obs = 90.0 - np.degrees(theta_obs)
    ra_obs, dec_obs = wrap_angles_ra_dec(ra_obs, dec_obs)

    return ra_obs, dec_obs, kappa, shear1, shear2, w, mu


def interpolate_between_source_planes(
    snapshot_step,
    z_gal,
    sim,
    path_maps,
    dec_gal,
    ra_gal,
    depth,
    chunk_size=100_000,
):
    idx_low, idx_high = find_bounding_plane_indices(snapshot_step, sim)

    low = interpolate_gals_to_source_basis(
        idx_low,
        sim,
        path_maps,
        dec_gal,
        ra_gal,
        depth,
        chunk_size=chunk_size,
    )
    if idx_low == idx_high:
        return low

    high = interpolate_gals_to_source_basis(
        idx_high,
        sim,
        path_maps,
        dec_gal,
        ra_gal,
        depth,
        chunk_size=chunk_size,
    )

    chi_low = plane_chi(idx_low, sim)
    chi_high = plane_chi(idx_high, sim)
    if chi_high < chi_low:
        chi_low, chi_high = chi_high, chi_low
        low, high = high, low

    ra_low, dec_low, kappa_low, shear1_low, shear2_low, w_low, _ = low
    ra_high, dec_high, kappa_high, shear1_high, shear2_high, w_high, _ = high

    theta_low = np.deg2rad(90.0 - dec_low)
    phi_low = np.deg2rad(ra_low)
    theta_high = np.deg2rad(90.0 - dec_high)
    phi_high = np.deg2rad(ra_high)
    n_low = np.asarray(hp.ang2vec(theta_low, phi_low))
    n_high = np.asarray(hp.ang2vec(theta_high, phi_high))

    chi_gal = chi_from_redshift(z_gal, sim)
    t = (chi_gal - chi_low) / (chi_high - chi_low)
    t = np.clip(t, 0.0, 1.0)

    n_obs = (1.0 - t)[None, :] * n_low + t[None, :] * n_high
    n_obs /= np.sqrt(np.sum(n_obs**2, axis=0))
    theta_obs = np.arccos(np.clip(n_obs[2], -1.0, 1.0))
    phi_obs = np.mod(np.arctan2(n_obs[1], n_obs[0]), 2.0 * np.pi)

    ra_obs = np.degrees(phi_obs)
    dec_obs = 90.0 - np.degrees(theta_obs)
    ra_obs, dec_obs = wrap_angles_ra_dec(ra_obs, dec_obs)

    kappa = linear_interp(z_gal, z_low, z_high, kappa_low, kappa_high)
    shear1 = linear_interp(z_gal, z_low, z_high, shear1_low, shear1_high)
    shear2 = linear_interp(z_gal, z_low, z_high, shear2_low, shear2_high)
    w = linear_interp(z_gal, z_low, z_high, w_low, w_high)
    mu = 1.0 / ((1.0 - kappa) ** 2 - shear1**2 - shear2**2 + w**2)

    return ra_obs, dec_obs, kappa, shear1, shear2, w, mu


def ensure_unlensed_backups(gal_file):
    data = gal_file["data"]
    if "unlensed_magnitudes" not in data:
        data.create_group("unlensed_magnitudes")
    if "unlensed_fluxes" not in data:
        data.create_group("unlensed_fluxes")

    for mag in MAG_COLS:
        if mag in data and mag not in data["unlensed_magnitudes"]:
            data["unlensed_magnitudes"].create_dataset(mag, data=data[mag][:])
    for flux in FLUX_COLS:
        if flux in data and flux not in data["unlensed_fluxes"]:
            data["unlensed_fluxes"].create_dataset(flux, data=data[flux][:])


def replace_dataset(group, name, values):
    if name in group:
        del group[name]
    group.create_dataset(name, data=values)


def write_lensing_to_galaxy_file(gal_file, ra_obs, dec_obs, kappa, shear1, shear2, w, mu):
    data = gal_file["data"]

    for name, values in [
        ("ra_obs", ra_obs),
        ("dec_obs", dec_obs),
        ("kappa", kappa),
        ("shear1", shear1),
        ("shear2", shear2),
        ("w", w),
        ("magnification", mu),
    ]:
        replace_dataset(data, name, values)

    ensure_unlensed_backups(gal_file)

    abs_mu = np.maximum(np.abs(mu), 1e-30)
    for mag in MAG_COLS:
        if mag in data and mag in data["unlensed_magnitudes"]:
            data[mag][:] = data["unlensed_magnitudes"][mag][:] - 2.5 * np.log10(abs_mu)
    for flux in FLUX_COLS:
        if flux in data and flux in data["unlensed_fluxes"]:
            data[flux][:] = data["unlensed_fluxes"][flux][:] * mu


def process_galaxy_file(step, patch, sim, path_gals, path_maps, path_end, depth, chunk_size):
    filename = path_gals + "lc_cores-" + str(step) + "." + str(patch) + path_end
    with h5py.File(filename, "r+") as gal_file:
        data = gal_file["data"]
        ra_in = data["ra_nfw"][:]
        dec_in = data["dec_nfw"][:]
        z_in = data["redshift_true"][:]

        ra_obs, dec_obs, kappa, shear1, shear2, w, mu = interpolate_between_source_planes(
            int(step),
            z_in,
            sim,
            path_maps,
            dec_in,
            ra_in,
            depth,
            chunk_size=chunk_size,
        )
        write_lensing_to_galaxy_file(gal_file, ra_obs, dec_obs, kappa, shear1, shear2, w, mu)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) < 3:
        raise SystemExit(
            "usage: python interpolation.py PATH_GALS SYNTHETIC_FLAG PATCH_LIST "
            "[PATH_MAPS] [DEPTH] [CHUNK_SIZE]"
        )

    sim = LJ_simulation()
    path_gals = argv[0]
    synthetic = argv[1]
    patch_list = parse_patch_list(argv[2])
    path_maps = argv[3] if len(argv) > 3 else sim.get("output_path_rt", DEFAULT_PATH_MAPS)
    depth = int(argv[4]) if len(argv) > 4 else 2
    chunk_size = int(argv[5]) if len(argv) > 5 else 100_000

    if path_gals and not path_gals.endswith("/"):
        path_gals = path_gals + "/"
    if path_maps and not path_maps.endswith("/"):
        path_maps = path_maps + "/"

    if synthetic == "S":
        path_end = ".diffsky_gals.synthetic_halos.hdf5"
    else:
        path_end = ".diffsky_gals.hdf5"

    for step in STEP_LIST_GALS:
        print(step, flush=True)
        for patch in patch_list:
            print("  patch", patch, flush=True)
            process_galaxy_file(
                step,
                patch,
                sim,
                path_gals,
                path_maps,
                path_end,
                depth,
                chunk_size,
            )


if __name__ == "__main__":
    main()
