import numpy as np
import h5py
from hacc_sims import step2z
from simulation import get_chi_step, chi_from_redshift
from angle_updates import wrap_angles_ra_dec
from interpolation_routines import get_lensing_interpolation, redshift_interpolation, get_plane_vals


def parse_patch_list(raw):
    """
    Read in list of patches we want to run over
    Accept '0,1,2', '0:4', or a plain string of one-character patch ids.
    """
    if "," in raw:
        return [x.strip() for x in raw.split(",") if x.strip()]
    if ":" in raw:
        start, stop = raw.split(":", 1)
        return [str(i) for i in range(int(start), int(stop))]
    if raw.isdigit() and len(raw) > 1:
        return [raw]
    return list(raw)


def galaxy_file_intervals_chi(step_list_gals, sim):
    """
    Determine chi intervals for the galaxy catalog steps
    """
    labels = np.asarray([int(s) for s in step_list_gals], dtype=np.int64)

    intervals = []
    for i, step_start in enumerate(labels):
        z_start = step2z(step_start, sim["zfin"], sim["zinit"], sim["nsteps"])
        chi_start = chi_from_redshift(z_start, sim)

        if i - 1 >= 0:
            step_end = labels[i - 1]
            z_end = step2z(step_end, sim["zfin"], sim["zinit"], sim["nsteps"])
            chi_end = chi_from_redshift(z_end, sim)
        else:
            z_end = 0.0
            chi_end = 0.0

        chi_lo = min(chi_start, chi_end)
        chi_hi = max(chi_start, chi_end)

        intervals.append({
            "label": str(step_start),
            "step_start": int(step_start),
            "step_end": int(labels[i - 1]) if i - 1 >= 0 else None,
            "chi_lo": float(chi_lo),
            "chi_hi": float(chi_hi),
            "z_start": float(z_start),
            "z_end": float(z_end),
        })

    return intervals   

def galaxy_steps_overlapping_chi_interval(step_idx, step_idx2, step_list_gals, sim):
    """
    Work out all galaxy steps that overlap the source plane inputs 
    """    

    chi_a = get_chi_step(step_idx, sim)[2]
    chi_b = get_chi_step(step_idx2, sim)[2]

    chi_lo = min(chi_a, chi_b)
    chi_hi = max(chi_a, chi_b)

    intervals = galaxy_file_intervals_chi(step_list_gals, sim)

    steps = []
    for item in intervals:
        overlaps = (item["chi_hi"] >= chi_lo) and (item["chi_lo"] <= chi_hi)
        if overlaps:
            steps.append(item["label"])

    return steps, chi_lo, chi_hi


def galaxy_steps_overlapping_chi_interval_z0(step_idx, step_list_gals, sim):
    """
    Work out galaxy steps that are below the minimum source plane 
    
    chi_a here is the mean of the lowest source plane. We want to catch every galaxy  
    below this source plane and make sure it is properly set.
    """
    chi_a = get_chi_step(step_idx, sim)[2]

    intervals = galaxy_file_intervals_chi(step_list_gals, sim)

    steps = []
    for item in intervals:
        overlaps = (item["chi_lo"] <= chi_a)
        if overlaps:
            steps.append(item["label"])

    return steps, chi_a



def duplicate_unlensed_data(f, mag_cols, flux_cols):
    """
    Duplicate unlensed data into new datasets
    """
    if "data/unlensed_magnitudes" not in f:
        f['data'].create_group("unlensed_magnitudes")
    if "data/unlensed_fluxes" not in f:
        f['data'].create_group("unlensed_fluxes")

    for mag in mag_cols:
        str_mag = "data/unlensed_magnitudes/" + mag
        if str_mag not in f:
            f.create_dataset(str_mag, data = f["data/"+mag][:])
    for flux in flux_cols:
        str_flux = "data/unlensed_fluxes/" + flux
        if str_flux not in f:
            f.create_dataset(str_flux, data = f["data/"+flux][:])
    return

def initialize_lensing_data(f, ngals):
    """
    Initialize lensing data in files
    """
    if "data/ra_obs" not in f:
        f.create_dataset("data/ra_obs", data=f["data/ra_nfw"][:])
    if "data/dec_obs" not in f:
        f.create_dataset("data/dec_obs", data=f["data/dec_nfw"][:])
    if "data/kappa" not in f:
        f.create_dataset("data/kappa", data=np.zeros(ngals, dtype='f4'))
    if "data/shear1" not in f:
        f.create_dataset("data/shear1", data=np.zeros(ngals, dtype='f4'))
    if "data/shear2" not in f:
        f.create_dataset("data/shear2", data=np.zeros(ngals, dtype='f4'))
    if "data/lensing_w" not in f:
        f.create_dataset("data/lensing_w", data=np.zeros(ngals, dtype='f4'))
    if "data/magnification" not in f:
        f.create_dataset("data/magnification", data=np.ones(ngals, dtype='f4'))
    return

def update_gal_file(gal_file, mask, ra_val, dec_val, kappa_val, shear1_val, shear2_val, w_val, mag_cols, flux_cols):
    """
    Update values in file 
    """
    gal_file['data']['ra_obs'][mask] = ra_val
    gal_file['data']['dec_obs'][mask] = dec_val
    gal_file['data']['kappa'][mask] = kappa_val
    gal_file['data']['shear1'][mask] = shear1_val
    gal_file['data']['shear2'][mask] = shear2_val
    gal_file['data']['lensing_w'][mask] = w_val

    deta = (1-kappa_val)**2-(shear1_val**2+shear2_val**2) + w_val**2
    mag_vals = 1./(deta)
    gal_file["data"]["magnification"][mask] = mag_vals[:]

    for mag in mag_cols:
        gal_file['data'][mag][mask] = gal_file['data']['unlensed_magnitudes'][mag][mask] - 2.5*np.log10(np.abs(mag_vals[:]))
    for flux in flux_cols:
        gal_file['data'][flux][mask] = gal_file['data']['unlensed_fluxes'][flux][mask] * mag_vals

    return

def add_lensing(filenames, state, state2, nside, chi_min, chi_max, mag_cols, flux_cols, sim, first_plane=False):
    """
    Add lensing to galaxy files  
    """

    for filename in filenames:
        with h5py.File(filename, "r+") as gal_file:
            data = gal_file["data"]
            ra_in = data["ra_nfw"][:]
            dec_in = data["dec_nfw"][:]
            z_in = data["redshift_true"][:]
            ngals = ra_in.size

            # determine which galaxies need updating
            chi_gal = chi_from_redshift(z_in, sim)
            mask = (chi_gal < chi_max) * (chi_gal >= chi_min)

            if np.sum(mask)==0:
                continue

            # upper and lower planes then interpolation
            plane_vals = get_lensing_interpolation(state, ra_in[mask], dec_in[mask], nside)
            if not first_plane:
                plane_vals2 = get_lensing_interpolation(state2, ra_in[mask], dec_in[mask], nside)
                ra_val, dec_val, kappa_val, shear1_val, shear2_val, w_val = redshift_interpolation(chi_gal[mask], chi_min, chi_max, plane_vals, plane_vals2)
            else: 
                ra_val, dec_val, kappa_val, shear1_val, shear2_val, w_val = get_plane_vals(chi_gal[mask], chi_min, chi_max, plane_vals)

            ra_val, dec_val =  wrap_angles_ra_dec(ra_val, dec_val)

            # updating the file data

            duplicate_unlensed_data(gal_file, mag_cols, flux_cols)
            initialize_lensing_data(gal_file, ngals)

            update_gal_file(gal_file, mask, ra_val, dec_val, kappa_val, shear1_val, shear2_val, w_val, mag_cols, flux_cols)

    return

