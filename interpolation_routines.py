import numpy as np
import healpy as hp
import json
import os
import tempfile
from utils import maps_from_jacobian


# Routines for interpolation that are generic to different galaxy catalogs 

def remove_duplicates(pixel_list):
    """ 
    Replace duplicate pixels in input pixel list with -1 values, 
    then reduce shape so that the size is the maximum number of 
    unique pixels x the number of galaxies.
    """ 

    keep = np.ones(pixel_list.shape, dtype=bool)
    pixel_list_sorted = np.sort(pixel_list, axis=0)

    # defining duplicates and sorting the -1s to the end 
    keep[1:, :] = pixel_list_sorted[1:, :] != pixel_list_sorted[:-1, :]
    pixel_list_sorted[~keep] = -1
    pixel_list_sorted = np.sort(pixel_list_sorted, axis=0)

    # reduce overall size 
    pixel_list_sorted = pixel_list_sorted[np.any(pixel_list_sorted != -1, axis=1),:]
    return pixel_list_sorted


def get_candidates(nside, ngal, levels=1):
    """ 
    get nested neighbour pixels with optional nesting level parameter 
    as candidate source plane locations for galaxy. As deflections are 
    small, a low level should be adequate. 

    vectorized over galaxies 
    """ 
    pix_guess = hp.vec2pix(nside,ngal[0],ngal[1],ngal[2])
    gal_count = len(pix_guess)
    pix_list = np.array([pix_guess])
    # Deduplicate each expansion rather than retaining exponentially many paths.
    for _ in range(levels):
        bad = pix_list < 0
        neighbs = hp.get_all_neighbours(nside, np.where(bad, 0, pix_list))
        neighbs[:, bad] = -1
        pix_list = remove_duplicates(np.concatenate(
            [pix_list, neighbs.reshape(-1, gal_count)], axis=0,
        ))
    return pix_list


def get_distance(ngal, nside, x_pix, y_pix, z_pix, pix):
    """ 
    Get the distance between a pixel location and the galaxy 
    """ 
    bad = pix<0
    dist = (x_pix-ngal[0])**2 + (y_pix-ngal[1])**2  + (z_pix-ngal[2])**2 
    dist = np.sqrt(dist)
    dist[bad] = 1e9
    idx_dist = np.argsort(dist,axis=0)
    return idx_dist

def get_tangent_vec(ngal, x_pix, y_pix, z_pix):
    """
    Get tangent vector component of pixel location to a galaxy norm vector  
    """
    scal = ngal[0]*x_pix + ngal[1]*y_pix + ngal[2]*z_pix
    x_tan = x_pix - scal*ngal[0] 
    y_tan = y_pix - scal*ngal[1] 
    z_tan = z_pix - scal*ngal[2] 
    return x_tan, y_tan, z_tan

def get_tangent_basis(ngal):
    """
    Get tangent basis to a galaxy norm vector  
    """
    idx_ref = np.argmin(np.abs(ngal),axis=0)
    
    x_ref = 1.0*(idx_ref==0)
    y_ref = 1.0*(idx_ref==1)
    z_ref = 1.0*(idx_ref==2)

    #scal_ref = ngal[idx_ref]*1.0 
    #scal_ref = np.take_along_axis(ngal, idx_ref, axis=0)
    scal_ref = ngal[idx_ref, np.arange(ngal.shape[1])]

    x_tan_ref = x_ref - scal_ref*ngal[0] 
    y_tan_ref = y_ref - scal_ref*ngal[1] 
    z_tan_ref = z_ref - scal_ref*ngal[2] 
    norm_ref = np.sqrt(x_tan_ref**2 + y_tan_ref**2 + z_tan_ref**2)

    x_tan_ref= x_tan_ref/norm_ref

    y_tan_ref= y_tan_ref/norm_ref
    z_tan_ref= z_tan_ref/norm_ref

    x_tan_ref2 = ngal[1] * z_tan_ref - ngal[2]* y_tan_ref
    y_tan_ref2 =  - ngal[0] * z_tan_ref  + ngal[2] * x_tan_ref
    z_tan_ref2 = ngal[0] * y_tan_ref  - ngal[1] * x_tan_ref 
    norm_ref2 = np.sqrt(x_tan_ref2**2 + y_tan_ref2**2 + z_tan_ref2**2)
    x_tan_ref2= x_tan_ref2/norm_ref2
    y_tan_ref2= y_tan_ref2/norm_ref2
    z_tan_ref2= z_tan_ref2/norm_ref2
    return x_tan_ref, y_tan_ref, z_tan_ref, x_tan_ref2, y_tan_ref2, z_tan_ref2

def project(x_tan, y_tan, z_tan, x_tan_ref, y_tan_ref, z_tan_ref, x_tan_ref2, y_tan_ref2, z_tan_ref2):
    """ 
    Project the tangent componet of a pixel to the tangent basis of a galaxy 
    """ 
    e1_pix = x_tan_ref*x_tan + y_tan_ref*y_tan+ z_tan_ref*z_tan
    e2_pix = x_tan_ref2*x_tan + y_tan_ref2*y_tan + z_tan_ref2*z_tan
    return e1_pix, e2_pix


def get_det(e1_pix, e2_pix, idx1, idx2, idx3):
    """ 
    Get determinant for a set of triangle points 
    """
    det = (e2_pix[idx2] - e2_pix[idx3]) * (e1_pix[idx1] - e1_pix[idx3]) + (e1_pix[idx3] - e1_pix[idx2]) * (e2_pix[idx1] - e2_pix[idx3])
    return det

def get_w(e1_pix, e2_pix, idx1, idx2, idx3, det):
    """
    Get barycentric weights for a set of triangle points 
    """
    w0 = ((e2_pix[idx2] - e2_pix[idx3]) * (0.0 - e1_pix[idx3]) + (e1_pix[idx3] - e1_pix[idx2]) * (0.0 - e2_pix[idx3])) / det
    w1 = ((e2_pix[idx3] - e2_pix[idx1]) * (0.0 - e1_pix[idx3]) + (e1_pix[idx1] - e1_pix[idx3]) * (0.0 - e2_pix[idx3])) / det
    w2 = 1.0 - w0 - w1
    return w0, w1, w2
    
    
def get_weights(e1_pix, e2_pix, tol = 1.e-20, ra_in=None, dec_in=None,
                allow_partial=False):
    """
    Return barycentric weights and three sorted-candidate indices per galaxy.
    Try the nearest four rays first, then additional triples only for failures.
    """

    # excluding each point 
    idx_a = (0,1,2)
    idx_b = (0,1,3)
    idx_c = (0,2,3)
    idx_d = (1,2,3)

    det_a = get_det(e1_pix, e2_pix, *idx_a)
    det_b = get_det(e1_pix, e2_pix, *idx_b)
    det_c = get_det(e1_pix, e2_pix, *idx_c)
    det_d = get_det(e1_pix, e2_pix, *idx_d)

    w0_a, w1_a, w2_a = get_w(e1_pix, e2_pix, *idx_a, det_a)
    w0_b, w1_b, w2_b = get_w(e1_pix, e2_pix, *idx_b, det_b)
    w0_c, w1_c, w2_c = get_w(e1_pix, e2_pix, *idx_c, det_c)
    w0_d, w1_d, w2_d = get_w(e1_pix, e2_pix, *idx_d, det_d)

    candidate_a = (w0_a>-tol)&(w1_a>-tol)&(w2_a>-tol)&(np.abs(det_a)>tol)
    candidate_b = (w0_b>-tol)&(w1_b>-tol)&(w2_b>-tol)&(np.abs(det_b)>tol)
    candidate_c = (w0_c>-tol)&(w1_c>-tol)&(w2_c>-tol)&(np.abs(det_c)>tol)
    candidate_d = (w0_d>-tol)&(w1_d>-tol)&(w2_d>-tol)&(np.abs(det_d)>tol)

    candidates = np.stack([candidate_a, candidate_b, candidate_c, candidate_d,], axis=0)
    has_triangle = np.any(candidates, axis=0)
    fallback_count = int(np.count_nonzero(~has_triangle))
    if fallback_count:
        print(f"[triangles] Trying additional candidate rays for {fallback_count}/{has_triangle.size} galaxies", flush=True)
    triangle_choice = np.argmax(candidates, axis=0)
    idx_options = np.array([idx_a, idx_b, idx_c, idx_d])
    triangle_indices = idx_options[triangle_choice].copy()
    gal_idx = np.arange(candidates.shape[1])
    w0 = np.stack([w0_a, w0_b, w0_c, w0_d], axis=0)[triangle_choice, gal_idx]
    w1 = np.stack([w1_a, w1_b, w1_c, w1_d], axis=0)[triangle_choice, gal_idx]
    w2 = np.stack([w2_a, w2_b, w2_c, w2_d], axis=0)[triangle_choice, gal_idx]

    # Introduce farther rays in distance order, without changing successful results.
    for farthest in range(4, e1_pix.shape[0]):
        if has_triangle.all():
            break
        for first in range(farthest - 1):
            if has_triangle.all():
                break
            for second in range(first + 1, farthest):
                unresolved = np.flatnonzero(~has_triangle)
                if unresolved.size == 0:
                    break
                x = e1_pix[:, unresolved]
                y = e2_pix[:, unresolved]
                triple = (first, second, farthest)
                det = get_det(x, y, *triple)
                with np.errstate(divide='ignore', invalid='ignore'):
                    a, b, c = get_w(x, y, *triple, det)
                valid = (
                    np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
                    & (a > -tol) & (b > -tol) & (c > -tol)
                    & (np.abs(det) > tol)
                )
                resolved = unresolved[valid]
                w0[resolved], w1[resolved], w2[resolved] = a[valid], b[valid], c[valid]
                triangle_indices[resolved] = triple
                has_triangle[resolved] = True

    if not has_triangle.all() and not allow_partial:
        failed = np.flatnonzero(~has_triangle)
        sample = failed[:5]
        weights = np.stack([
            [w0_a[sample], w1_a[sample], w2_a[sample]],
            [w0_b[sample], w1_b[sample], w2_b[sample]],
            [w0_c[sample], w1_c[sample], w2_c[sample]],
            [w0_d[sample], w1_d[sample], w2_d[sample]],
        ], axis=0)
        determinants = np.stack([d[sample] for d in [det_a, det_b, det_c, det_d]], axis=0)
        lines = [
            f"No valid triangle encompassing the galaxy: {failed.size}/{has_triangle.size} galaxies failed.",
            f"No enclosing triangle after testing all triples among {e1_pix.shape[0]} candidate slots; tol={tol:g}.",
            "Indices refer to the galaxy subset passed to spatial interpolation.",
        ]
        for sample_idx, idx in enumerate(sample):
            position = ""
            if ra_in is not None and dec_in is not None:
                position = f", RA={ra_in[idx]:.10g} deg, Dec={dec_in[idx]:.10g} deg"
            offsets = np.column_stack((e1_pix[:4, idx], e2_pix[:4, idx]))
            lines.append(f"Galaxy index {idx}{position}")
            lines.append(f"  nearest-ray tangent offsets (dimensionless): {offsets.tolist()}")
            lines.append(f"  determinants: {determinants[:, sample_idx].tolist()}")
            lines.append(f"  weights for triangles 012, 013, 023, 123: {weights[:, :, sample_idx].tolist()}")
        if failed.size > 5:
            lines.append(f"Only the first 5 failures are shown; {failed.size - 5} more omitted.")
        error = ValueError("\n".join(lines))
        error.failed_indices = failed
        raise error
        
    if fallback_count:
        recovered = fallback_count - int(np.count_nonzero(~has_triangle))
        print(f"[triangles] Recovered {recovered}/{fallback_count} galaxies using additional candidate rays", flush=True)
    if allow_partial:
        return w0, w1, w2, triangle_indices, has_triangle
    return w0, w1, w2, triangle_indices


def get_obs_position(w0, w1, w2, x_pix, y_pix, z_pix):
    """
    Get observer plane position from barycentric weights
    """
    x_new = w0*x_pix[0] + w1*x_pix[1] + w2*x_pix[2]
    y_new = w0*y_pix[0] + w1*y_pix[1] + w2*y_pix[2]
    z_new = w0*z_pix[0] + w1*z_pix[1] + w2*z_pix[2]
    return x_new, y_new, z_new 

def get_obs_lensing(w0, w1, w2, state, pix_list):
    """
    Get observer plane lensing variables from barycentric weights and plane state
    """    

    kappa_pix, shear1_pix, shear2_pix, w_pix = maps_from_jacobian(state.A_11[pix_list],state.A_12[pix_list],state.A_21[pix_list],state.A_22[pix_list])
    kappa_new = w0*kappa_pix[0] + w1*kappa_pix[1] + w2*kappa_pix[2]
    shear1_new = w0*shear1_pix[0] + w1*shear1_pix[1] + w2*shear1_pix[2]
    shear2_new = w0*shear2_pix[0] + w1*shear2_pix[1] + w2*shear2_pix[2]
    w_new = w0*w_pix[0] + w1*w_pix[1] + w2*w_pix[2]

    return kappa_new, shear1_new, shear2_new, w_new

    

def write_triangle_failure_diagnostics(state, ra_in, dec_in, nside, pix_list,
                                       idx_dist, failed, levels,
                                       catalog_filename=None, catalog_mask=None,
                                       distance_context=None, subset_indices=None):
    """Save full candidate geometry for up to five failures, without changing data."""
    from scipy.spatial import ConvexHull, QhullError

    arcsec = 180.0 / np.pi * 3600.0
    report = {
        'catalog_filename': catalog_filename,
        'plane_index': int(state.step_idx), 'nside': int(nside),
        'neighbor_levels': int(levels), 'failed_count': int(len(failed)),
        'attempt_subset_size': int(len(ra_in)),
        'failed_subset_indices': (subset_indices[failed] if subset_indices is not None else failed).tolist(),
        'offset_units': 'arcsec equivalent of projected unit-vector tangent components',
        'basis': 'get_tangent_basis at the input galaxy source direction',
        'samples': [],
    }
    if distance_context is not None:
        report['distance_interval'] = {
            'chi_min_mpc_h': float(distance_context['chi_min']),
            'chi_max_mpc_h': float(distance_context['chi_max']),
            'selection_rule': 'chi_min <= chi_gal < chi_max',
            'first_plane': bool(distance_context['first_plane']),
            'source_planes': distance_context['source_planes'],
        }
    sample = failed[:5]
    original_indices = subset_indices[sample] if subset_indices is not None else sample
    rows = np.flatnonzero(catalog_mask)[original_indices] if catalog_mask is not None else None
    for sample_idx, idx in enumerate(sample):
        original_idx = int(original_indices[sample_idx])
        pixels = pix_list[idx_dist[:, idx], idx]
        pixels = pixels[pixels >= 0]
        galaxy = hp.ang2vec(float(ra_in[idx]), float(dec_in[idx]), lonlat=True)
        observer = np.array(hp.pix2vec(nside, pixels))
        source = hp.ang2vec(state.theta[pixels], state.phi[pixels]).T
        basis = get_tangent_basis(galaxy[:, None])
        e1 = np.array(basis[:3]).reshape(3)
        e2 = np.array(basis[3:]).reshape(3)
        source_offsets = np.column_stack((e1 @ source, e2 @ source))
        observer_offsets = np.column_stack((e1 @ observer, e2 @ observer))
        dots = np.sum(observer * source, axis=0)
        crosses = np.linalg.norm(np.cross(observer.T, source.T), axis=1)
        deflections = np.arctan2(crosses, dots) * arcsec
        radius = np.arctan2(np.linalg.norm(np.cross(observer.T, galaxy), axis=1), galaxy @ observer) * arcsec
        neighbors = hp.get_all_neighbours(nside, pixels)
        boundary = np.any((neighbors >= 0) & ~np.isin(neighbors, pixels), axis=0)
        if levels > 1:
            inner = get_candidates(nside, galaxy[:, None], levels=levels - 1)[:, 0]
            outer_layer = ~np.isin(pixels, inner[inner >= 0])
        else:
            outer_layer = pixels != hp.vec2pix(nside, *galaxy)
        finite = np.all(np.isfinite(source_offsets), axis=1)
        hull_info = {'status': 'unavailable'}
        try:
            hull = ConvexHull(source_offsets[finite])
            edge_offsets = hull.equations[:, -1]
            max_offset = float(np.max(edge_offsets))
            hull_info = {
                'status': 'ok',
                'contains_galaxy': bool(max_offset <= 0.0),
                'max_signed_edge_distance_arcsec': max_offset * arcsec,
                'sign_convention': 'positive means outside; negative means inside',
                'vertex_pixel_ids': pixels[finite][hull.vertices].tolist(),
            }
        except (QhullError, ValueError) as exc:
            hull_info['reason'] = str(exc).splitlines()[0]
        obs_ra, obs_dec = hp.vec2ang(observer.T, lonlat=True)
        src_ra, src_dec = hp.vec2ang(source.T, lonlat=True)
        entry = {
            'subset_index': original_idx,
            'catalog_row': int(rows[sample_idx]) if rows is not None else None,
            'galaxy_ra_deg': float(ra_in[idx]), 'galaxy_dec_deg': float(dec_in[idx]),
            'valid_candidate_count': int(len(pixels)), 'source_convex_hull': hull_info,
            'candidates': [],
        }
        if distance_context is not None:
            # Context arrays are full catalog arrays; recover the original row.
            row = int(rows[sample_idx]) if rows is not None else original_idx
            chi = float(distance_context['catalog_chi_gal'][row])
            lower = float(distance_context['chi_min'])
            upper = float(distance_context['chi_max'])
            entry['redshift_true'] = float(distance_context['catalog_redshift_true'][row])
            entry['chi_gal_mpc_h'] = chi
            entry['within_selected_interval'] = bool(lower <= chi < upper)
            entry['distance_from_lower_bound_mpc_h'] = chi - lower
            entry['distance_to_upper_bound_mpc_h'] = upper - chi
            print(f"[diagnostics] Galaxy subset index {original_idx}: z_true={entry['redshift_true']:.10g}, "
                  f"chi_gal={chi:.10g} Mpc/h, interval=[{lower:.10g}, {upper:.10g}), "
                  f"within interval={entry['within_selected_interval']}", flush=True)
        for rank, pixel in enumerate(pixels):
            entry['candidates'].append({
                'source_distance_rank': rank, 'pixel_id': int(pixel),
                'observer_ra_deg': float(obs_ra[rank]), 'observer_dec_deg': float(obs_dec[rank]),
                'source_ra_deg': float(src_ra[rank]), 'source_dec_deg': float(src_dec[rank]),
                'saved_theta_rad': float(state.theta[pixel]), 'saved_phi_rad': float(state.phi[pixel]),
                'observer_offsets_arcsec': (observer_offsets[rank] * arcsec).tolist(),
                'source_offsets_arcsec': (source_offsets[rank] * arcsec).tolist(),
                'observer_radius_arcsec': float(radius[rank]),
                'ray_deflection_arcsec': float(deflections[rank]),
                'on_candidate_boundary': bool(boundary[rank]),
                'in_outer_neighbor_layer': bool(outer_layer[rank]),
                'finite_source_coordinates': bool(finite[rank]),
            })
        report['samples'].append(entry)
        print(f"[diagnostics] Galaxy subset index {original_idx}: hull status={hull_info['status']}, "
              f"contains galaxy={hull_info.get('contains_galaxy')}, "
              f"max signed edge distance={hull_info.get('max_signed_edge_distance_arcsec')} arcsec; "
              f"nearest-four boundary flags={boundary[:4].tolist()}; "
              f"ray deflection range={np.nanmin(deflections):.3g}..{np.nanmax(deflections):.3g} arcsec", flush=True)

    # Represent nonfinite diagnostic values as JSON null instead of invalid JSON.
    def json_safe(value):
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    with tempfile.NamedTemporaryFile(mode='w', prefix=f'triangle_failure_plane_{state.step_idx}_',
                                     suffix='.json', dir=os.getcwd(), delete=False) as output:
        json.dump(json_safe(report), output, indent=2, allow_nan=False)
        path = output.name
    print(f"[diagnostics] Catalog={catalog_filename}; plane={state.step_idx}; details saved to {path}", flush=True)
    return path


def get_lensing_interpolation(state, ra_in, dec_in, nside,
                              catalog_filename=None, catalog_mask=None,
                              distance_context=None, initial_levels=3, max_levels=8):
    """Interpolate with wider neighborhoods only for unresolved galaxies."""
    if initial_levels < 1 or max_levels < initial_levels:
        raise ValueError('Require 1 <= initial_levels <= max_levels')
    ra_in = np.asarray(ra_in)
    dec_in = np.asarray(dec_in)
    keys = ('x_obs', 'y_obs', 'z_obs', 'kappa_obs', 'shear1_obs', 'shear2_obs', 'w_obs')
    result = {key: np.empty(ra_in.size, dtype=np.float64) for key in keys}
    pending = np.arange(ra_in.size)
    for levels in range(initial_levels, max_levels + 1):
        if pending.size == 0:
            return result
        values, resolved, pix_list, idx_dist = _get_lensing_interpolation_at_level(
            state, ra_in[pending], dec_in[pending], nside, levels,
        )
        for key in keys:
            result[key][pending[resolved]] = values[key][resolved]
        failed = np.flatnonzero(~resolved)
        if failed.size == 0:
            if levels > initial_levels:
                print(f"[triangles] Source state {state.step_idx}: all remaining galaxies recovered at neighbor level {levels}", flush=True)
            return result
        if levels == max_levels:
            try:
                write_triangle_failure_diagnostics(
                    state, ra_in[pending], dec_in[pending], nside, pix_list, idx_dist,
                    failed, levels, catalog_filename, catalog_mask, distance_context,
                    subset_indices=pending,
                )
            except Exception as diagnostic_error:
                print(f"[diagnostics] Could not save diagnostics: {diagnostic_error}", flush=True)
            error = ValueError(
                f"No valid triangle encompassing the galaxy: {failed.size}/{ra_in.size} galaxies "
                f"failed after neighbor levels {initial_levels}..{max_levels}; "
                f"source state={state.step_idx}, catalog={catalog_filename}. "
                f"First failed subset indices: {pending[failed[:5]].tolist()}"
            )
            error.failed_indices = pending[failed]
            raise error
        pending = pending[failed]
        print(f"[triangles] Source state {state.step_idx}: expanding only {pending.size}/{ra_in.size} "
              f"unresolved galaxies from neighbor level {levels} to {levels + 1}", flush=True)
    return result


def _get_lensing_interpolation_at_level(state, ra_in, dec_in, nside, levels):
    """
    Attempt one neighborhood size; return partial values and a resolution mask.
    """
    ngals = hp.ang2vec(ra_in, dec_in, lonlat=True).T # first galaxy only for now
    pix_list = get_candidates(nside, ngals, levels=levels)

    x_pix, y_pix, z_pix = hp.ang2vec(state.theta[pix_list], state.phi[pix_list]).T
    x_obs_pix, y_obs_pix, z_obs_pix = hp.pix2vec(nside, pix_list)
    idx_dist = get_distance(ngals, nside, x_pix, y_pix, z_pix, pix_list)

    x_tan, y_tan, z_tan = get_tangent_vec(ngals, x_pix, y_pix, z_pix)
    x_tan_ref, y_tan_ref, z_tan_ref, x_tan_ref2, y_tan_ref2, z_tan_ref2 = get_tangent_basis(ngals)
    e1_pix, e2_pix = project(x_tan, y_tan, z_tan, x_tan_ref, y_tan_ref, z_tan_ref, x_tan_ref2, y_tan_ref2, z_tan_ref2)

    e1_pix = np.take_along_axis(e1_pix, idx_dist, axis=0)  
    e2_pix = np.take_along_axis(e2_pix, idx_dist, axis=0)

    # Missing HEALPix neighbors use -1; never allow their placeholder values
    # to become vertices in the wider triangle search.
    invalid = np.take_along_axis(pix_list < 0, idx_dist, axis=0)
    e1_pix[invalid] = np.nan
    e2_pix[invalid] = np.nan

    w0, w1, w2, idx_list, resolved = get_weights(
        e1_pix, e2_pix, ra_in=ra_in, dec_in=dec_in, allow_partial=True,
    )
    columns = np.flatnonzero(resolved)
    vertices = idx_list[resolved].T
    x_obs_pix = np.take_along_axis(x_obs_pix, idx_dist, axis=0)[vertices, columns[None, :]]
    y_obs_pix = np.take_along_axis(y_obs_pix, idx_dist, axis=0)[vertices, columns[None, :]]
    z_obs_pix = np.take_along_axis(z_obs_pix, idx_dist, axis=0)[vertices, columns[None, :]]
    selected_pixels = np.take_along_axis(pix_list, idx_dist, axis=0)[vertices, columns[None, :]]


    w0, w1, w2 = w0[resolved], w1[resolved], w2[resolved]
    x_obs, y_obs, z_obs = get_obs_position(w0, w1, w2, x_obs_pix, y_obs_pix, z_obs_pix)
    kappa_obs, shear1_obs, shear2_obs, w_obs = get_obs_lensing(w0, w1, w2, state, selected_pixels)

    keys = ('x_obs', 'y_obs', 'z_obs', 'kappa_obs', 'shear1_obs', 'shear2_obs', 'w_obs')
    plane_vals = {key: np.full(ra_in.size, np.nan) for key in keys}
    for key, values in zip(keys, (x_obs, y_obs, z_obs, kappa_obs, shear1_obs, shear2_obs, w_obs)):
        plane_vals[key][resolved] = values
    return plane_vals, resolved, pix_list, idx_dist

def linear_interp(x, x0, x1, y0, y1):
    """Linearly interpolate to estimate y at x, given two bounding points."""
    if x1 == x0:
        return 0.5 * (y0 + y1)
    w = (x - x0) / (x1 - x0)
    return (1 - w) * y0 + w * y1

def redshift_interpolation(chi_gal, chi_min, chi_max, plane_vals, plane_vals2):
    """
    Interpolate lensing between neighbouring source planes to get overall lensing values  
    """
    x_val = linear_interp(chi_gal, chi_min, chi_max, plane_vals['x_obs'], plane_vals2['x_obs'])
    y_val = linear_interp(chi_gal, chi_min, chi_max, plane_vals['y_obs'], plane_vals2['y_obs'])
    z_val = linear_interp(chi_gal, chi_min, chi_max, plane_vals['z_obs'], plane_vals2['z_obs'])
    ra_val, dec_val = hp.vec2ang(np.column_stack((x_val,y_val,z_val)),lonlat=True)
    kappa_val = linear_interp(chi_gal, chi_min, chi_max, plane_vals['kappa_obs'], plane_vals2['kappa_obs'])
    shear1_val = linear_interp(chi_gal, chi_min, chi_max, plane_vals['shear1_obs'], plane_vals2['shear1_obs'])
    shear2_val = linear_interp(chi_gal, chi_min, chi_max, plane_vals['shear2_obs'], plane_vals2['shear2_obs'])
    w_val = linear_interp(chi_gal, chi_min, chi_max, plane_vals['w_obs'], plane_vals2['w_obs'])

    return ra_val, dec_val, kappa_val, shear1_val, shear2_val, w_val


def get_plane_vals(chi_gal, chi_min, chi_max, plane_vals):
    """
    Get single plane values     
    """
    x_val = plane_vals['x_obs']
    y_val = plane_vals['y_obs']
    z_val = plane_vals['z_obs']
    ra_val, dec_val = hp.vec2ang(np.column_stack((x_val,y_val,z_val)),lonlat=True)
    return ra_val, dec_val, plane_vals['kappa_obs'], plane_vals['shear1_obs'], plane_vals['shear2_obs'], plane_vals['w_obs']

