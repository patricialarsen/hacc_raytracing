import numpy as np
import healpy as hp
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
    neighbs_level = {}
    bad = {}
    for i in range(levels):
        if i==0:
            neighbs = hp.get_all_neighbours(nside,pix_guess) # should be 8x8xN
        else:
            # assign fill value so healpix doesn't complain about finding neighbors for invalid pixels 
            neighbs_masked = np.copy(neighbs_level[i-1])
            neighbs_masked[bad[i-1]] = 0 
            neighbs = hp.get_all_neighbours(nside, neighbs_masked)
            # then mask out the neighbors of that fill pixel 
            neighbs[:,bad[i-1]] = -1 
        bad[i] = neighbs==-1
        neighbs_level[i] = neighbs
    pix_list = np.concatenate([neighbs_level[i].reshape(-1, gal_count) for i in range(levels)],axis=0)
    pix_list = np.concatenate([pix_list, np.array([pix_guess])],axis=0)
    pix_list = remove_duplicates(pix_list)
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
    
    
def get_weights(e1_pix, e2_pix, tol = 1.e-20):
    """
    Get barycentric triangle weights and triangle choice. 
    We pick the closest 4 points and test whether any combination 
    of these will form a triangle containing the point. 
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
    if not has_triangle.all():
        print(candidates)
        print(w0_a, w1_a, w2_a)
        print(det_a)
        raise ValueError("No valid triangle encompassing the galaxy")
        
    triangle_choice = np.argmax(candidates, axis=0)
    w0_all = np.stack([w0_a, w0_b, w0_c, w0_d], axis=0)
    w1_all = np.stack([w1_a, w1_b, w1_c, w1_d], axis=0)
    w2_all = np.stack([w2_a, w2_b, w2_c, w2_d], axis=0)
    gal_idx = np.arange(candidates.shape[1])

    w0 = w0_all[triangle_choice, gal_idx]
    w1 = w1_all[triangle_choice, gal_idx]
    w2 = w2_all[triangle_choice, gal_idx]
    
    return w0, w1, w2, triangle_choice


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

    

def get_lensing_interpolation(state, ra_in, dec_in, nside):
    """
    Get single plane lensing for a set of galaxies based on RA, dec position
    """
    ngals = hp.ang2vec(ra_in, dec_in, lonlat=True) # first galaxy only for now
    pix_list = get_candidates(nside, ngals, levels=2)

    x_pix, y_pix, z_pix = hp.ang2vec(state.theta[pix_list], state.phi[pix_list])
    x_obs_pix, y_obs_pix, z_obs_pix = hp.pix2vec(nside, pix_list)
    idx_dist = get_distance(ngals, nside, x_pix, y_pix, z_pix, pix_list)

    x_tan, y_tan, z_tan = get_tangent_vec(ngals, x_pix, y_pix, z_pix)
    x_tan_ref, y_tan_ref, z_tan_ref, x_tan_ref2, y_tan_ref2, z_tan_ref2 = get_tangent_basis(ngals)
    e1_pix, e2_pix = project(x_tan, y_tan, z_tan, x_tan_ref, y_tan_ref, z_tan_ref, x_tan_ref2, y_tan_ref2, z_tan_ref2)

    e1_pix = np.take_along_axis(e1_pix, idx_dist, axis=0)  
    e2_pix = np.take_along_axis(e2_pix, idx_dist, axis=0)

    idx_options = np.array([
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 3),
        (1, 2, 3),
    ])

    w0, w1, w2, triangle_choice  = get_weights(e1_pix, e2_pix)
    idx_list = idx_options[triangle_choice]
    x_obs_pix = np.take_along_axis(x_obs_pix, idx_dist, axis=0)[idx_list.T, np.arange(x_obs_pix.shape[1])[None, :]]  
    y_obs_pix = np.take_along_axis(y_obs_pix, idx_dist, axis=0)[idx_list.T, np.arange(x_obs_pix.shape[1])[None, :]] 
    z_obs_pix = np.take_along_axis(z_obs_pix, idx_dist, axis=0)[idx_list.T, np.arange(x_obs_pix.shape[1])[None, :]] 
    pix_list = np.take_along_axis(pix_list, idx_dist, axis=0)[idx_list.T, np.arange(pix_list.shape[1])[None, :]]


    x_obs, y_obs, z_obs = get_obs_position(w0, w1, w2, x_obs_pix, y_obs_pix, z_obs_pix)
    kappa_obs, shear1_obs, shear2_obs, w_obs = get_obs_lensing(w0, w1, w2, state, pix_list)

    plane_vals = {}
    plane_vals['x_obs'] = x_obs; plane_vals['y_obs'] = y_obs; plane_vals['z_obs']=z_obs; plane_vals['kappa_obs'] = kappa_obs;
    plane_vals['shear1_obs'] = shear1_obs; plane_vals['shear2_obs'] = shear2_obs; plane_vals['w_obs'] = w_obs;

    return plane_vals

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
    ra_val, dec_val = hp.vec2ang(np.array([x_val,y_val,z_val]),lonlat=True)
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
    ra_val, dec_val = hp.vec2ang(np.array([x_val,y_val,z_val]),lonlat=True)
    return ra_val, dec_val, plane_vals['kappa_obs'], plane_vals['shear1_obs'], plane_vals['shear2_obs'], plane_vals['w_obs']

