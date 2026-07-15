#!/bin/bash/python

import numpy as np
import healpy as hp
import numpy as np
from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy import units as u
import pyccl as ccl

rho_crit0 = 2.77536627e11  # (Msun/h) / (Mpc/h)^3

def step2a(step,zfin,zstart,nsteps):
    aini = 1./(zstart+1.)
    afin = 1./(zfin+1)
    return aini + (afin - aini)/nsteps*(step+1)
def step2z(step, zfin, zstart, nsteps):
    return 1./step2a(step,zfin,zstart,nsteps)-1.


def LJ_simulation():
    LJ_simulation = {}
    LJ_simulation['name'] = "Last Journey"
    LJ_simulation['mpp'] = 2717396224.0
    LJ_simulation["Omega_m"] = 0.30964
    LJ_simulation['h'] = 0.6766
    LJ_simulation['rho_m0'] = LJ_simulation["Omega_m"] * rho_crit0
    LJ_simulation['nsteps']= 500
    LJ_simulation['zinit']= 200.0
    LJ_simulation['zfin'] = 0.0
    LJ_simulation['z_cmb'] = 1089.0
    LJ_simulation['pixwin_datapath']=None

    LJ_simulation['step_list_max'] = [484, 468, 453, 439, 426, 413, 400, 388, 376, 365, 355, 
                                  344, 334, 324, 315, 306, 297, 288, 280, 272, 264, 256,
                                  249, 241, 234, 227, 220, 213, 207, 200, 194, 188, 182, 
                                  176, 170, 165, 159, 154, 148, 143, 138, 133, 128, 123, 
                                  118, 114, 109, 105, 101,  96,  92,  88,  84,  80,  77,  
                                  73,  69,  66,  62,  59,  56, 53,  50,  47]

    LJ_simulation['step_list_min'] = [500, 484, 468, 453, 439, 426, 413, 400, 388, 376, 365, 355,
                                  344, 334, 324, 315, 306, 297, 288, 280, 272, 264, 256, 249, 
                                  241, 234, 227, 220, 213, 207, 200, 194, 188, 182, 176, 170,
                                  165, 159, 154, 148, 143, 138, 133, 128, 123, 118, 114, 109, 
                                  105, 101,  96,  92,  88,  84,  80,  77,  73,  69,  66,  62,  
                                  59,  56, 53,  50]

    # changeable parameters 
    LJ_simulation['nside'] = 8192 # note that reducing nside will cause a downgrade in the map, and will likely induce aliasing. 
    LJ_simulation['n_steps_cmb'] = 10 
    LJ_simulation['output_path_cls'] = '/pscratch/sd/p/plarsen/ray_tracing_tests/'
    LJ_simulation['output_path_rt'] = '/pscratch/sd/p/plarsen/ray_tracing_tests/map_8192_nocut_wiener_seamfix_alloutputs/'


    LJ_simulation['cosmo'] = FlatLambdaCDM(H0 = 67.66, Om0 = 0.30964, Ob0 = 0.04897468161869667, Tcmb0 = 0, Neff = 0) 
    LJ_simulation['cosmo_ccl'] = ccl.Cosmology(Omega_c=0.262, Omega_b=0.0488, h=0.6766, n_s=0.965, sigma8=0.8102,  transfer_function='boltzmann_camb',
matter_power_spectrum='halofit')


    # derived parameters
    LJ_simulation['nplanes'] =  len(LJ_simulation['step_list_max'])
    LJ_simulation['chi_cmb'] = ccl.background.comoving_radial_distance(LJ_simulation['cosmo_ccl'],a=1./(LJ_simulation['z_cmb']+1))*LJ_simulation['h']
    LJ_simulation['sim_z_max'] = step2z(LJ_simulation['step_list_max'][-1]-1, LJ_simulation['zfin'], LJ_simulation['zinit'], LJ_simulation['nsteps'])
    LJ_simulation['sim_chi_max'] = ccl.background.comoving_radial_distance(LJ_simulation['cosmo_ccl'],a=1./(LJ_simulation['sim_z_max']+1))*LJ_simulation['h'] 

    # correction inputs for LJ 
    LJ_simulation['signed_dist_pix'] = hp.read_map('/pscratch/sd/p/plarsen/ray_tracing_tests/signed_dist_pix_80.fits')
    LJ_simulation['A_array'] = np.loadtxt('/global/u2/p/plarsen/codes/cmblensing/A_vals_w7_20_30.txt')
    LJ_simulation['w_array'] = np.loadtxt('/global/u2/p/plarsen/codes/cmblensing/w_vals_w7_20_30.txt')

    # path to files 
    LJ_simulation['path_maps'] = '/pscratch/sd/p/plarsen/sharing/mass_sheets_LJ/'
    LJ_simulation['has_maps'] = True


    # additional testing values 
    LJ_simulation['nperst'] = [9536746.,77409054.,208227182.,380599129.,612972568.]
    LJ_simulation['nperst_steps'] = [0,1,2,3,4]

    LJ_simulation['path_alms'] = '/pscratch/sd/p/plarsen/ray_tracing_tests/alms/'
    LJ_simulation['has_alms'] = False
    return LJ_simulation



def FrontierE_simulation():
    FrontierE_simulation = {}
    FrontierE_simulation['name'] = "Frontier-E"
    FrontierE_simulation['mpp'] = 1342777220.0 # in Msun/h for dark matter-only run, but used as an approximation to get shot noise for hydro sim too  
    FrontierE_simulation["Omega_m"] = 0.30964468
    FrontierE_simulation['h'] = 0.6766
    FrontierE_simulation['rho_m0'] = FrontierE_simulation["Omega_m"] * rho_crit0
    FrontierE_simulation['nsteps']= 625
    FrontierE_simulation['zinit']= 200.0
    FrontierE_simulation['zfin'] = 0.0
    FrontierE_simulation['z_cmb'] = 1089.0

    FrontierE_simulation['step_list_max'] = [ 604, 584, 566, 548, 531, 515, 499, 484, 470, 456, 442,
                                  429, 417, 405, 393, 381, 370, 359, 349, 339, 329, 319, 310, 
                                  300, 291, 283, 274, 266, 257, 249, 242, 234, 226, 219, 212,
                                  205, 198, 191, 184, 178, 171, 165, 159, 153, 147, 141, 136, 
                                  130, 125,  119,  114,  109,  104, 101]

    FrontierE_simulation['step_list_min'] = [624, 604, 584, 566, 548, 531, 515, 499, 484, 470, 456, 442,
                                  429, 417, 405, 393, 381, 370, 359, 349, 339, 329, 319, 310, 
                                  300, 291, 283, 274, 266, 257, 249, 242, 234, 226, 219, 212,
                                  205, 198, 191, 184, 178, 171, 165, 159, 153, 147, 141, 136, 
                                  130, 125,  119,  114,  109,  104]

    # changeable parameters 
    FrontierE_simulation['nside'] = 16384 # note that reducing nside will cause a downgrade in the map, and will likely induce aliasing. 
    FrontierE_simulation['n_steps_cmb'] = 20 
    FrontierE_simulation['pixwin_datapath']='/pscratch/sd/p/plarsen/ray_tracing_tests/data/'

    
    FrontierE_simulation['output_path_cls'] = '/pscratch/sd/p/plarsen/ray_tracing_tests/FrontierE'
    FrontierE_simulation['output_path_rt'] = '/pscratch/sd/p/plarsen/ray_tracing_tests/FrontierE/map_16384_2p5_wiener/'


    FrontierE_simulation['cosmo'] = FlatLambdaCDM(H0 = 67.66, Om0 = FrontierE_simulation["Omega_m"], Ob0 = 0.04897468161869667, Tcmb0 = 0, Neff = 0) 
    FrontierE_simulation['cosmo_ccl'] = ccl.Cosmology(Omega_c=0.26067, Omega_b=0.04897468161869667, h=0.6766, n_s=0.9665, sigma8=0.8102,  transfer_function='boltzmann_camb',
matter_power_spectrum='halofit')


    # derived parameters
    FrontierE_simulation['nplanes'] =  len(FrontierE_simulation['step_list_max'])
    FrontierE_simulation['chi_cmb'] = ccl.background.comoving_radial_distance(FrontierE_simulation['cosmo_ccl'],a=1./(FrontierE_simulation['z_cmb']+1))*FrontierE_simulation['h']
    FrontierE_simulation['sim_z_max'] = step2z(FrontierE_simulation['step_list_max'][-1]-1, FrontierE_simulation['zfin'], FrontierE_simulation['zinit'], FrontierE_simulation['nsteps'])
    FrontierE_simulation['sim_chi_max'] = ccl.background.comoving_radial_distance(FrontierE_simulation['cosmo_ccl'],a=1./(FrontierE_simulation['sim_z_max']+1))*FrontierE_simulation['h'] 


    # path to files 
    FrontierE_simulation['path_maps'] = '/pscratch/sd/p/plarsen/mass_sheets_FrontierE/'

    # maps for Frontier-E are CIC-weighted density map in Msun/h/steradian on HEALPix map of nside 16384 in NESTED order
    FrontierE_simulation['has_maps'] = True


    # additional testing values 
    FrontierE_simulation['nperst'] = None
    FrontierE_simulation['nperst_steps'] = None

    FrontierE_simulation['path_alms'] = None
    FrontierE_simulation['has_alms'] = False
    
    return FrontierE_simulation
    

def FrontierE_simulation_hydro():

    FrontierE_simulation_hydro = FrontierE_simulation()
    FrontierE_simulation_hydro['name'] = "Frontier-E (hydro)"
    FrontierE_simulation_hydro['path_maps'] =  '/pscratch/sd/p/plarsen/mass_sheets_FrontierE_hydro/'
    FrontierE_simulation_hydro['fb'] = 0.1582

    return FrontierE_simulation_hydro
    


def test_simulation():
    test_simulation = {}
    test_simulation['name'] = "point source test"
    # we need some cosmology and distances to run the test over 
    test_simulation['mpp'] = 2717396224.0
    test_simulation["Omega_m"] = 0.30964
    test_simulation['h'] = 0.6766
    test_simulation['rho_m0'] = test_simulation["Omega_m"] * rho_crit0
    test_simulation['nsteps']= 500
    test_simulation['zinit']= 200.0
    test_simulation['zfin'] = 0.0
    test_simulation['z_source'] = 0.2 # we're going to assume a source plane at z=0.2
    test_simulation['pixwin_datapath']=None

    
    test_simulation['step_list_max'] = [484, 468, 453, 439, 426]
    test_simulation['step_list_min'] = [500, 484, 468, 453, 439]

    # changeable parameters 
    test_simulation['nside'] = 1024 
    test_simulation['output_path_cls'] = '/pscratch/sd/p/plarsen/ray_tracing_tests/'
    test_simulation['output_path_rt'] = '/pscratch/sd/p/plarsen/ray_tracing_tests/test_outputs/'

    test_simulation['cosmo'] = FlatLambdaCDM(H0 = 67.66, Om0 = 0.30964, Ob0 = 0.04897468161869667, Tcmb0 = 0, Neff = 0) 
    test_simulation['cosmo_ccl'] = ccl.Cosmology(Omega_c=0.262, Omega_b=0.0488, h=0.6766, n_s=0.965, sigma8=0.8102, transfer_function='boltzmann_camb', matter_power_spectrum='halofit')

    test_simulation['chi_source'] = ccl.background.comoving_radial_distance(test_simulation['cosmo_ccl'],a=1./(test_simulation['z_source']+1))*test_simulation['h']


    # derived parameters
    test_simulation['nplanes'] =  len(test_simulation['step_list_max'])
    return test_simulation




