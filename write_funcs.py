import os
import shutil
import numpy as np
import healpy as hp
from simulation import get_chi_step

def write_checkpoint_dir(step_idx, sim, kappa_map, shear1_map, shear2_map,
                         w_map, theta, phi, psi=None, kappa_born_alm=None,
                         add_psi=True, born_CMB=True):

    path_out = sim['output_path_rt']
    tmp = path_out + 'checkpoint_tmp'
    current = path_out + 'checkpoint_current'
    previous = path_out + 'checkpoint_prev'
    previous_m1 = path_out + 'checkpoint_prev_prev'

    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)

    hp.write_map(tmp + '/kappa.fits', kappa_map, dtype=np.float64, overwrite=True)
    hp.write_map(tmp + '/shear1.fits', shear1_map, dtype=np.float64, overwrite=True)
    hp.write_map(tmp + '/shear2.fits', shear2_map, dtype=np.float64, overwrite=True)
    hp.write_map(tmp + '/w_map.fits', w_map, dtype=np.float64, overwrite=True)
    hp.write_map(tmp + '/theta.fits', theta, dtype=np.float64, overwrite=True)
    hp.write_map(tmp + '/phi.fits', phi, dtype=np.float64, overwrite=True)

    if add_psi:
        hp.write_map(tmp + '/psi.fits', psi, dtype=np.float64, overwrite=True)

    if born_CMB:
        hp.write_alm(tmp + '/kappa_born_CMB_alm.fits', kappa_born_alm, overwrite=True)

    if step_idx>sim['nplanes']-1:
        with open(tmp + '/checkpoint_meta.txt', 'w') as f:
            f.write(str(step_idx) + '\n')
    else:
        step_high = sim['step_list_max'][step_idx]
        step_low = sim['step_list_min'][step_idx]
        with open(tmp + '/checkpoint_meta.txt', 'w') as f:
            f.write(str(step_idx) + '\n')
            f.write(f'{step_high}_{step_low}\n')

    if os.path.exists(previous_m1):
        shutil.rmtree(previous_m1)
    if os.path.exists(previous):
        os.rename(previous, previous_m1)
    if os.path.exists(current):
        os.rename(current, previous)

    os.rename(tmp, current)

def write_outputs(step_idx, sim, kappa_map=None, shear1_map=None, shear2_map=None, w_map=None, CMB=False, kappa_born=None, test=False, born_only=False):
    """After basis rotation"""
    path_out = sim['output_path_rt']
    if CMB and kappa_born is None:
        raise ValueError("CMB=True but kappa_born=None")
    if CMB:
        suffix = 'CMB.fits'
    elif test:
        suffix = 'test.fits'
    else:
        if step_idx>sim['nplanes']-1:
            suffix = "tail_plane_" + str(step_idx - sim['nplanes']) + '.fits'
        else:
            step_high = sim['step_list_max'][step_idx]
            step_low = sim['step_list_min'][step_idx]
            suffix = str(step_high)+'_'+str(step_low)+'.fits'

    if not born_only:
        hp.write_map(path_out + 'kappa_' + suffix, kappa_map, dtype=np.float32, overwrite=True)
        hp.write_map(path_out + 'shear1_' + suffix, shear1_map, dtype=np.float32, overwrite=True)
        hp.write_map(path_out + 'shear2_' + suffix, shear2_map, dtype=np.float32, overwrite=True)
        hp.write_map(path_out + 'w_map_' + suffix, w_map, dtype=np.float32, overwrite=True)
    if kappa_born is not None:
        hp.write_map(path_out + 'kappa_born_' + suffix, kappa_born, dtype=np.float32, overwrite=True)

    return 

def write_native_state( step_idx, sim, A_11, A_12, A_21, A_22, theta, phi, psi):
    path_out = sim['output_path_rt']
    if step_idx>sim['nplanes']-1:
        suffix = "tail_plane_" + str(step_idx - sim['nplanes']) + '_source_basis.fits'
    else:
        step_high = sim['step_list_max'][step_idx]
        step_low = sim['step_list_min'][step_idx]
        suffix = str(step_high)+'_'+str(step_low)+'_source_basis.fits'
    hp.write_map(path_out + 'dA_11_' + suffix, A_11-1.0, dtype=np.float32, overwrite=True)
    hp.write_map(path_out + 'A_12_' + suffix, A_12, dtype=np.float32, overwrite=True)
    hp.write_map(path_out + 'A_21_' + suffix, A_21, dtype=np.float32, overwrite=True)
    hp.write_map(path_out + 'dA_22_' + suffix, A_22-1.0, dtype=np.float32, overwrite=True)
    hp.write_map(path_out + 'theta_' + suffix, theta, dtype=np.float32, overwrite=True)
    hp.write_map(path_out + 'phi_' + suffix, phi, dtype=np.float32, overwrite=True)
    hp.write_map(path_out + 'psi_' + suffix, psi, dtype=np.float32, overwrite=True)
    return 



def read_checkpoint_meta(checkpoint_dir):
    with open(checkpoint_dir + '/checkpoint_meta.txt', 'r') as f:
        lines = [x.strip() for x in f.readlines()]
    return int(lines[0]) 


def read_checkpoint_state(checkpoint_dir, add_psi=True, born_CMB=False):
    theta = hp.read_map(checkpoint_dir + '/theta.fits').astype(np.float64)
    phi = hp.read_map(checkpoint_dir + '/phi.fits').astype(np.float64)

    kappa = hp.read_map(checkpoint_dir + '/kappa.fits').astype(np.float64)
    shear1 = hp.read_map(checkpoint_dir + '/shear1.fits').astype(np.float64)
    shear2 = hp.read_map(checkpoint_dir + '/shear2.fits').astype(np.float64)
    w = hp.read_map(checkpoint_dir + '/w_map.fits').astype(np.float64)

    A_11 = 1.0 - kappa - shear1
    A_22 = 1.0 - kappa + shear1
    A_12 = shear2 + w
    A_21 = shear2 - w


    if add_psi:
        psi = hp.read_map(checkpoint_dir + '/psi.fits').astype(np.float64)
    else:
        psi = None

    if born_CMB:
        kappa_born_alm = hp.read_alm(checkpoint_dir + '/kappa_born_CMB_alm.fits')
        return theta, phi, A_11, A_22, A_12, A_21, psi, kappa_born_alm

    return theta, phi, A_11, A_22, A_12, A_21, psi

def restart_from_checkpoint(sim,  add_psi=True):
    path_out = sim['output_path_rt']
    current_dir = path_out + 'checkpoint_current'
    prev_dir = path_out + 'checkpoint_prev'

    current_step = read_checkpoint_meta(current_dir)
    prev_step = read_checkpoint_meta(prev_dir)

    if prev_step != current_step - 1:
        raise RuntimeError(
            f"checkpoint mismatch: current={current_step}, previous={prev_step}"
        )

    theta, phi, A_11, A_22, A_12, A_21, psi, kappa_born_alm = read_checkpoint_state(
        current_dir,
        add_psi=add_psi,
        born_CMB=True,
    )

    theta_m1, phi_m1, A_11_m1, A_22_m1, A_12_m1, A_21_m1, psi_m1 = read_checkpoint_state(
        prev_dir,
        add_psi=add_psi,
        born_CMB=False,
    )

    step_idx = current_step + 1
    _, _, chi_k = get_chi_step(current_step, sim)
    _, _, chi_km1 = get_chi_step(prev_step, sim)

    return (
        step_idx,
        theta, phi, A_11, A_22, A_12, A_21, psi,
        theta_m1, phi_m1, A_11_m1, A_22_m1, A_12_m1, A_21_m1, psi_m1,
        kappa_born_alm,
        chi_km1, chi_k,
    )
