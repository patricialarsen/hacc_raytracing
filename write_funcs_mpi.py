import os
import shutil
import numpy as np
import healpy as hp
from simulation import get_chi_step

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


def gather_map_to_rank0(local_map, pix_min, pix_max, npix, comm):
    """Gather contiguous local pixel chunks onto rank 0 as a full HEALPix map."""
    rank = comm.Get_rank()

    local_map = np.ascontiguousarray(local_map, dtype=np.float32)
    local_count = local_map.size

    counts = comm.gather(local_count, root=0)
    starts = comm.gather(pix_min, root=0)

    if rank == 0:
        counts = np.asarray(counts, dtype=np.int64)
        starts = np.asarray(starts, dtype=np.int64)
        displs = np.zeros_like(counts)
        displs[1:] = np.cumsum(counts[:-1])

        recvbuf = np.empty(np.sum(counts), dtype=np.float32)
    else:
        counts = None
        starts = None
        displs = None
        recvbuf = None

    comm.Gatherv(local_map, (recvbuf, counts, displs, MPI.FLOAT), root=0,)

    if rank != 0:
        return None

    full_map = np.empty(npix, dtype=np.float32)

    for start, count, disp in zip(starts, counts, displs):
        stop = start + count
        full_map[start:stop] = recvbuf[disp:disp + count]

    return full_map
    
def gather_map_to_rank0_64bit(local_map, pix_min, pix_max, npix, comm):
    """Gather contiguous local pixel chunks onto rank 0 as a full HEALPix map."""
    rank = comm.Get_rank()

    local_map = np.ascontiguousarray(local_map, dtype=np.float64)
    local_count = local_map.size

    counts = comm.gather(local_count, root=0)
    starts = comm.gather(pix_min, root=0)

    if rank == 0:
        counts = np.asarray(counts, dtype=np.int64)
        starts = np.asarray(starts, dtype=np.int64)
        displs = np.zeros_like(counts)
        displs[1:] = np.cumsum(counts[:-1])

        recvbuf = np.empty(np.sum(counts), dtype=np.float64)
    else:
        counts = None
        starts = None
        displs = None
        recvbuf = None

    comm.Gatherv(local_map, (recvbuf, counts, displs, MPI.DOUBLE), root=0,)

    if rank != 0:
        return None

    full_map = np.empty(npix, dtype=np.float64)

    for start, count, disp in zip(starts, counts, displs):
        stop = start + count
        full_map[start:stop] = recvbuf[disp:disp + count]

    return full_map


def scatter_map_from_rank0(full_map, pix_min, pix_max, npix, comm, dtype=np.float64):
    """Scatter full HEALPix map on rank 0 into contiguous pixel chunks."""
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_count = pix_max - pix_min
    local_map = np.empty(local_count, dtype=dtype)

    counts = comm.gather(local_count, root=0)
    starts = comm.gather(pix_min, root=0)

    if rank == 0:
        full_map = np.ascontiguousarray(full_map, dtype=dtype)
        counts = np.asarray(counts, dtype=np.int64)
        starts = np.asarray(starts, dtype=np.int64)

        sendbuf = np.empty(np.sum(counts), dtype=dtype)
        displs = np.zeros_like(counts)
        displs[1:] = np.cumsum(counts[:-1])

        for start, count, disp in zip(starts, counts, displs):
            sendbuf[disp:disp + count] = full_map[start:start + count]
    else:
        sendbuf = None
        counts = None
        displs = None

    mpi_type = MPI.DOUBLE if np.dtype(dtype) == np.dtype(np.float64) else MPI.FLOAT

    comm.Scatterv(
        (sendbuf, counts, displs, mpi_type),
        local_map,
        root=0,
    )
    return local_map





def write_checkpoint_dir(step_idx, sim, kappa_map, shear1_map, shear2_map,
                         w_map, theta, phi, pix_min, pix_max, comm, psi=None, 
                         kappa_born_alm=None, add_psi=True, born_CMB=True):

    rank = comm.Get_rank()
    npix = hp.nside2npix(sim["nside"])

    path_out = sim['output_path_rt']
    tmp = path_out + 'checkpoint_tmp'
    current = path_out + 'checkpoint_current'
    previous = path_out + 'checkpoint_prev'
    previous_m1 = path_out + 'checkpoint_prev_prev'

    if rank==0:
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp)
    comm.Barrier()

    # gather all maps onto rank 0 
    kappa_full = gather_map_to_rank0_64bit(kappa_map, pix_min, pix_max, npix, comm)
    shear1_full = gather_map_to_rank0_64bit(shear1_map, pix_min, pix_max, npix, comm)
    shear2_full = gather_map_to_rank0_64bit(shear2_map, pix_min, pix_max, npix, comm)
    w_full = gather_map_to_rank0_64bit(w_map, pix_min, pix_max, npix, comm)
    theta_full = gather_map_to_rank0_64bit(theta, pix_min, pix_max, npix, comm)
    phi_full = gather_map_to_rank0_64bit(phi, pix_min, pix_max, npix, comm)

    if rank==0:
        hp.write_map(tmp + '/kappa.fits', kappa_full, dtype=np.float64, overwrite=True)
        hp.write_map(tmp + '/shear1.fits', shear1_full, dtype=np.float64, overwrite=True)
        hp.write_map(tmp + '/shear2.fits', shear2_full, dtype=np.float64, overwrite=True)
        hp.write_map(tmp + '/w_map.fits', w_full, dtype=np.float64, overwrite=True)
        hp.write_map(tmp + '/theta.fits', theta_full, dtype=np.float64, overwrite=True)
        hp.write_map(tmp + '/phi.fits', phi_full, dtype=np.float64, overwrite=True)

    if add_psi:
        psi_full = gather_map_to_rank0_64bit(psi, pix_min, pix_max, npix, comm)
        if rank==0:
            hp.write_map(tmp + '/psi.fits', psi_full, dtype=np.float64, overwrite=True)

    if born_CMB:
        if rank==0:
            hp.write_alm(tmp + '/kappa_born_CMB_alm.fits', kappa_born_alm, overwrite=True)
    comm.Barrier()

    # write checkpoint metadata and rename folders to update checkpoints iteratively
    if rank==0:
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

    comm.Barrier()
    return 




    
    
def write_outputs(step_idx, sim, pix_min, pix_max, comm, kappa_map=None, shear1_map=None, shear2_map=None, w_map=None, CMB=False, kappa_born=None, test=False, born_only=False):
    """After basis rotation"""

    rank = comm.Get_rank()
    npix = hp.nside2npix(sim["nside"])
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
            
    if kappa_born is not None:
        if rank==0:
            hp.write_map(path_out + 'kappa_born_' + suffix, kappa_born, dtype=np.float32, overwrite=True)
            
    if not born_only:
        kappa_full = gather_map_to_rank0(kappa_map, pix_min, pix_max, npix, comm)
        shear1_full = gather_map_to_rank0(shear1_map, pix_min, pix_max, npix, comm)
        shear2_full = gather_map_to_rank0(shear2_map, pix_min, pix_max, npix, comm)
        w_full = gather_map_to_rank0(w_map, pix_min, pix_max, npix, comm)
        if rank == 0:
            hp.write_map(path_out + 'kappa_' + suffix, kappa_full, dtype=np.float32, overwrite=True)
            hp.write_map(path_out + 'shear1_' + suffix, shear1_full, dtype=np.float32, overwrite=True)
            hp.write_map(path_out + 'shear2_' + suffix, shear2_full, dtype=np.float32, overwrite=True)
            hp.write_map(path_out + 'w_map_' + suffix, w_full, dtype=np.float32, overwrite=True)

    return 


def write_native_state( step_idx, sim, A_11, A_12, A_21, A_22, theta, phi, psi, pix_min, pix_max, comm):
    
    rank = comm.Get_rank()
    npix = hp.nside2npix(sim["nside"])

    path_out = sim['output_path_rt']
    if step_idx>sim['nplanes']-1:
        suffix = "tail_plane_" + str(step_idx - sim['nplanes']) + '_source_basis.fits'
    else:
        step_high = sim['step_list_max'][step_idx]
        step_low = sim['step_list_min'][step_idx]
        suffix = str(step_high)+'_'+str(step_low)+'_source_basis.fits'

    A_11_full = gather_map_to_rank0_64bit(A_11, pix_min, pix_max, npix, comm)
    A_12_full = gather_map_to_rank0_64bit(A_12, pix_min, pix_max, npix, comm)
    A_21_full = gather_map_to_rank0_64bit(A_21, pix_min, pix_max, npix, comm)
    A_22_full = gather_map_to_rank0_64bit(A_22, pix_min, pix_max, npix, comm)
    theta_full = gather_map_to_rank0_64bit(theta, pix_min, pix_max, npix, comm)
    phi_full = gather_map_to_rank0_64bit(phi, pix_min, pix_max, npix, comm)
    psi_full = gather_map_to_rank0_64bit(psi, pix_min, pix_max, npix, comm)

    if rank==0:
        hp.write_map(path_out + 'dA_11_' + suffix, A_11_full-1.0, dtype=np.float32, overwrite=True)
        hp.write_map(path_out + 'A_12_' + suffix, A_12_full, dtype=np.float32, overwrite=True)
        hp.write_map(path_out + 'A_21_' + suffix, A_21_full, dtype=np.float32, overwrite=True)
        hp.write_map(path_out + 'dA_22_' + suffix, A_22_full-1.0, dtype=np.float32, overwrite=True)
        hp.write_map(path_out + 'theta_' + suffix, theta_full, dtype=np.float32, overwrite=True)
        hp.write_map(path_out + 'phi_' + suffix, phi_full, dtype=np.float32, overwrite=True)
        hp.write_map(path_out + 'psi_' + suffix, psi_full, dtype=np.float32, overwrite=True)

    return 



def read_checkpoint_meta(checkpoint_dir):
    with open(checkpoint_dir + '/checkpoint_meta.txt', 'r') as f:
        lines = [x.strip() for x in f.readlines()]
    return int(lines[0]) 



def read_checkpoint_state_chunked(checkpoint_dir, pix_min, pix_max, npix, comm, add_psi=True, born_CMB=True,):
    rank = comm.Get_rank()

    if rank == 0:
        kappa = hp.read_map(checkpoint_dir + '/kappa.fits').astype(np.float64)
        shear1 = hp.read_map(checkpoint_dir + '/shear1.fits').astype(np.float64)
        shear2 = hp.read_map(checkpoint_dir + '/shear2.fits').astype(np.float64)
        w_map = hp.read_map(checkpoint_dir + '/w_map.fits').astype(np.float64)
        theta = hp.read_map(checkpoint_dir + '/theta.fits').astype(np.float64)
        phi = hp.read_map(checkpoint_dir + '/phi.fits').astype(np.float64)

        A_11 = 1.0 - kappa - shear1
        A_22 = 1.0 - kappa + shear1
        A_12 = shear2 + w_map
        A_21 = shear2 - w_map

        if add_psi:
            psi = hp.read_map(checkpoint_dir + '/psi.fits').astype(np.float64)
        else:
            psi = None

        if born_CMB:
            kappa_born_alm = hp.read_alm(checkpoint_dir + '/kappa_born_CMB_alm.fits')
        else:
            kappa_born_alm = None
    else:
        theta = phi = A_11 = A_22 = A_12 = A_21 = psi = None
        kappa_born_alm = None

    theta = scatter_map_from_rank0(theta, pix_min, pix_max, npix, comm)
    phi = scatter_map_from_rank0(phi, pix_min, pix_max, npix, comm)
    A_11 = scatter_map_from_rank0(A_11, pix_min, pix_max, npix, comm)
    A_22 = scatter_map_from_rank0(A_22, pix_min, pix_max, npix, comm)
    A_12 = scatter_map_from_rank0(A_12, pix_min, pix_max, npix, comm)
    A_21 = scatter_map_from_rank0(A_21, pix_min, pix_max, npix, comm)

    if add_psi:
        psi = scatter_map_from_rank0(psi, pix_min, pix_max, npix, comm)
    else:
        psi = np.zeros(pix_max - pix_min, dtype=np.float64)

    if born_CMB:
        kappa_born_alm = comm.bcast(kappa_born_alm, root=0)

    return theta, phi, A_11, A_22, A_12, A_21, psi, kappa_born_alm
    

def restart_from_checkpoint(sim, pix_min, pix_max, comm, add_psi=True):

    rank = comm.Get_rank()
    npix = hp.nside2npix(sim["nside"])

    path_out = sim['output_path_rt']
    current_dir = path_out + 'checkpoint_current'
    prev_dir = path_out + 'checkpoint_prev'

    if rank == 0:
        current_step = read_checkpoint_meta(current_dir)
        prev_step = read_checkpoint_meta(prev_dir)
        if prev_step != current_step - 1:
            raise RuntimeError(f"checkpoint mismatch: current={current_step}, previous={prev_step}")
    else:
        current_step = None
        prev_step = None 

    current_step = comm.bcast(current_step, root=0)
    prev_step = comm.bcast(prev_step, root=0)

    
    theta, phi, A_11, A_22, A_12, A_21, psi, kappa_born_alm = read_checkpoint_state_chunked( current_dir, pix_min, pix_max, npix,  comm, add_psi=add_psi, born_CMB=True, )
    theta_m1, phi_m1, A_11_m1, A_22_m1, A_12_m1, A_21_m1, psi_m1, _ = read_checkpoint_state_chunked( prev_dir, pix_min, pix_max, npix,  comm, add_psi=add_psi, born_CMB=False, )

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




