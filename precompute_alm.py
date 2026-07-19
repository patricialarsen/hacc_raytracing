#!/bin/bash/python

import os
import sys
os.environ["OMP_NUM_THREADS"] = "128"
import time
from simulation import precompute_input_map_alms
from utils import  timed, timed_rank
from hacc_sims import LJ_simulation, FrontierE_simulation, FrontierE_simulation_hydro


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

sim = FrontierE_simulation_hydro()


nthreads = 128


# constants 
vc = 2.998e5 #km/s
G = 4.3011790220362e-09 # Mpc/h (Msun/h)^-1 (km/s)^2

nside = sim['nside']

lmax = int(2.5*nside)

step_idx = 0
print("Precomputing alms for  ", sim['nplanes'], " planes, starting with step ", step_idx, flush=True)    
while step_idx<24:#sim['nplanes']:
    print('step_idx',step_idx, flush=True)   
    with timed_rank(f"step {step_idx} read/input map",comm):
        alms_filtered = precompute_input_map_alms(step_idx, sim, lmax, nthreads, comm=comm)
    step_idx +=1
