#!/bin/bash
#SBATCH -A cusp
#SBATCH -C cpu
#SBATCH -q debug
##SBATCH -q regular
#SBATCH -t 0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128

export NUMEXPR_MAX_THREADS=128
export OMP_NUM_THREADS=128

module load cray-mpich/9.0.1
module load conda 
cd /global/u2/p/plarsen/codes/cmblensing/hacc_raytracing
conda activate ray_trace

srun -n 1 --cpu-bind=cores python precompute_alm.py

#srun -n 1 python born.py
#srun -n 2 --cpu-bind=cores python main_mpi.py 0
#srun -n 2 --cpu-bind=none python main_mpi.py 0
#srun -n 1 python run_raytracing_nersc_pt_xyz.py 0
