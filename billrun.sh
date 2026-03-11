#!/bin/bash

#SBATCH -J name
#SBATCH --ntasks=384
#SBATCH --nodes=8
#SBATCH --time=24:00:00
#SBATCH --partition=xeon-p8 #(xeon-p8: 48 cores/nodes)

module purge
module load intel-oneapi/2023.1
ulimit -s unlimited

mpirun -np ${SLURM_NPROCS} /home/gridsan/groups/byildiz/vasp.6.4.2/bin/vasp_std