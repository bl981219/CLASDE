#!/bin/bash

# CLASDE VASP Template
# Use 'name' as a placeholder for autonomous job naming
#SBATCH -J name
#SBATCH --ntasks=48
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=xeon-p8

# Environment Setup
module purge
module load intel-oneapi/2023.1
ulimit -s unlimited

# Execution
# SLURM_NTASKS is automatically set by Slurm based on --ntasks
mpirun -np ${SLURM_NTASKS} /home/gridsan/groups/byildiz/vasp.6.4.2/bin/vasp_std
