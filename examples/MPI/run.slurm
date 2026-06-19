#!/bin/bash
#SBATCH --job-name=mpi-vecmat
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# TODO: Uncomment and adjust for your cluster's module system.
# module load gcc/12
# module load openmpi/4.1     # or intel-oneapi/mpi, mpich, etc.

cd $SLURM_SUBMIT_DIR

# Build and run Example 3 (distributed matrix-vector multiply).
# --ntasks above sets the number of MPI ranks; srun places one rank
# per task across whatever nodes SLURM allocated.
cd Example3
mpicxx -O3 vector_matrix_mpi.cpp -o vmm
srun ./vmm
