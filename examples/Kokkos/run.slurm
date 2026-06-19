#!/bin/bash
#SBATCH --job-name=kokkos-matmul
# TODO: Charge the job to your allocation/account (required on many clusters).
# #SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# Requesting a GPU only matters for the cuda/hip backends. It is harmless to
# leave requested for serial/openmp runs, but you may comment it out to avoid
# queueing for a GPU node when running on the CPU.
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Backend to build: serial | openmp | pthreads | cuda | hip
# Override on submission, e.g.:  sbatch run.slurm openmp
BACKEND="${1:-openmp}"

# TODO: Uncomment and adjust for your cluster's module system.
# A CPU backend needs a recent compiler + CMake; cuda needs the CUDA toolkit.
# module load gcc/12 cmake
# module load cuda/12.0          # cuda backend only

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd $SLURM_SUBMIT_DIR/Example1

# build.sh clones and builds Kokkos for the chosen backend (several minutes
# the first time), then compiles matmul.cpp against it.
./build.sh -t "${BACKEND}"

cd "build_${BACKEND}"
./matmul
