#!/bin/bash
#SBATCH --job-name=omp-offload-triad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# TODO: Uncomment and adjust for your cluster's module system.
# Requires a compiler with GPU offload support, e.g. GCC built with
# the NVPTX plugin (gcc-offload-nvptx) or the NVIDIA HPC SDK (nvhpc).
# module load gcc/12        # must include nvptx offload support
# module load cuda/12.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd $SLURM_SUBMIT_DIR

mkdir -p build && cd build
cmake ..
make -j
cd ..

./build/stream_triad
