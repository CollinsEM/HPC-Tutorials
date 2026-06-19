#!/bin/bash
#SBATCH --job-name=stdpar-triad
# TODO: Charge the job to your allocation/account (required on many clusters).
# #SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# TODO: Uncomment and adjust for your cluster's module system
# module load nvhpc/24.1     # provides nvc++

cd $SLURM_SUBMIT_DIR

# GPU build (requires nvc++ from NVIDIA HPC SDK)
make
./triad

# To run CPU-only build instead:
# make -f Makefile.openmp
# ./triad
