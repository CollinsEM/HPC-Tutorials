#!/bin/bash
#SBATCH --job-name=omp-vecadd
# TODO: Charge the job to your allocation/account (required on many clusters).
# #SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# TODO: Uncomment and adjust for your cluster's module system
# module load gcc/12

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=close

cd $SLURM_SUBMIT_DIR

# Build and run Example 2 (vector add)
cd Example2
mkdir -p build && cd build
cmake ..
make -j
cd ..
./build/vecadd

# Build and run Example 3 (health data)
cd ../Example3
mkdir -p build && cd build
cmake ..
make -j
cd ..
./build/health_data
