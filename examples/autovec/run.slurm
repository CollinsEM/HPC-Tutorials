#!/bin/bash
#SBATCH --job-name=stream-triad-autovec
# TODO: Charge the job to your allocation/account (required on many clusters).
# #SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# TODO: Uncomment and adjust for your cluster's module system
# module load gcc/12

cd $SLURM_SUBMIT_DIR

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
cd ..

./build/stream_triad
