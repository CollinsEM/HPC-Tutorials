#!/bin/bash
#SBATCH --job-name=cuda-stream-triad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# TODO: Uncomment and adjust for your cluster's module system
# module load cuda/12.0

cd $SLURM_SUBMIT_DIR
make
./stream_triad
