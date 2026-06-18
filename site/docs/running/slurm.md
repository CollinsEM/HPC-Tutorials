# Running on a SLURM Cluster

Practical guide for submitting HPC jobs. Covers SLURM fundamentals and how to use the provided job scripts.

## SLURM overview

SLURM (Simple Linux Utility for Resource Management) is the job scheduler used by most HPC clusters. You describe your resource requirements in a job script and submit it; SLURM queues the job and runs it when resources are available.

Key commands:

| Command | Purpose |
|---------|---------|
| `sbatch job.slurm` | Submit a job script |
| `squeue -u $USER` | List your running/pending jobs |
| `squeue -j JOBID` | Check status of a specific job |
| `scancel JOBID` | Cancel a job |
| `sacct -j JOBID` | Show completed job stats |
| `sinfo` | Show available partitions and node state |

## Anatomy of a SLURM script

```bash
#!/bin/bash
#SBATCH --job-name=my_job         # Name shown in squeue
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks=1                # Number of MPI tasks (1 for non-MPI)
#SBATCH --cpus-per-task=8         # CPU cores per task (= OMP_NUM_THREADS)
#SBATCH --gres=gpu:1              # Request 1 GPU (omit for CPU-only jobs)
#SBATCH --time=00:30:00           # Wall-clock limit (HH:MM:SS)
#SBATCH --output=slurm-%j.out     # stdout log (%j = job ID)
#SBATCH --error=slurm-%j.err      # stderr log

# Load compiler and library modules (names are site-specific)
module load gcc/12
module load cuda/12.0             # GPU jobs only

# Build and run
cd $SLURM_SUBMIT_DIR
make
./my_program
```

!!! warning "Module names are site-specific"
    The `module load` commands vary by cluster. Use `module avail` to list what is installed. Common names: `gcc/12.3`, `cuda/12.2`, `openmpi/4.1`, `nvhpc/24.1`.

## Provided job scripts

Each example includes a `run.slurm` script. Before submitting, open the script and update:

1. `--time` — increase if your run will take longer
2. `--partition` — set to your cluster's partition name (use `sinfo` to list them)
3. `module load` lines — match your cluster's module names

### Auto-vectorization

```bash
cd examples/autovec
sbatch run.slurm
```

### OpenMP

```bash
cd examples/OpenMP
# Adjust --cpus-per-task in run.slurm to set thread count
sbatch run.slurm
```

### CUDA

```bash
cd examples/CUDA/Example1
# Requires --gres=gpu:1 (already set in run.slurm)
sbatch run.slurm
```

### std::par

```bash
cd examples/stdpar/triad
# GPU build requires nvc++ (NVIDIA HPC SDK)
# CPU build uses GCC
sbatch run.slurm
```

## Reading output files

After a job completes:

```bash
cat slurm-<JOBID>.out    # stdout: your program's output
cat slurm-<JOBID>.err    # stderr: compiler warnings, runtime errors
sacct -j <JOBID> --format=JobID,Elapsed,CPUTime,MaxRSS
```

## Debugging failed jobs

If the job fails immediately, check `.err` for missing modules or build errors. If it times out, increase `--time`. If it reports "bus error" on a GPU job, the node may not have a GPU — verify with `squeue -j JOBID -o "%R"` to see the reason.
