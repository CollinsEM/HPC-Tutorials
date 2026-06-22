# Running on a Workstation

Practical reference for setting up a development environment and building each example on a local Linux workstation.

## Prerequisites

| Module | Tool | Minimum version | Install command (Ubuntu/Debian) |
|--------|------|----------------|---------------------------------|
| All | GCC | 9 | `apt install build-essential` |
| All | CMake | 3.12 | `apt install cmake` |
| All | Git | any | `apt install git` |
| OpenMP | GCC with libgomp | 9 | included with GCC |
| CUDA | CUDA Toolkit | 11.0 | from developer.nvidia.com |
| CUDA | nvidia-smi | — | included with NVIDIA driver |
| stdpar | NVIDIA HPC SDK (nvc++) | 22.x | from developer.nvidia.com/hpc-sdk |
| Profiling | LIKWID | 5.x | build from https://github.com/RRZE-HPC/likwid |
| Profiling | Valgrind | 3.x | `apt install valgrind` |

!!! note "No administrator access?"
    The `apt install` commands above assume a Debian/Ubuntu machine on which
    you have `sudo`. On a shared cluster you typically will **not** — instead,
    load pre-built tools with environment modules (`module load gcc cuda
    openmpi`; see [Running on a SLURM Cluster](slurm.md)) or build into your home
    directory. None of the examples in this tutorial require root to *build* or
    *run*; only LIKWID's hardware counters need a one-time privileged setup
    (covered below).

## Verify GPU access

```bash
nvidia-smi        # check driver and GPU model
nvcc --version    # check CUDA toolkit
```

If `nvidia-smi` fails, the NVIDIA driver is not installed. If `nvcc` is missing, install the CUDA Toolkit separately.

## Cloning the repository

```bash
git clone https://github.com/CollinsEM/HPC-Tutorials.git
cd HPC-Tutorials
```

All example paths below are relative to the repository root.

## Building each example

### Auto-vectorization (examples/autovec/)

```bash
cd examples/autovec
mkdir build && cd build
cmake ..
make
./stream_triad
```

To enable vectorization (GCC): edit `CMakeLists.txt`, uncomment the GCC vectorization flags (`-fstrict-aliasing -ftree-vectorize -march=native -fopt-info-vec-optimized`), then rebuild.

### OpenMP (examples/OpenMP/)

Example 1 (Hello):

```bash
cd examples/OpenMP/Example1
mkdir build && cd build
cmake ..
make
export OMP_NUM_THREADS=4
./hello_omp
```

Example 2 (vecadd):

```bash
cd ../../Example2
mkdir build && cd build
cmake ..
make
export OMP_NUM_THREADS=4
./vecadd
```

Example 3 (HealthData):

```bash
cd ../../Example3
mkdir build && cd build
cmake ..
make
export OMP_NUM_THREADS=8
./health_data
```

!!! tip "Controlling thread count"
    `OMP_NUM_THREADS` sets the default team size. You can also call `omp_set_num_threads(N)` programmatically. On NUMA systems, set `OMP_PROC_BIND=close` to keep threads on one socket.

### CUDA (examples/CUDA/Example1/)

```bash
cd examples/CUDA/Example1
make       # auto-detects GPU compute capability via nvidia-smi
./stream_triad
```

The Makefile calls `nvidia-smi --query-gpu=compute_cap --format=csv,noheader` to set the `-arch=sm_XX` flag automatically.

### C++17 std::par (examples/stdpar/triad/)

```bash
cd examples/stdpar/triad

# CPU build (GCC + libTBB):
make -f Makefile.openmp
./triad

# GPU build (requires nvc++):
make
./triad
```

## Profiling with LIKWID

```bash
# List available performance groups
likwid-perfctr -a

# Inspect system topology
likwid-topology

# Run stream_triad under memory bandwidth counter
likwid-perfctr -C 0 -g MEM1 ./stream_triad
```

Build LIKWID from source following the instructions at https://github.com/RRZE-HPC/likwid. Its hardware-counter access (the `MEM` groups above) requires a one-time privileged setup — either a permissive `perf_event_paranoid` setting or LIKWID's setuid access daemon — which an administrator configures once. On HPC clusters this is normally already done; load it with `module load likwid`. Where counters are unavailable, you can still derive achieved bandwidth from the runtime the benchmark prints (see Module 2).

## Profiling with Valgrind

```bash
# Memory error detection
valgrind ./your_program

# Call graph profiling (when kcachegrind is not available)
valgrind --tool=callgrind --callgrind-out-file=callgrind.out ./your_program
callgrind_annotate callgrind.out
```
