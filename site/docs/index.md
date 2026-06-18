# Parallel Programming for HPC — A Practical Guide

This tutorial covers the essential skills for writing parallel programs that run
efficiently on modern CPUs and GPUs, and for deploying them on SLURM-managed
HPC clusters.

## Who this is for

Software developers and scientists who write compute-intensive code and want to
exploit the full parallelism available in modern hardware — SIMD vector units,
multi-core CPUs, and GPU accelerators.

## What you will learn

| Module | Topic | Key skill |
|--------|-------|-----------|
| [0](modules/00-parallel-theory.md) | Parallel computing theory | Amdahl's Law, Flynn's taxonomy, scaling limits |
| [1](modules/01-hardware.md) | Hardware & data layout | Memory hierarchy, cache-friendly code, DOP vs OOP |
| [2](modules/02-autovec.md) | Auto-vectorization | SIMD, compiler flags, measuring bandwidth |
| [3](modules/03-openmp.md) | OpenMP | Thread-level CPU parallelism with `#pragma omp` |
| [4](modules/04-cuda.md) | CUDA | GPU kernels, device memory, data transfers |
| [5](modules/05-stdpar.md) | C++17 `std::par` | GPU-accelerated standard algorithms |

## How to use this tutorial

Each module can be read on its own or followed in sequence. Modules 2–5 build
on the hardware intuition developed in Modules 0–1, so first-time readers are
encouraged to start there.

Each coding module contains:

- **Concept section** — the key ideas, grounded in real hardware behavior
- **Code walkthrough** — annotated source showing the critical patterns
- **Assignment** — starter code with `TODO` comments; a hidden solution you can reveal
- **Build & run instructions** — both for a local workstation and for a SLURM cluster

## Common benchmark: Stream Triad

All coding modules use a single kernel as their benchmark:

$$A[i] = B[i] + \alpha \times C[i]$$

This *Stream Triad* operation is memory-bound, meaning its performance is limited
by memory bandwidth rather than floating-point throughput — which is the common
case for most HPC codes. Implementing it across multiple programming models
lets you compare apples-to-apples as you move from serial code to OpenMP to CUDA.

## Getting the code

```bash
git clone https://github.com/CollinsEM/HPC-Tutorials.git
cd HPC-Tutorials/examples
```

Each subdirectory contains a self-contained example with a `README` or inline
build instructions. See [Running on a Workstation](running/workstation.md) and
[Running on a SLURM Cluster](running/slurm.md) for environment setup.
