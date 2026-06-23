# MATAR Examples

These examples demonstrate [MATAR](https://github.com/lanl/MATAR), a
higher-level array and loop abstraction built on top of Kokkos.  The build
scripts clone Kokkos from its main branch, so you always get a recent release.

## Prerequisites

| Tool | Minimum version | Notes |
|------|----------------|-------|
| CMake | 3.16 | |
| C++ compiler | GCC ≥ 10, Clang ≥ 10, or MSVC 19.29 | C++20 required by Kokkos 5.x |
| CUDA Toolkit | 11.0+ | `-t cuda` only |
| ROCm / HIP | 5.0+ | `-t hip` only |

C++20 support was introduced in GCC 10 (2020), Clang 10 (2020), and MSVC
2019 v16.11.  Any modern HPC cluster should satisfy this requirement.  If your
system provides only GCC 9 or older, you will need to install a newer compiler
or use a container.

## Examples

### Example 1 — Matrix multiply (with a deliberate data-race bug)

Demonstrates MATAR's `CArrayKokkos` and `FOR_ALL` macro, and shows how a
race condition that is invisible on CPU backends becomes catastrophic on CUDA.

```bash
cd Example1
./build.sh -t serial      # single-threaded baseline (correct output)
./build.sh -t openmp      # multicore CPU (race hidden by tiling — looks correct)
./build.sh -t cuda        # NVIDIA GPU (race visible — values ~4 instead of 4096)
```

A corrected version with the `k` loop kept sequential is in `Example1/solution/`.

See `../../site/docs/modules/08-kokkos.md` for the full tutorial walkthrough,
including a detailed explanation of why the OpenMP backend hides the race and
the CUDA backend exposes it.
