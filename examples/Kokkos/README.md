# Kokkos Examples

These examples demonstrate performance-portable parallel programming with
[Kokkos](https://github.com/kokkos/kokkos).  The build scripts clone Kokkos
from its main branch and build it for the requested backend, so you always get
a recent release.

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

### Example 1 — Matrix multiply

Demonstrates a Kokkos `MDRangePolicy` kernel and how the same source file
produces CPU or GPU binaries by changing one build flag.

```bash
cd Example1
./build.sh -t serial      # single-threaded baseline
./build.sh -t openmp      # multicore CPU
./build.sh -t cuda        # NVIDIA GPU (requires nvcc)
./build.sh -t hip         # AMD GPU (requires hipcc)
```

Each invocation builds Kokkos from source and then compiles the example
against it.  The first build takes several minutes; subsequent builds with the
same `-t` reuse the existing Kokkos install.

See `../../site/docs/modules/08-kokkos.md` for the full tutorial walkthrough.
