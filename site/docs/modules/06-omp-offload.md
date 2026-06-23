# Module 6: OpenMP Offload — GPU Programming with Directives

## Learning Objectives

By the end of this module you will be able to:

- Explain how OpenMP `target` directives offload computation to a GPU
- Map data between host and device with `map(to:)`, `map(from:)`, and `map(tofrom:)`
- Use the `target teams distribute parallel for` construct and explain how it maps to GPU hardware
- Query for available devices at runtime and fall back to the host gracefully
- Compare the directive-based offload model to explicit CUDA and to `std::par`

---

## Three Ways to Run the Triad on a GPU

You have now seen the Stream Triad on a GPU two ways:

- **[CUDA](04-cuda.md)** — explicit kernel, explicit `cudaMalloc`/`cudaMemcpy`. Maximum control, NVIDIA only.
- **[`std::par`](05-stdpar.md)** — a standard-library algorithm with an execution policy. Minimal code, relies on the compiler.

OpenMP offload sits between them. You keep writing ordinary loops, but you annotate them with `#pragma omp target` directives that tell the compiler "run this region on the device, and move this data there and back." It is portable across vendors (NVIDIA, AMD, Intel) and compilers (GCC, Clang, NVHPC) — the same directives, different backend flags.

| Approach | Who writes the kernel | Who moves the data | Portability |
|----------|----------------------|--------------------|-------------|
| CUDA | You (explicit `__global__`) | You (explicit `cudaMemcpy`) | NVIDIA only |
| OpenMP offload | Compiler (from your loop) | You (declarative `map` clauses) | Multi-vendor |
| `std::par` | Compiler | Compiler (unified memory) | Compiler-dependent |

---

## The Offload Execution Model

A GPU is organized as a hierarchy: many *teams* (CUDA calls these *thread blocks*), each containing many *threads*. OpenMP exposes this hierarchy through three composable constructs:

```cpp
#pragma omp target teams distribute parallel for
for (int i = 0; i < n; i++) {
  c[i] = a[i] + b[i];
}
```

Reading the directive left to right:

- **`target`** — execute this region on the default device (the GPU). This is the boundary where host execution hands off to the device.
- **`teams`** — launch a *league* of teams. Each team maps to a GPU thread block / streaming multiprocessor.
- **`distribute`** — partition the loop iterations *across* the teams.
- **`parallel for`** — within each team, run the iterations on multiple threads.

!!! note "Mapping to CUDA terms"
    `teams` ≈ grid of thread blocks; the threads spawned by `parallel for` ≈ threads within a block. The combined `target teams distribute parallel for` is the OpenMP equivalent of a CUDA kernel launched with `<<<gridsize, blocksize>>>` — but you never write the index arithmetic `blockIdx.x*blockDim.x + threadIdx.x`. The compiler generates it.

---

## Mapping Data Between Host and Device

The GPU cannot read host memory directly. Just as CUDA requires `cudaMemcpy`, OpenMP requires you to declare which arrays cross the boundary — but you do it *declaratively* with `map` clauses on a `target data` region:

```cpp
#pragma omp target data map(to: a[0:n], b[0:n]) map(from: c[0:n])
{
  #pragma omp target teams distribute parallel for
  for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
  }
}
```

The `map` clauses specify direction and extent:

| Clause | Meaning | CUDA analogue |
|--------|---------|---------------|
| `map(to: a[0:n])` | Copy `a` host → device before the region | `cudaMemcpy(..., HostToDevice)` |
| `map(from: c[0:n])` | Copy `c` device → host after the region | `cudaMemcpy(..., DeviceToHost)` |
| `map(tofrom: x[0:n])` | Copy both directions (the default) | Both copies |
| `map(alloc: t[0:n])` | Allocate on device, no copy | `cudaMalloc` only |

The `[0:n]` is *array-section* syntax: start index `0`, length `n`. Always specify it for heap or pointer data — the compiler cannot infer the length of a `double *`.

!!! warning "Choose the map direction deliberately"
    Defaulting everything to `tofrom` doubles your PCIe traffic. The triad only *reads* `a` and `b` and only *writes* `c`, so `map(to: a, b)` and `map(from: c)` move each array across the bus exactly once. Getting this wrong is the most common performance bug in offload code — and, as [Module 4](04-cuda.md) showed, data transfer usually dominates the runtime of a memory-bound kernel.

---

## Source Walkthrough

The complete example is `examples/OMP_Offload/Example1/stream_triad.cpp`. The interesting parts:

### Device discovery and fallback

```cpp
int num_devices = omp_get_num_devices();
printf("Number of available devices: %d\n", num_devices);

if (num_devices < 1) {
  printf("No GPU devices available, falling back to CPU execution\n");
} else {
  printf("Using GPU device 0\n");
  omp_set_default_device(0);
}
```

`omp_get_num_devices()` returns the number of attached accelerators. A robust offload program checks this — if no device is found, `target` regions still execute correctly *on the host*, so the same binary runs on a workstation without a GPU.

### The offloaded kernel

```cpp
void vector_add(double *c, double *a, double *b, int n) {
  #pragma omp target data map(to: a[0:n], b[0:n]) map(from: c[0:n])
  {
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < n; i++) {
      c[i] = a[i] + b[i];
    }
  }
}
```

The `target data` region establishes the device data environment; the inner `target` region runs the loop on the device using data already mapped. Separating the two means that if you had *multiple* kernels operating on the same arrays, you would map once and launch several times — exactly the optimization the discussion questions below ask about.

!!! note "This example computes `a + b`, not the full triad"
    The loop body here is `c[i] = a[i] + b[i]` (a vector add), not `c[i] = a[i] + scalar*b[i]`. The verification checks for `3.0` (= `1.0 + 2.0`). The assignment below asks you to restore the scalar multiply so it matches the triad used in the other modules.

---

## Build Flags

GPU offload requires both an OpenMP-capable compiler *and* an offload backend for your GPU vendor. The flags differ by compiler:

| Compiler | CPU multithread | GPU offload (NVIDIA) |
|----------|-----------------|----------------------|
| GCC | `-fopenmp` | `-fopenmp -foffload=nvptx-none` |
| Clang | `-fopenmp` | `-fopenmp -fopenmp-targets=nvptx64` |
| NVHPC (`nvc++`) | `-mp` | `-mp=gpu` |

!!! warning "GCC offload is a separate package"
    Stock GCC is usually built *without* the NVPTX offload plugin. On Debian/Ubuntu you need `gcc-offload-nvptx`. If the `-foffload=nvptx-none` build links but the program reports zero devices at runtime, the offload plugin is missing — install it and rebuild. The NVIDIA HPC SDK (`nvc++ -mp=gpu`) is often the smoother path for NVIDIA hardware.

---

## Assignment

**Task 1 — Build and run.** Build the example and confirm it reports a device. Note the runtime.

=== "Workstation"
    The NVIDIA HPC SDK (`nvc++`) is the most reliable path on NVIDIA hardware.
    If you have it available (e.g. via `module load nvhpc`):
    ```bash
    cd examples/OMP_Offload/Example1
    mkdir build && cd build
    CXX=nvc++ cmake ..
    make
    ./stream_triad
    ```
    With GCC, offload to NVIDIA requires the `gcc-offload-nvptx` plugin
    (often a separate package). If it is installed:
    ```bash
    g++ -fopenmp -foffload=nvptx-none -O3 stream_triad.cpp timer.cpp -o stream_triad
    ```
    If the binary reports zero devices, the plugin is missing or not configured.

=== "SLURM cluster"
    See `examples/OMP_Offload/Example1/run.slurm`. It requests one GPU with `--gres=gpu:1` and loads a compiler module with offload support. Adjust the module name for your cluster.

**Task 2 — Restore the triad.** Change the loop body to the full Stream Triad, `c[i] = a[i] + scalar*b[i]` with `scalar = 3.0`, and update the verification (the expected value becomes `1.0 + 3.0*2.0 = 7.0`). Confirm verification passes.

??? success "Show solution"
    ```cpp
    void vector_add(double *c, double *a, double *b, int n) {
      const double scalar = 3.0;
      #pragma omp target data map(to: a[0:n], b[0:n]) map(from: c[0:n])
      {
        #pragma omp target teams distribute parallel for
        for (int i = 0; i < n; i++) {
          c[i] = a[i] + scalar*b[i];
        }
      }
    }
    ```
    And in `main`, the verification target changes from `3.0` to `7.0`:
    ```cpp
    if (c[i] != 7.0) { errors++; /* ... */ }
    ```

**Task 3 — Measure the map cost.** Temporarily change `map(to: a[0:n], b[0:n]) map(from: c[0:n])` to `map(tofrom: a[0:n], b[0:n], c[0:n])` and compare the runtime. You are now copying all three arrays both directions. Quantify the slowdown.

**Task 4 — Compare the three GPU models.** You now have runtimes for the same 80M-element triad from CUDA ([Module 4](04-cuda.md)), `std::par` ([Module 5](05-stdpar.md)), and OpenMP offload. Tabulate kernel time and total time for each. Which gives the best performance? Which required the least code?

---

## Expected Output

```
Number of available devices: 1
Using GPU device 0
Running with 8 host thread(s)
Verification passed for first 100 elements
Runtime is 52.3 msecs
```

The reported runtime includes the data mapping (host↔device transfer), which — as in the CUDA example — dominates for this memory-bound kernel. If the program prints `Number of available devices: 0`, it ran on the host; check that your offload plugin is installed (see the warning above).

---

## Discussion Questions

1. The example wraps the loop in a `target data` region even though there is only one kernel. What is the benefit of `target data` when a function launches *several* kernels over the same arrays? How does this relate to the "data transfer dominates" observation from Module 4?

2. `target teams distribute parallel for` has four composable parts. What happens to performance if you drop `teams distribute` and write only `target parallel for`? (Hint: how many of the GPU's streaming multiprocessors get used?)

3. OpenMP offload, CUDA, and `std::par` all expressed the same computation. Rank them by (a) lines of code changed from the serial version, (b) control over data movement, and (c) portability across GPU vendors. When would you reach for each?

4. The same OpenMP-offload binary runs on a machine with no GPU by executing `target` regions on the host. What does this portability cost you, and when is it worth it?

---

## References

### Reference materials

- [OpenMP specifications](https://www.openmp.org/specifications/) — the `target` offload directives.
- [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk) — the `nvc++ -mp=gpu` offload compiler.
- [GCC offloading](https://gcc.gnu.org/wiki/Offloading) — building and using GCC's NVPTX offload support.
- [LLVM/Clang OpenMP](https://openmp.llvm.org/) — Clang's OpenMP and offload runtime.
