# Module 8: Kokkos & MATAR — Performance Portability

## Learning Objectives

By the end of this module you will be able to:

- Explain the performance-portability problem and how an abstraction layer solves it
- Describe Kokkos *execution spaces*, *memory spaces*, and *Views*
- Write a parallel kernel with `parallel_for`, `KOKKOS_LAMBDA`, and `MDRangePolicy`
- Move data between host and device with mirror views and `deep_copy`
- Build the *same source* for serial, OpenMP, CUDA, or HIP backends by changing one flag
- Recognize how MATAR layers a terser syntax on top of Kokkos — and spot a data race in nested parallelism

---

## The Portability Problem

[Module 0](00-parallel-theory.md) ended on a warning: every GPU vendor has its own native programming model.

| Model | Targets |
|-------|---------|
| CUDA | NVIDIA GPUs only |
| HIP | AMD GPUs (and NVIDIA via a shim) |
| SYCL / oneAPI | Intel GPUs (and others) |
| OpenMP / OpenACC | CPUs and, with offload, GPUs |

A code written in CUDA runs only on NVIDIA hardware. Porting it to an AMD or Intel machine means rewriting every kernel. For a large scientific code with a decade of development behind it, that is untenable.

[**Kokkos**](https://github.com/kokkos/kokkos) solves this by interposing a C++ abstraction layer between your code and the backend. You write the algorithm *once* using Kokkos constructs; at compile time you select a backend (Serial, OpenMP, Threads, CUDA, HIP), and Kokkos generates the appropriate low-level code. It is plain C++ — no special compiler required — so the same `matmul.cpp` in this module compiles to a multicore CPU binary or an NVIDIA GPU binary depending only on how you build Kokkos.

!!! note "A different benchmark: matrix multiply"
    Like the [MPI module](07-mpi.md), this module uses a **dense matrix multiply** ($C = A \times B$, $N = 1024$) rather than the Stream Triad. Matmul performs $2N^3$ floating-point operations on $3N^2$ data — an arithmetic intensity of $O(N)$, which makes it **compute-bound** for large $N$. That is the opposite regime from the bandwidth-bound triad, and it puts the kernel on the flat "compute ceiling" of the [roofline model](02-autovec.md) where extra FLOP/s actually help.

---

## Kokkos Core Concepts

### Execution and memory spaces

Kokkos separates *where code runs* from *where data lives*:

- An **execution space** is a place that runs parallel work — `Kokkos::Serial`, `Kokkos::OpenMP`, `Kokkos::Cuda`, `Kokkos::HIP`. The `DefaultExecutionSpace` is whichever backend you compiled with.
- A **memory space** is a place that stores data — host DRAM (`HostSpace`), CUDA device memory (`CudaSpace`), etc.

On a GPU build, the default execution and memory spaces are the device; on a CPU build, they are the host. Writing to the default spaces is what makes one source portable.

### Views

A `Kokkos::View` is a reference-counted, multidimensional array that lives in a memory space and carries a hardware-optimal data layout:

```cpp
Kokkos::View<double**> A("A", N, N);   // N x N array of doubles on the default device
```

The `double**` denotes rank-2; the string `"A"` is a debug label. By default a View lives in device memory, so you cannot index it directly from host code — you mirror it (below).

### Parallel execution

`Kokkos::parallel_for` launches a kernel. The iteration space is described by an *execution policy*, and the body is a `KOKKOS_LAMBDA` — a macro that expands to a lambda annotated to run on host *or* device:

```cpp
// 1-D range policy: idx runs 0..N*N-1
Kokkos::parallel_for("init", N*N, KOKKOS_LAMBDA(const int idx) { /* ... */ });

// 2-D range policy: (i, j) ranges over the N x N tile
Kokkos::parallel_for("matmul",
  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, N}),
  KOKKOS_LAMBDA(const int i, const int j) { /* ... */ });
```

`MDRangePolicy<Rank<n>>` gives Kokkos an *n*-dimensional iteration space it can tile efficiently for the target hardware — the portable equivalent of CUDA's nested grid/block indexing.

!!! warning "Parallelize the independent loops only"
    In an MDRangePolicy of rank *n*, **all *n* indices run in parallel**. That is correct only when each `(i, j, ...)` iteration writes a distinct output. The matmul reduction over `k` must *not* be one of the parallel ranks, or multiple threads race to update the same `C(i,j)`. Keep `k` as a sequential loop inside the kernel. This is exactly the bug we examine in the MATAR section.

### Synchronization and host access

GPU kernels are asynchronous, so you call `Kokkos::fence()` before timing or reading results. To inspect device data on the host, create a mirror and copy:

```cpp
auto h_C = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), C);
// h_C(i, j) is now safe to read on the host
```

`Kokkos::deep_copy(C, 0.0)` fills a View; `deep_copy(dst, src)` moves data between spaces. Every program brackets its work in `Kokkos::initialize(argc, argv)` / `Kokkos::finalize()`.

---

## The Kokkos Matmul (Example 1)

The core of `examples/Kokkos/Example1/matmul.cpp`:

```cpp
Kokkos::View<double**> A("A", N, N), B("B", N, N), C("C", N, N);

// Random initialization, run in parallel on the device
Kokkos::Random_XorShift64_Pool<> random_pool(12345);
Kokkos::parallel_for("init_matrices", N*N, KOKKOS_LAMBDA(const int idx) {
  const int i = idx / N;
  const int j = idx % N;
  auto rand_gen = random_pool.get_state();
  A(i, j) = rand_gen.drand(0.0, 1.0);
  B(i, j) = rand_gen.drand(0.0, 1.0);
  random_pool.free_state(rand_gen);
});
Kokkos::deep_copy(C, 0.0);

// Matrix multiply: parallel over (i, j), sequential reduction over k
Kokkos::parallel_for("matrix_multiply",
  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, N}),
  KOKKOS_LAMBDA(const int i, const int j) {
    double sum = 0.0;
    for (int k = 0; k < N; ++k) {
      sum += A(i, k) * B(k, j);
    }
    C(i, j) = sum;
  });
Kokkos::fence();
```

This is the **correct** decomposition: each `(i, j)` thread owns one output element and accumulates into a *private* `sum`, so there is no race. The program then mirrors `A`, `B`, `C` to the host, recomputes a 3×3 corner, checks the relative error, and reports GFLOPS as $2N^3 / (t \times 10^6)$.

It also prints `typeid(Kokkos::DefaultExecutionSpace).name()` — a quick way to confirm which backend the binary was built for.

---

## Building for Any Backend

This is where portability pays off. The example ships two scripts:

- **`kokkos-install.sh`** clones Kokkos and builds it for a chosen backend (`-t serial|openmp|pthreads|cuda|hip`), setting flags like `-DKokkos_ENABLE_CUDA=ON` or `-DKokkos_ENABLE_OPENMP=ON`.
- **`build.sh`** invokes the installer, then configures and compiles `matmul.cpp` against that Kokkos install.

The **same `matmul.cpp`** produces a CPU or GPU binary depending only on `-t`:

=== "Workstation — multicore CPU"
    ```bash
    cd examples/Kokkos/Example1
    ./build.sh -t openmp
    cd build_openmp
    export OMP_NUM_THREADS=8
    ./matmul
    ```

=== "Workstation — NVIDIA GPU"
    ```bash
    cd examples/Kokkos/Example1
    ./build.sh -t cuda      # requires nvcc + CUDA toolkit
    cd build_cuda
    ./matmul
    ```

=== "SLURM cluster"
    See `examples/Kokkos/run.slurm`. It takes the backend as an argument and requests a GPU only when needed. Building Kokkos from source the first time takes several minutes, so the script's wall-clock request is generous.

Run the same kernel under `serial`, `openmp`, and (if you have a GPU) `cuda`, and compare the reported GFLOPS. The execution-space line printed at startup confirms each binary really used a different backend — from one unchanged source file.

---

## MATAR: A Higher-Level Layer

[MATAR](https://github.com/lanl/MATAR) is a thin array/loop abstraction built *on top of* Kokkos. It trades some explicitness for terser code: `CArrayKokkos` replaces `Kokkos::View`, and the `FOR_ALL` macro replaces `parallel_for` + policy + lambda. The same matmul, from `examples/MATAR/Example1/matmul.cpp`:

```cpp
using namespace mtr;

CArrayKokkos<int> A(MATRIX_SIZE, MATRIX_SIZE);
CArrayKokkos<int> B(MATRIX_SIZE, MATRIX_SIZE);
CArrayKokkos<int> C(MATRIX_SIZE, MATRIX_SIZE);
A.set_values(2);
B.set_values(2);
C.set_values(0);

FOR_ALL(i, 0, MATRIX_SIZE,
        j, 0, MATRIX_SIZE,
        k, 0, MATRIX_SIZE, {
    C(i,j) += A(i,k) * B(k,j);
});
```

This is undeniably compact. But it harbors a bug, and it is instructive precisely because the bug is invisible until you change backends.

!!! warning "This `FOR_ALL` has a data race"
    The three-index `FOR_ALL(i, …, j, …, k, …)` expands to a Kokkos `MDRangePolicy<Rank<3>>` — it parallelizes **all three** loops, `k` included. So many threads with different `k` simultaneously execute `C(i,j) += …` on the *same* element, a classic read-modify-write race.

    On the **serial** backend there is no concurrency, so it prints the right answer ($C_{ij} = 1024 \times 2 \times 2 = 4096$) and looks fine. The **OpenMP** backend also appears correct in practice: Kokkos's `MDRangePolicy` schedules work in tiles, and with default tile sizes the k iterations for a given (i,j) pair happen to land in the same tile — serializing the race away. The code is still wrong, but the bug is invisible.

    Build with **`-t cuda`** and the race explodes: hundreds of threads with different `k` race to update `C(i,j)` simultaneously, and only the last write survives. The result is nondeterministic and off by a factor of ~1024 — values near 4 instead of 4096, changing run-to-run. This is the canonical GPU surprise: a latent race that passes all CPU testing then fails catastrophically on the first GPU run.

    **The fix** is to keep `k` sequential — parallelize only `i` and `j`:
    ```cpp
    FOR_ALL(i, 0, MATRIX_SIZE,
            j, 0, MATRIX_SIZE, {
        int sum = 0;
        for (int k = 0; k < MATRIX_SIZE; k++) {
            sum += A(i,k) * B(k,j);
        }
        C(i,j) = sum;
    });
    ```

The lesson is not that MATAR is bad — it is a productive abstraction — but that **an abstraction does not relieve you of reasoning about correctness.** A higher-level loop macro makes it *easier* to accidentally parallelize a reduction.

MATAR is built the same way as Kokkos (`build.sh -t <backend>`), since it compiles down to Kokkos.

---

## Assignment

1. **Build and run the Kokkos example** on the `serial` and `openmp` backends. Set `OMP_NUM_THREADS` to your core count. Record GFLOPS for each and confirm the execution-space line changes.

2. **(If you have a GPU)** Build with `-t cuda` and compare. Explain why matmul shows a much larger CPU→GPU speedup than the Stream Triad did in earlier modules. (Hint: arithmetic intensity and the roofline.)

3. **Reproduce the MATAR race.** Build the MATAR example `-t serial` and confirm `C(i,j) = 4096` in the corner output. Build again with `-t openmp` — the result still looks correct (the race is latent, hidden by tiling). Now build with `-t cuda` and observe values near 4 instead of 4096, changing run-to-run. Apply the fix above, rebuild with `-t cuda`, and confirm all elements return to 4096.

??? success "Why matmul scales to the GPU but the triad barely did"
    The Stream Triad moves 24 bytes per 2 FLOPs — intensity ≈ 0.08 FLOP/byte — so it is pinned to the memory-bandwidth ceiling on every device, and a GPU's main advantage there is just its higher memory bandwidth. Matmul does $2N^3$ FLOPs over $3N^2$ data — intensity grows with $N$ — so for $N = 1024$ it sits on the *compute* ceiling, where a GPU's thousands of FPUs deliver a large speedup over a CPU. Same hardware, opposite regime, because of arithmetic intensity.

---

## Discussion Questions

1. Kokkos chooses the backend at **compile** time, not run time. What are the advantages of that design over a runtime dispatch, and what flexibility does it cost?

2. The Kokkos matmul accumulates into a private `double sum` and writes `C(i,j)` once; the buggy MATAR version updates `C(i,j)` in place across a parallel `k`. Beyond correctness, why is the private-accumulator version also usually *faster* on a GPU?

3. A `Kokkos::View` defaults to device memory, requiring mirror views for host access. How does this explicit host/device separation compare to the unified-memory model used by `std::par` ([Module 5](05-stdpar.md)) and OpenMP offload ([Module 6](06-omp-offload.md))? When is explicit better?

4. You have a CUDA code today and an AMD GPU cluster arriving next year. Estimate the relative effort of (a) porting the CUDA by hand to HIP versus (b) rewriting it in Kokkos now. What assumptions drive your answer?

---

## References

### Reference materials

- [Kokkos](https://github.com/kokkos/kokkos) — the performance-portability library.
- [Kokkos documentation (Core Wiki)](https://kokkos.org/kokkos-core-wiki/) — programming guide and API reference.
- [MATAR](https://github.com/lanl/MATAR) — the array/loop abstraction layered on top of Kokkos.
