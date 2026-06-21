# Module 5: C++17 Standard Parallel Algorithms

## Learning Objectives

By the end of this module you will be able to:

- Explain the three C++17 execution policies: `seq`, `par`, `par_unseq`
- Use `std::transform` with a parallel execution policy for the Stream Triad
- Compile for GPU execution with NVIDIA's `nvc++` and `-stdpar=gpu`
- Compare `std::par` to explicit [OpenMP](03-openmp.md) and [CUDA](04-cuda.md) implementations

---

## 1. Standard Parallel Algorithms

C++17 added [*execution policies*](https://en.cppreference.com/w/cpp/algorithm/execution_policy_tag_t) to most `<algorithm>` functions. The policy is
passed as the first argument and tells the implementation how it may schedule work:

| Policy | Meaning |
|--------|---------|
| `std::execution::seq` | Sequential, in-order — identical to pre-C++17 algorithms |
| `std::execution::par` | Parallel, may execute on multiple threads; order unspecified |
| `std::execution::par_unseq` | Parallel **and** vectorized; may interleave within a thread (SIMD) |

These policies apply to a broad set of algorithms: `std::transform`, `std::reduce`,
`std::sort`, `std::for_each`, `std::fill`, and others.

!!! note "Backend dependency"
    On CPU, `par` typically dispatches to Intel's libTBB or an OpenMP thread pool,
    depending on how the standard library was built. With NVIDIA's [`nvc++`](https://developer.nvidia.com/hpc-sdk) compiler
    and `-stdpar=gpu`, the **same** `par` policy compiles to CUDA kernels with no
    source changes. The execution policy is the only knob.

!!! warning "par_unseq and thread safety"
    `par_unseq` permits the runtime to vectorize *within* a thread and interleave
    iterations in ways that are unsafe if your callable acquires locks or calls
    non-reentrant functions. For pure arithmetic lambdas this is never a problem,
    but be cautious if the lambda has side effects.

---

## 2. Stream Triad with `std::transform`

The Stream Triad computes:

$$c_i = a_i + s \cdot b_i \quad \forall \, i \in [0, N)$$

With standard parallel algorithms, this maps directly to `std::transform` with a
two-range overload: it reads from two input iterators simultaneously and writes to
an output iterator. The lambda provides the per-element operation; the execution
policy controls how iterations are scheduled.

The complete source is at `examples/stdpar/triad/triad.cc`:

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <execution>
#include <numeric>
#include <cmath>

void vector_operation(const std::vector<double>& a,
                      const std::vector<double>& b,
                      std::vector<double>& c,
                      double scalar) {
  if (a.size() != b.size() || a.size() != c.size())
    throw std::runtime_error("Vectors must have the same size.");

  std::transform(std::execution::par,
                 a.begin(), a.end(),
                 b.begin(),
                 c.begin(),
                 [scalar](double a_val, double b_val) {
                   return a_val + scalar * b_val;
                 });
}

int main() {
  size_t size = 100;
  std::vector<double> a(size), b(size), c(size);
  double scalar = 2.0;

  std::iota(a.begin(), a.end(), 1.0);
  std::transform(a.begin(), a.end(), b.begin(),
                 [](double x){ return std::sqrt(x * 10); });

  vector_operation(a, b, c, scalar);

  // verify
  for (int i=0; i<size; ++i) {
    if (fabs(c[i] - (a[i] + scalar*b[i])) > 1e-8)
      std::cout << "Error: " << c[i] << " != " << (a[i] + scalar*b[i]) << "\n";
  }
  return 0;
}
```

The critical line is:

```cpp
std::transform(std::execution::par, a.begin(), a.end(), b.begin(), c.begin(), lambda);
```

Swapping `par` for `seq` or `par_unseq` is all that is needed to change the
parallelism strategy. When compiled with `nvc++ -stdpar=gpu`, `par` becomes a
GPU kernel automatically.

---

## 3. Build Flags

=== "CPU (OpenMP backend)"
    Most compilers support `std::execution::par` via libTBB or OpenMP:

    ```bash
    # GCC with libTBB
    g++ -std=c++17 -O3 -ltbb triad.cc -o triad

    # GCC with OpenMP
    g++ -std=c++17 -O3 -fopenmp triad.cc -o triad
    ```

    If neither TBB nor OpenMP is available, `par` falls back silently to sequential
    execution on some implementations (GCC with libstdc++ pre-11 may do this).

=== "GPU (nvc++ / NVIDIA HPC SDK)"
    ```bash
    nvc++ -std=c++17 -stdpar=gpu -O3 triad.cc -o triad_gpu
    ```

    `-stdpar=gpu` routes `std::execution::par` to CUDA. Memory management is
    handled transparently via CUDA Unified Memory — `std::vector` data is
    automatically migrated to the GPU on first access by a kernel.

    See `examples/stdpar/triad/Makefile` for the complete build recipe.

=== "SLURM cluster"
    See `examples/stdpar/triad/run.slurm`. Load the NVIDIA HPC SDK module and
    use the GPU build flags above. Check available modules with `module avail nvhpc`.

---

## 4. Assignment

The current code uses `size = 100`, which is far too small to see meaningful parallel
speedup or to measure bandwidth accurately.

### Task 1 — Scale up and add timing

Increase `size` to 80,000,000 (matching the CUDA example). Add timing around
`vector_operation()` using `std::chrono::high_resolution_clock`.

### Task 2 — Compare execution policies

Run with each of the three policies and record the wall time:

- `std::execution::seq`
- `std::execution::par`
- `std::execution::par_unseq`

Change only the policy argument inside `vector_operation()`.

### Task 3 — GPU comparison (if nvc++ is available)

Compile with `-stdpar=gpu` and compare the reported time to the CUDA Example1 kernel
time. Note whether the comparison is fair (hint: consider what Unified Memory is doing
on first access).

??? success "Show solution — timing and size changes"
    ```cpp
    #include <chrono>

    int main() {
      size_t size = 80000000;
      std::vector<double> a(size, 1.0);
      std::vector<double> b(size, 2.0);
      std::vector<double> c(size, 0.0);
      double scalar = 3.0;

      auto t0 = std::chrono::high_resolution_clock::now();
      vector_operation(a, b, c, scalar);
      auto t1 = std::chrono::high_resolution_clock::now();

      double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      // Three arrays, 8 bytes each, read or written once per element.
      double bw = 3.0 * size * 8.0 / (ms * 1e-3) / 1e9;
      std::cout << "Runtime: " << ms << " ms  BW: " << bw << " GB/s\n";
    }
    ```

    Change `std::execution::par` to `seq` or `par_unseq` inside `vector_operation()`
    to compare policies without touching `main`.

---

## 5. Comparison to Other Approaches

| Approach | Code complexity | Portability | Typical overhead |
|----------|----------------|-------------|-----------------|
| Serial (`seq`) | Baseline | Any C++17 compiler | — |
| OpenMP | One pragma per loop | CPU only (without offload) | Low |
| CUDA | Explicit kernel + memory management | NVIDIA GPUs only | High (explicit) |
| `std::par` (CPU) | Zero changes from serial | C++17 + TBB or OpenMP | Low |
| `std::par` (GPU via nvc++) | Zero code changes | NVIDIA only (compiler flag) | Unified memory overhead |

The appeal of `std::par` is that the same source file compiles to CPU threads or
GPU kernels by changing a compiler flag. The tradeoff: less control over memory
placement and transfer timing than explicit CUDA.

!!! tip "When std::par shines"
    `std::par` is attractive for porting existing CPU code to GPU quickly, or for
    code that must remain readable and portable. If you need fine-grained control
    over memory transfers, stream concurrency, or shared memory usage, explicit
    CUDA gives you options that `std::par` cannot expose.

---

## 6. Discussion Questions

1. `std::execution::par_unseq` allows the implementation to use SIMD within threads.
   On CPU, under what conditions does this outperform `par` alone? When might it
   perform worse?

2. Unified Memory (used by `nvc++ -stdpar=gpu`) migrates pages between CPU and GPU
   on first access. How does this compare to explicit `cudaMemcpy` in terms of peak
   bandwidth and first-call latency?

3. When would you choose explicit CUDA over `std::par` even if both produce correct
   results? Name at least two scenarios where the extra complexity is justified.
