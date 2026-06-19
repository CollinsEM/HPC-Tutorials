# Module 3: OpenMP — CPU Parallelism

## Learning Objectives

By the end of this module you will be able to:

- Describe the OpenMP fork-join execution model
- Add `#pragma omp parallel` and `#pragma omp parallel for` to a serial loop
- Control thread count and query thread ID at runtime
- Time a parallel region and compute speedup
- Recognize data race conditions and use the `reduction` clause to avoid them

---

## 1. OpenMP Basics

**OpenMP** is a directive-based API for shared-memory parallelism. It requires
no changes to data structures. You annotate loops and parallel regions with
`#pragma` directives, link with `-fopenmp`, and the compiler generates
multithreaded code. The same source file compiles correctly without `-fopenmp`
and runs serially — the pragmas are ignored by compilers that do not recognize
them.

### The Fork-Join Model

OpenMP programs follow a **fork-join** execution model:

1. The program begins as a single *master thread*.
2. When execution reaches a `#pragma omp parallel` block, the runtime **forks**
   a team of threads. All threads execute the block concurrently.
3. At the closing `}`, threads **join** and the master thread continues alone.

```
master ─────────────────┬──────────────────────┬─────────────────▶
                        │  fork                │  join
                    ────┼────────────────────  │
thread 1 ───────────▶  │  ────────────────▶   │
thread 2 ───────────▶  │  ────────────────▶   │
thread N ───────────▶  │  ────────────────▶   │
                        └──────────────────────┘
```

### Controlling Threads

| Mechanism | Effect |
|-----------|--------|
| `OMP_NUM_THREADS=N` (env var) | Set default team size before launching the program |
| `omp_set_num_threads(N)` | Set team size in code before entering a parallel region |
| `omp_get_num_threads()` | Query team size from inside a parallel region |
| `omp_get_thread_num()` | Query caller's zero-based rank (0 = master thread) |

!!! warning "Calling omp_get_num_threads() outside a parallel region"
    Outside a `#pragma omp parallel` block, `omp_get_num_threads()` always
    returns 1 regardless of `OMP_NUM_THREADS`. This is a common source of
    confusion in the first example below.

---

## 2. Example 1 — Hello OpenMP

Source: `examples/OpenMP/Example1/HelloOpenMP.c`

```c
#include <stdio.h>
#include <omp.h>

int main(int argc, char *argv[]){
  int nthreads, thread_id;

  nthreads = omp_get_num_threads();
  thread_id = omp_get_thread_num();

  printf("Goodbye slow serial world and Hello OpenMP!\n");
  printf("  I have %d thread(s) and my thread id is %d\n", nthreads, thread_id);
}
```

As written, both `omp_get_*` calls execute outside any parallel region, so
`nthreads` is always 1 and `thread_id` is always 0. The parallel region is
missing.

**Assignment:** Wrap the body of `main` in `#pragma omp parallel { ... }`,
set `OMP_NUM_THREADS=4`, and verify that all four threads print.

??? success "Show solution"
    ```c
    #include <stdio.h>
    #include <omp.h>

    int main(int argc, char *argv[]){
      int nthreads, thread_id;
    #pragma omp parallel
      {
        nthreads = omp_get_num_threads();
        thread_id = omp_get_thread_num();
        printf("Goodbye slow serial world and Hello OpenMP!\n");
        printf("  I have %d thread(s) and my thread id is %d\n", nthreads, thread_id);
      }
    }
    ```
    Run with:
    ```bash
    export OMP_NUM_THREADS=4
    ./hello_omp
    ```
    You will see 4 sets of lines printed in non-deterministic order — this is
    correct. Threads are independent and the scheduler may interleave their
    output in any order.

---

## 3. Example 2 — Parallel Vector Addition

Source: `examples/OpenMP/Example2/vecadd.c`

```c
#include <stdio.h>
#include "timer.h"

#define ARRAY_SIZE 8000000
static double a[ARRAY_SIZE], b[ARRAY_SIZE], c[ARRAY_SIZE];

void vector_add(double *c, double *a, double *b, int n);

int main(int argc, char *argv[]){
  struct timespec tstart;
  double time_sum = 0.0;
  for (int i = 0; i < ARRAY_SIZE; i++) {
    a[i] = 1.0;
    b[i] = 2.0;
  }

  cpu_timer_start(&tstart);
  vector_add(c, a, b, ARRAY_SIZE);
  time_sum += cpu_timer_stop(tstart);

  printf("Runtime is %lf msecs\n", time_sum);
}

void vector_add(double *c, double *a, double *b, int n)
{
  for (int i = 0; i < n; i++){
    c[i] = a[i] + b[i];
  }
}
```

Each iteration is independent: `c[i]` depends only on `a[i]` and `b[i]`, not
on any other iteration. This is the ideal case for `#pragma omp parallel for`.

**Assignment:** Add `#pragma omp parallel for` to the loop in `vector_add`.
Print the thread count in `main`. Measure speedup over the serial version.

??? success "Show solution"
    ```c
    #include <stdio.h>
    #include <omp.h>
    #include "timer.h"

    #define ARRAY_SIZE 8000000
    static double a[ARRAY_SIZE], b[ARRAY_SIZE], c[ARRAY_SIZE];

    void vector_add(double *c, double *a, double *b, int n);

    int main(int argc, char *argv[]){
    #pragma omp parallel
      if (omp_get_thread_num() == 0)
        printf("Running with %d thread(s)\n", omp_get_num_threads());

      struct timespec tstart;
      double time_sum = 0.0;
      for (int i = 0; i < ARRAY_SIZE; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
      }

      cpu_timer_start(&tstart);
      vector_add(c, a, b, ARRAY_SIZE);
      time_sum += cpu_timer_stop(tstart);

      printf("Runtime is %lf msecs\n", time_sum);
    }

    void vector_add(double *c, double *a, double *b, int n)
    {
    #pragma omp parallel for
      for (int i = 0; i < n; i++){
        c[i] = a[i] + b[i];
      }
    }
    ```

    !!! note
        The short `#pragma omp parallel` block in `main` is used only to query
        the thread count before timing begins. It spawns and immediately
        terminates a team of threads; the real work happens in `vector_add`.

---

## 4. Example 3 — Health Data Analysis

Source: `examples/OpenMP/Example3/HealthData.cpp`

This exercise scales up to a realistic data size: 1,000,000 patients, each
described by 10 features drawn from a uniform distribution on [50, 150]. The
program computes, serially, the mean and variance of each feature and the
full $10 \times 10$ correlation matrix. The serial implementation takes several
seconds; your task is to parallelize the hot loops.

### Data Structure

The data is stored as `std::vector<std::vector<double>> data(NumPatients,
std::vector<double>(NumFeatures))`. This is **array-of-structs (AoS)** layout:
each row `data[i]` is a contiguous vector of 10 doubles for patient $i$, but
rows themselves are separate heap allocations. Accessing column $j$ across all
rows (`data[0][j], data[1][j], ...`) requires strided loads — each stride is
the size of one row's heap block plus the allocator overhead. This is
intentionally not the optimal layout for the inner loops, which is part of
the discussion at the end of this section.

### Serial Source

```cpp
#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <omp.h>

constexpr int NumPatients = 1000000;
constexpr int NumFeatures = 10;

// RAII timer: prints elapsed time on destruction
class Timer {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    std::string function_name;
public:
    Timer(const std::string& name) : function_name(name) {
        start_time = std::chrono::high_resolution_clock::now();
        std::cout << "Starting " << function_name << "...\n";
    }
    ~Timer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        std::cout << function_name << " completed in " << duration.count() << " ms\n";
    }
};

void generate_data(std::vector<std::vector<double>>& data) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(50, 150);
    for (int i = 0; i < NumPatients; i++)
        for (int j = 0; j < NumFeatures; j++)
            data[i][j] = dist(gen);
}

void compute_mean(const std::vector<std::vector<double>>& data,
                  std::vector<double>& mean) {
    for (int j = 0; j < NumFeatures; j++) {
        double sum = 0.0;
        for (int i = 0; i < NumPatients; i++)
            sum += data[i][j];
        mean[j] = sum / NumPatients;
    }
}

void compute_variance(const std::vector<std::vector<double>>& data,
                      const std::vector<double>& mean,
                      std::vector<double>& variance) {
    for (int j = 0; j < NumFeatures; j++) {
        double sum_sq = 0.0;
        for (int i = 0; i < NumPatients; i++)
            sum_sq += std::pow(data[i][j] - mean[j], 2);
        variance[j] = sum_sq / (NumPatients - 1);  // sample variance
    }
}

void compute_correlation(const std::vector<std::vector<double>>& data,
                         const std::vector<double>& mean,
                         const std::vector<double>& variance,
                         std::vector<std::vector<double>>& correlation) {
    for (int j1 = 0; j1 < NumFeatures; j1++) {
        for (int j2 = 0; j2 < NumFeatures; j2++) {
            if (j1 == j2) { correlation[j1][j2] = 1.0; continue; }
            double sum = 0.0;
            for (int i = 0; i < NumPatients; i++)
                sum += (data[i][j1] - mean[j1]) * (data[i][j2] - mean[j2]);
            correlation[j1][j2] = sum /
                ((NumPatients - 1) * std::sqrt(variance[j1]) * std::sqrt(variance[j2]));
        }
    }
}

int main() {
    std::cout << "Analyzing health data for " << NumPatients
              << " patients with " << NumFeatures << " features\n";
    {
        Timer timer("Total Runtime");
        std::vector<std::vector<double>> data(NumPatients,
            std::vector<double>(NumFeatures));
        std::vector<double> mean(NumFeatures, 0.0), variance(NumFeatures, 0.0);
        std::vector<std::vector<double>> correlation(NumFeatures,
            std::vector<double>(NumFeatures, 0.0));

        { Timer t("Generate Data");    generate_data(data); }
        { Timer t("Compute Mean");     compute_mean(data, mean); }
        { Timer t("Compute Variance"); compute_variance(data, mean, variance); }
        { Timer t("Compute Correlation");
          compute_correlation(data, mean, variance, correlation); }

        std::cout << "\nHealth Feature Summary:\n";
        for (int j = 0; j < NumFeatures; j++)
            std::cout << "Feature " << j << " - Mean: " << mean[j]
                      << ", Variance: " << variance[j] << "\n";

        std::cout << "\nCorrelation Matrix (first 3x3 section):\n";
        for (int j1 = 0; j1 < 3; j1++) {
            for (int j2 = 0; j2 < 3; j2++)
                std::cout << correlation[j1][j2] << " ";
            std::cout << "\n";
        }
    }
    return 0;
}
```

### Assignment Tasks

1. Build and time the serial version with 1 thread (`OMP_NUM_THREADS=1`).

2. Add `#pragma omp parallel for reduction(+:sum)` to the inner loop over
   patients in `compute_mean`, `compute_variance`, and `compute_correlation`.
   The `reduction` clause is required wherever multiple threads accumulate into
   a shared scalar — omitting it creates a data race.

3. Verify correctness: mean values should be near 100 for all features (the
   distribution is uniform on [50, 150], so $\mu = 100$).

4. Sweep the thread count from 1 to the number of physical cores on your
   machine and record the speedup for each function independently.

??? success "Show solution (mean computation)"
    ```cpp
    void compute_mean(const std::vector<std::vector<double>>& data,
                      std::vector<double>& mean) {
      for (int j = 0; j < NumFeatures; j++) {
        double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < NumPatients; i++) {
          sum += data[i][j];
        }
        mean[j] = sum / NumPatients;
      }
    }
    ```
    The `reduction(+:sum)` clause gives each thread a **private copy** of `sum`
    initialized to zero. Each thread accumulates into its own copy. At the end
    of the parallel loop the runtime adds all private copies together and stores
    the result in the original `sum` variable. This eliminates the data race
    without requiring any explicit synchronization in user code.

    Apply the same pattern to the `sum_sq` accumulator in `compute_variance`
    and the `sum` accumulator in the innermost loop of `compute_correlation`.

!!! warning "Race condition without reduction"
    Without `reduction(+:sum)`, multiple threads simultaneously read and write
    the same memory location. The result is non-deterministic and almost
    certainly wrong. The bug may not manifest on every run, which makes it
    particularly dangerous. Always use `reduction` (or a critical section, or
    thread-local storage) when accumulating across threads.

---

## 5. Build and Run

=== "Workstation"
    ```bash
    # Example 1 — Hello OpenMP
    cd examples/OpenMP/Example1
    mkdir build && cd build
    cmake ..
    make
    export OMP_NUM_THREADS=4
    ./hello_omp

    # Example 2 — Vector addition
    cd ../../Example2
    mkdir build && cd build
    cmake ..
    make
    export OMP_NUM_THREADS=4
    ./vecadd

    # Example 3 — Health data analysis
    cd ../../Example3
    mkdir build && cd build
    cmake ..
    make
    export OMP_NUM_THREADS=8
    ./health_data
    ```

=== "SLURM cluster"
    See `examples/OpenMP/run.slurm`. Adjust `--cpus-per-task` to match the
    number of threads you want to test. The script exports `OMP_NUM_THREADS` to
    match `--cpus-per-task` automatically.

    ```bash
    sbatch examples/OpenMP/run.slurm
    ```

---

## 6. Expected Results

For Example 3 with 8 threads you should observe roughly 6–7x speedup over
the 1-thread baseline for `compute_mean` and `compute_variance`. The speedup
for `compute_correlation` is typically higher because it has more work per call.

Performance will plateau — and may even decrease — above the number of
**physical cores** on your socket. Hyperthreading (SMT) adds logical cores but
shares the memory subsystem, so memory-bound loops gain little from it.

### Discussion Questions

1. In `compute_correlation` there are two nested loops over features (outer,
   $j_1, j_2 \in [0, 10)$) and one inner loop over patients ($i \in [0, 10^6)$).
   Which loop should carry the `#pragma omp parallel for`, and why? What happens
   if you place it on the outer feature loop instead?

2. Why does speedup plateau below the thread count? Use Amdahl's Law and the
   roofline model together: what fraction of the runtime is serial, and does
   bandwidth saturation impose a separate ceiling?

3. The data is stored as `vector<vector<double>>`. Accessing column $j$ across
   all rows requires a pointer dereference plus a stride equal to the row
   allocation size — this is not cache-friendly for the inner loop. How would
   changing to a flat `double data[NumPatients * NumFeatures]` layout with
   row-major indexing (`data[i * NumFeatures + j]`) affect cache behavior and
   SIMD auto-vectorization?
