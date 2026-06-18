# Module 2: Auto-Vectorization and the Stream Triad

## Learning Objectives

By the end of this module you will be able to:

- Explain SIMD vectorization and why it matters for single-core performance
- Identify vectorization methods in ascending order of programmer effort
- Enable and verify compiler auto-vectorization using flags and diagnostic reports
- Measure memory bandwidth using LIKWID performance counters
- Interpret the roofline model and locate the Stream Triad on it

---

## 1. Vectorization: SIMD Parallelism on a Single Core

Modern CPUs contain *vector units* — wide execution pipelines that apply a single
instruction to multiple data elements simultaneously. This is called **SIMD**
(Single Instruction, Multiple Data) parallelism.

A scalar addition processes one `double` per cycle. A 512-bit vector unit
(AVX-512) holds 8 doubles and processes all 8 in one instruction — an 8x
throughput increase over scalar code, all on a single core, at no additional
clock frequency.

There is a convenient coincidence in the hardware: a 64-byte cache line holds
exactly 8 doubles. A perfectly vectorized inner loop therefore processes one
entire cache line per clock cycle.

### Key Terminology

| Term | Definition |
|------|------------|
| *Vector lane* | One element slot in a vector register (analogous to a lane on a freeway) |
| *Vector width* | Register width in bits: 128/256/512 for SSE/AVX/AVX-512 |
| *Vector length* | Number of elements processed per instruction = width / element\_size\_bits |
| *SIMD instruction set* | SSE2, AVX, AVX2, AVX-512 on x86; NEON/SVE on ARM |

!!! note "Two requirements for vectorization"
    Vectorization requires both software and hardware support. The compiler must
    generate vector instructions **and** the CPU must implement the corresponding
    instruction set. Compiling with `-march=native` ensures the compiler targets
    the exact CPU you are running on. Targeting a broader ISA (e.g., generic SSE2
    for portability) leaves AVX-512 throughput on the table.

---

## 2. Vectorization Methods (Ascending Programmer Effort)

| Level | Method | Notes |
|-------|--------|-------|
| 1 | **Optimized libraries** (BLAS, LAPACK, FFTW, Intel MKL) | Already fully vectorized; use when the operation fits |
| 2 | **Auto-vectorization** | Compiler analyzes loops and emits SIMD instructions automatically |
| 3 | **Compiler hints** (`#pragma GCC ivdep`, `restrict`, OpenMP SIMD) | Remove ambiguities that block auto-vectorization |
| 4 | **Vector intrinsics** (`_mm512_add_pd`, etc.) | Explicit SIMD; portable across compilers, not across ISAs |
| 5 | **Assembler** | Maximum control, zero portability |

Auto-vectorization is the recommended starting point. It requires zero code
changes and handles the majority of loops correctly. Its limitation is that the
compiler must make conservative assumptions: it does not know your array sizes at
compile time and must assume that pointer arguments could alias each other. It
therefore sometimes declines to vectorize even when it would be safe to do so.
The higher levels of the table exist to overcome those conservative assumptions.

---

## 3. The Stream Triad Benchmark

The **STREAM benchmark** is the standard measure of *sustainable* memory
bandwidth — the bandwidth a real application can sustain, as opposed to the
theoretical peak stated in a hardware datasheet. The **Triad** kernel is:

$$A[i] = B[i] + \alpha \times C[i]$$

where $A$, $B$, $C$ are large arrays (much larger than any cache level) and
$\alpha$ is a scalar constant. The arrays must be large enough that no useful
data remains in cache at the start of each iteration; otherwise the benchmark
measures cache bandwidth rather than DRAM bandwidth.

### Arithmetic Intensity of the Triad

For each element $i$:

- **Floating-point operations**: 1 multiply ($\alpha \times C[i]$) + 1 add = **2 FLOP**
- **Memory traffic**: read $B[i]$ + read $C[i]$ + write $A[i]$ = $3 \times 8$ bytes = **24 bytes**

$$I = \frac{2 \text{ FLOP}}{24 \text{ bytes}} \approx 0.083 \text{ FLOP/byte}$$

This is an extremely low arithmetic intensity. On nearly every machine the Triad
falls deep in the bandwidth-limited region of the roofline model: adding more
floating-point hardware cannot help because the bottleneck is moving data from
DRAM to the CPU.

---

## 4. The Roofline Model

The **Roofline model** is a visual performance bound that answers the question:
*given this hardware, what is the maximum attainable performance for a kernel
with this arithmetic intensity?*

The model plots attainable FLOP/s (y-axis, log scale) against arithmetic
intensity (FLOP/byte, x-axis, log scale). Two bounds define the "roofline":

- **Bandwidth ceiling** (diagonal): $\text{attainable FLOP/s} = I \times B_\text{peak}$,
  where $B_\text{peak}$ is peak memory bandwidth in GB/s.
- **Compute ceiling** (horizontal): $P_\text{peak}$, the processor's peak
  floating-point throughput.

The intersection of the two ceilings is the **ridge point**. Kernels whose
arithmetic intensity falls to the left of the ridge point are
*bandwidth-bound*; kernels to the right are *compute-bound*.

The Stream Triad, with $I \approx 0.08$ FLOP/byte, always falls far to the left
of the ridge point on modern hardware. Its performance ceiling is:

$$\text{Triad}_\text{peak} = 0.083 \text{ FLOP/byte} \times B_\text{peak} \text{ GB/s}$$

!!! tip "Implication for optimization"
    Because the Triad is bandwidth-bound, vectorization and instruction-level
    tricks cannot raise it above the bandwidth ceiling. The only levers are:
    reducing data movement (e.g., fusing loops to reuse data in cache) or
    increasing arithmetic reuse (increasing $I$). Vectorization can still help
    by ensuring the hardware's prefetch and load units are fully utilized, but
    gains are modest compared to a compute-bound kernel.

---

## 5. Vectorization Flags by Compiler

| Compiler | Vectorization flags | Report flag |
|----------|--------------------|----|
| GCC | `-O3 -ftree-vectorize -march=native` | `-fopt-info-vec-optimized` |
| GCC 8+ | add `-mprefer-vector-width=512` for AVX-512 preference | same |
| Clang | `-O3 -fvectorize -march=native` | `-Rpass-analysis=loop-vectorize` |
| Intel icc | `-O3 -xHost -restrict` | `-qopt-report=5 -qopt-report-phase=vec` |
| Intel 18+ | add `-qopt-zmm-usage=high` to prefer 512-bit registers | same |

!!! note "CMakeLists.txt in this example"
    `examples/autovec/CMakeLists.txt` already contains the GCC and Clang flag
    blocks; they are commented out by default so you can toggle them on and
    observe the effect. The Intel block is active when `icc` is detected.

---

## 6. Source Code

The full source is at `examples/autovec/stream_triad.c`:

```c
// To run with Likwid, use the following command:
// likwid-perfctr -C 0 -g MEM1 ./stream_triad

#include <stdio.h>
#include "timer.h"

#define NTIMES 16
// large enough to force into main memory
#define STREAM_ARRAY_SIZE 800000
static double a[STREAM_ARRAY_SIZE], b[STREAM_ARRAY_SIZE], c[STREAM_ARRAY_SIZE];

int main(int argc, char *argv[]){
  struct timespec tstart;

  double scalar = 3.0, time_sum = 0.0;
  for (int i = 0; i < STREAM_ARRAY_SIZE; i++) {
    a[i] = 1.0;
    b[i] = 2.0;
  }
  for (int k = 0; k < NTIMES; k++){
    cpu_timer_start(&tstart);

    for (int i = 0; i < STREAM_ARRAY_SIZE; i++){
      c[i] = a[i] + scalar*b[i];
    }

    time_sum += cpu_timer_stop(tstart);

    // prevent the compiler from optimizing out the loop
    c[1] = c[2];
  }
  printf("Average runtime is %lf msecs\n", time_sum/NTIMES);
}
```

Several details are worth noting:

- **Static allocation**: `a`, `b`, `c` are declared `static` at file scope.
  This guarantees they are contiguous in the BSS segment and aligned to at least
  8-byte boundaries, which is a prerequisite for many SIMD load/store
  instructions.
- **NTIMES averaging**: The Triad runs 16 times and the reported runtime is the
  average. The first iteration may be slower due to cold-cache effects and OS
  page faults; averaging over 16 runs captures the steady-state bandwidth.
- **Dead-code prevention**: `c[1] = c[2]` after each iteration creates a
  data dependency on the array that the compiler cannot eliminate. Without it, a
  sufficiently aggressive optimizer could determine that `c` is never read after
  the loop and delete the Triad entirely.

---

## 7. Assignment

### Tasks

1. **Baseline**: Build and run the unmodified code. Record the average runtime.

2. **Topology inspection**: Run `likwid-topology` to inspect your system's cache
   hierarchy and core layout. Note the L1/L2/L3 sizes and the number of memory
   channels.

3. **Bandwidth measurement**: Run with LIKWID memory performance counters to
   observe the actual DRAM bandwidth during the Triad:
   ```bash
   likwid-perfctr -C 0 -g MEM1 ./stream_triad
   ```
   Record the reported bandwidth in GB/s.

4. **Enable vectorization**: Open `examples/autovec/CMakeLists.txt`. Uncomment
   the GCC flags `-fstrict-aliasing -ftree-vectorize -march=native
   -fopt-info-vec-optimized` in the GNU block. Rebuild and run. The
   `-fopt-info-vec-optimized` flag prints a line for each loop that was
   successfully vectorized; verify it reports the Triad loop.

5. **(GCC 8+) Prefer 512-bit**: Uncomment `-mprefer-vector-width=512`. Rebuild
   and measure again.

6. **Tabulate results**: For each build (baseline, vectorized, AVX-512
   preferred), record: average runtime, vectorization report output, LIKWID
   MEM1 bandwidth, and bandwidth as a fraction of theoretical peak.

### Build and Run

=== "Workstation"
    ```bash
    cd examples/autovec
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make
    ./stream_triad
    ```

=== "SLURM cluster"
    Submit the provided job script:
    ```bash
    sbatch examples/autovec/run.slurm
    ```
    The script pins the process to a single core and requests exclusive node
    access to avoid bandwidth contention from other jobs.

### Expected Output

```
Average runtime is 1.234567 msecs
```

Compute the achieved bandwidth from the runtime:

$$B = \frac{3 \times 800000 \times 8 \text{ bytes}}{t_\text{s} \times 10^9} \text{ GB/s}$$

where $t_\text{s}$ is the average runtime in seconds. Compare to the bandwidth
reported by LIKWID and to the theoretical peak for your system.

### Discussion Questions

1. What is the arithmetic intensity of the Stream Triad? Locate it on the
   roofline model for your machine. Is it bandwidth-bound or compute-bound?

2. Did enabling vectorization change the runtime? By how much? Does the compiler
   report (`-fopt-info-vec-optimized`) confirm that the Triad loop was
   vectorized?

3. What would happen to performance if pointers `a`, `b`, `c` could alias each
   other? Add the `restrict` keyword to the function signature and observe
   whether the compiler report changes. (The static arrays in this example
   cannot actually alias, but the compiler does not always know that.)
