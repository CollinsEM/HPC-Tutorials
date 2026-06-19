# Module 9: Profiling & Performance Analysis

## Learning Objectives

By the end of this module you will be able to:

- Apply the "measure, don't guess" discipline and target the bottleneck that matters
- Inspect a machine's topology and bind threads with LIKWID
- Measure achieved memory bandwidth and cache behavior with hardware counters
- Find hotspots, cache misses, and memory errors with Valgrind
- Place a kernel on the roofline model and decide whether it is compute- or bandwidth-bound

---

## Measure, Don't Guess

The most common performance mistake is optimizing the wrong thing. [Amdahl's Law](00-parallel-theory.md) is blunt about it: speeding up code that accounts for 5% of runtime can never give more than a 5% improvement, no matter how clever the optimization. Before you change anything, **measure** where time actually goes and **what limits it** — compute throughput, memory bandwidth, or latency.

A disciplined workflow:

1. **Time** the whole program and its phases to find where the time is.
2. **Profile** the hot region to find *why* it is slow (counters, cache misses, call graph).
3. **Classify** the bottleneck with the roofline: compute-bound or bandwidth-bound?
4. **Optimize** the dominant cost.
5. **Re-measure** to confirm a real gain and catch regressions.

This module covers two complementary tools — **LIKWID** for hardware-counter and bandwidth analysis, **Valgrind** for hotspots, cache simulation, and memory correctness — and the **roofline model** that ties their numbers to a decision.

!!! note "Tools used here"
    LIKWID, Valgrind, and KCachegrind can be installed via the repository's `install_stuff.sh` (LIKWID needs `sudo` and access to hardware performance counters). The examples below profile the auto-vectorization benchmark from [Module 2](02-autovec.md), `examples/autovec/stream_triad.c`, which already carries a LIKWID marker comment at the top.

---

## LIKWID — "Like I Knew What I'm Doing"

LIKWID is a lightweight command-line toolkit for performance measurement on CPUs. The pieces you will use most:

### Inspect the machine: `likwid-topology`

```bash
likwid-topology -g
```

This prints the socket/core/cache layout and a graphical map of the cache hierarchy. You need it to interpret everything else — to know how big L1/L2/L3 are (so you know when an array spills to DRAM), how many physical cores exist, and how hyperthreads and NUMA domains are arranged.

### Measure hardware counters: `likwid-perfctr`

List the available *performance groups* for your CPU:

```bash
likwid-perfctr -a
```

Then run your program pinned to a core under a group. For a memory-bound kernel, the `MEM` (or `MEM1`) group reports achieved DRAM bandwidth:

```bash
likwid-perfctr -C 0 -g MEM1 ./stream_triad
```

- `-C 0` pins execution to core 0 (measurements are meaningless if the OS migrates the thread).
- `-g MEM1` selects the memory-bandwidth counter group.

The output reports bytes read/written and an achieved bandwidth in MB/s or GB/s. Compare that to your CPU's theoretical peak (from the spec sheet, or measured with the STREAM benchmark) — for the triad, you should be near peak if the loop vectorized and the prefetchers are keeping up.

Other useful groups: `FLOPS_DP`/`FLOPS_SP` (floating-point rate, needed for the roofline), `L2`/`L3` (cache traffic), `BRANCH` (misprediction), `CLOCK` (effective frequency).

### Pin threads: affinity matters

For OpenMP code, *where* threads run dominates memory performance on NUMA systems. LIKWID can launch and pin a threaded run:

```bash
likwid-pin -c 0-7 ./vecadd          # pin 8 threads to cores 0-7
likwid-perfctr -C 0-7 -g MEM ./vecadd
```

Binding threads close to the memory they touch avoids cross-socket traffic — often a larger effect than any source change. This is the measurement behind the `OMP_PROC_BIND=close` hint from [Module 3](03-openmp.md).

### Instrument a region: the Marker API

To measure one region instead of the whole program, bracket it with LIKWID markers and compile with `-DLIKWID_PERFMON -llikwid`:

```c
#include <likwid-marker.h>

LIKWID_MARKER_INIT;
LIKWID_MARKER_START("triad");
for (int i = 0; i < N; i++) c[i] = a[i] + scalar*b[i];
LIKWID_MARKER_STOP("triad");
LIKWID_MARKER_CLOSE;
```

Run with `likwid-perfctr -C 0 -g MEM1 -m ./stream_triad` (`-m` enables markers). This isolates the kernel from initialization and I/O.

---

## Valgrind — Hotspots, Caches, and Correctness

Valgrind runs your program on a synthetic CPU. It is slow (10–50× slowdown) but needs no special privileges and sees everything. Three tools matter here.

### callgrind — call-graph profiling

Find which functions cost the most:

```bash
valgrind --tool=callgrind --callgrind-out-file=callgrind.out ./stream_triad
callgrind_annotate callgrind.out
```

`callgrind_annotate` prints a ranked list of functions by instruction count, with per-line annotations. For a visual call graph, open `callgrind.out` in **KCachegrind**. (When KCachegrind is unavailable, `callgrind_annotate` gives the same data as text — this is exactly the fallback noted in the repo's `notes.txt`.)

### cachegrind — cache and branch simulation

```bash
valgrind --tool=cachegrind ./stream_triad
```

cachegrind simulates the L1/LL caches and branch predictor, reporting miss rates per function and per line. It is how you confirm a suspected cache problem — e.g., that a strided access pattern or an [array-of-structs layout](01-hardware.md) is thrashing the cache.

!!! warning "Simulated, not measured"
    cachegrind models a *generic* cache, not your exact CPU, and ignores hardware prefetchers and out-of-order execution. Use it to find *relative* problems (this loop misses far more than that one), and use LIKWID's hardware counters for *absolute* numbers on real silicon.

### memcheck — memory correctness

The default tool finds leaks, invalid reads/writes, and uninitialized memory:

```bash
valgrind ./your_program          # memcheck is the default
```

Not a speed tool, but correctness bugs (a heap overrun, an uninitialized accumulator) often masquerade as performance mysteries. Run it when results are wrong or nondeterministic — exactly the symptom of the [MATAR data race](08-kokkos.md) (though races specifically need the `helgrind`/`drd` tools, not memcheck).

---

## The Roofline Model

The roofline ([introduced in Module 2](02-autovec.md)) turns your measurements into a decision. It plots attainable performance against **arithmetic intensity** — FLOPs per byte of memory traffic:

$$\text{Attainable FLOP/s} = \min\!\left(\text{Peak FLOP/s},\; \text{Peak Bandwidth} \times \text{Arithmetic Intensity}\right)$$

Two ceilings bound every kernel:

- a sloped **bandwidth ceiling** (`bandwidth × intensity`) on the left, and
- a flat **compute ceiling** (`peak FLOP/s`) on the right.

The **ridge point** where they meet is the intensity above which a kernel can be compute-bound. Plot your kernel by its intensity:

| Kernel | FLOPs | Bytes moved | Intensity | Regime |
|--------|-------|-------------|-----------|--------|
| Stream Triad | $2N$ | $24N$ | $\approx 0.08$ | bandwidth-bound |
| Dense matmul ($N{=}1024$) | $2N^3$ | $\approx 24N^2$ | $\approx N/12$ | compute-bound |

This is why the [triad barely sped up on a GPU](08-kokkos.md) while [matmul scaled dramatically](08-kokkos.md): they sit on opposite sides of the ridge. The roofline tells you which optimization is even *possible*:

- **Bandwidth-bound** (left of ridge): reduce data movement — better layout, cache blocking, fusing loops, lower precision. Adding FLOP/s does nothing.
- **Compute-bound** (right of ridge): use vector units and more cores/FPUs. Adding bandwidth does nothing.

Tools like **Intel Advisor** generate a roofline automatically by measuring FLOPs and bytes; you can also build one by hand from LIKWID's `FLOPS_DP` and `MEM` groups.

---

## Assignment

Profile the Stream Triad from Module 2 and locate it on the roofline.

1. **Topology.** Run `likwid-topology -g`. Record your L1/L2/L3 sizes and physical core count. Is `STREAM_ARRAY_SIZE` (800000 doubles ≈ 6.4 MB) larger than your last-level cache? What does that imply about where the data lives?

2. **Bandwidth.** Build the benchmark and run:
   ```bash
   likwid-perfctr -C 0 -g MEM1 ./stream_triad
   ```
   Record the achieved bandwidth. Compute the triad's bandwidth by hand — $3 \times 800000 \times 8$ bytes per iteration divided by the per-iteration time — and compare.

3. **Vectorization effect.** Rebuild with the GCC vectorization flags enabled (from Module 2) and re-measure. Did achieved bandwidth move toward peak?

4. **Call graph.** Run under callgrind and confirm the triad loop dominates the instruction count:
   ```bash
   valgrind --tool=callgrind --callgrind-out-file=callgrind.out ./stream_triad
   callgrind_annotate callgrind.out
   ```

5. **Roofline.** Compute the triad's arithmetic intensity (≈ 0.08 FLOP/byte) and your machine's ridge point (peak FLOP/s ÷ peak bandwidth). Confirm the triad falls in the bandwidth-bound region, and explain why no amount of vectorization will push it past the bandwidth ceiling.

!!! tip "Running LIKWID under SLURM"
    Hardware counters need access that batch nodes sometimes restrict. The simplest approach is an interactive allocation: `salloc -N1 -n1 --exclusive`, then run the `likwid-perfctr` commands on the granted node. Request the node exclusively so no other job perturbs the counters.

---

## Discussion Questions

1. You profile a program and find it spends 70% of its time in function `A` and 30% in `B`. You can make `A` 2× faster or `B` 10× faster. Which gives the larger speedup? Work it out with Amdahl's Law.

2. LIKWID reports your triad achieves 85% of peak memory bandwidth. Is there meaningful performance left to win on a single core? What would you measure next to decide whether to spend effort on multithreading instead?

3. cachegrind says a loop has a 40% L1 miss rate, but LIKWID's hardware counters show the loop runs near peak bandwidth anyway. How can both be true? (Hint: hardware prefetchers.)

4. A kernel sits exactly on the ridge point of your roofline. You move from single to double precision. Which way does its arithmetic intensity shift, and does that make it more compute- or more bandwidth-bound?
