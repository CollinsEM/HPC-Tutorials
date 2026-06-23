# Module 9: Profiling & Performance Analysis

## Learning Objectives

By the end of this module you will be able to:

- Apply the "measure, don't guess" discipline and target the bottleneck that matters
- Inspect a machine's topology and bind threads to specific cores
- Measure achieved memory bandwidth and cache behavior with hardware counters or timing
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

This module covers two complementary tools — [**LIKWID**](https://github.com/RRZE-HPC/likwid/wiki) for hardware-counter and bandwidth analysis, **Valgrind** for hotspots, cache simulation, and memory correctness — and the **roofline model** that ties their numbers to a decision.

!!! note "Tools used here"
    Neither LIKWID nor Valgrind is guaranteed to be installed; check before
    starting the assignment. The examples below profile `examples/autovec/stream_triad.c`
    from [Module 2](02-autovec.md).

    | Tool | Availability | Purpose |
    |------|-------------|---------|
    | LIKWID | HPC clusters via `module load likwid`; self-install on workstations | Hardware bandwidth/FLOP counters, topology, thread pinning |
    | `perf` | Ships with the Linux kernel on most distros | Hardware counters without LIKWID; needs `perf_event_paranoid ≤ 1` |
    | `lscpu` / `numactl` | Standard on nearly all Linux systems | Topology inspection |
    | `taskset` / `numactl` | Standard on nearly all Linux systems | Thread pinning |
    | Valgrind | Common on workstations; less common on clusters | Simulated cache analysis, hotspot profiling, memory correctness |
    | Timing calculation | Always available | Bandwidth from reported runtime (see [Module 2](02-autovec.md)) |

    Where hardware counters are unavailable, the timing-based bandwidth calculation is a reliable substitute for the triad.

---

## LIKWID — "Like I Knew What I'm Doing"

LIKWID is a lightweight command-line toolkit for performance measurement on CPUs. The pieces you will use most:

!!! note "If LIKWID is not installed"
    Each LIKWID capability has a widely-available substitute:

    | LIKWID command | Alternative |
    |----------------|------------|
    | `likwid-topology -g` | `lscpu` (cache sizes, cores, NUMA); `numactl --hardware` (NUMA distances) |
    | `likwid-perfctr -g MEM1` | Timing-based bandwidth (see [Module 2](02-autovec.md)); or `perf stat -e LLC-loads,LLC-load-misses ./program` |
    | `likwid-perfctr -g FLOPS_DP` | Theoretical calculation from source; or `perf stat -e fp_arith_inst_retired.scalar_double ./program` (Intel) |
    | `likwid-pin -c 0-7` | `taskset -c 0-7 ./program`; or `OMP_PLACES=cores OMP_PROC_BIND=close` for OpenMP |
    | Marker API (region measurement) | Wrap the region in your own `clock_gettime` calls |

    `perf` is part of the Linux kernel tools (`linux-tools-$(uname -r)` on Debian/Ubuntu) and is almost always available without administrator action once the `perf_event_paranoid` sysctl is set to 1 or lower (`cat /proc/sys/kernel/perf_event_paranoid`). On a cluster where you cannot change that setting, the timing-based approach requires no privileges at all.

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

Where LIKWID reads real hardware counters, **Valgrind** runs your program on a synthetic CPU — slow, but needs no privileges and sees everything. It complements LIKWID along two axes: `callgrind` finds hotspots by instruction count, `cachegrind` simulates cache and branch behavior, and `memcheck`/`helgrind` catch the memory and threading bugs that often masquerade as performance mysteries (a heap overrun, an uninitialized accumulator, or the nondeterministic results of a data race like the [MATAR `FOR_ALL`](08-kokkos.md)).

Valgrind has enough depth — and enough day-to-day importance for debugging — to warrant its own module: see **[Module 10: Valgrind](10-valgrind.md)** for the full treatment of memcheck, helgrind/DRD, cachegrind, callgrind, and massif.

!!! note "CPU tools only"
    LIKWID and Valgrind both analyze the *host CPU*. They cannot see inside a GPU kernel — for CUDA profiling use the Nsight tools in **[Module 11](11-nsight.md)**.

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

Profile the Stream Triad from [Module 2](02-autovec.md) and locate it on the roofline.

1. **Topology.** Record your L1/L2/L3 sizes and physical core count. Is `STREAM_ARRAY_SIZE` (800000 doubles ≈ 6.4 MB) larger than your last-level cache? What does that imply about where the data lives?
   ```bash
   lscpu                    # always available
   likwid-topology -g       # richer view (if LIKWID is installed)
   numactl --hardware       # NUMA distances (if numactl is installed)
   ```

2. **Bandwidth.** Build the benchmark and measure achieved bandwidth. Compute the bandwidth from the printed runtime — $3 \times 800000 \times 8$ bytes per iteration divided by the per-iteration time in seconds. If LIKWID is available, cross-check with hardware counters:
   ```bash
   # Timing-based (always available — compute from the printed runtime)
   ./stream_triad

   # Hardware counters via LIKWID (if installed):
   likwid-perfctr -C 0 -g MEM1 ./stream_triad

   # Hardware counters via perf (if perf_event_paranoid ≤ 1):
   perf stat -e LLC-loads,LLC-load-misses,cache-misses taskset -c 0 ./stream_triad
   ```

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

??? "Show solutions"
    **Question 1**: Let total runtime = 1.0 (normalized). Function $A$ = 0.70,
    function $B$ = 0.30.

    - *Make $A$ 2×*: new time $= 0.70/2 + 0.30 = 0.35 + 0.30 = 0.65$.
      Speedup $= 1/0.65 \approx 1.54\times$.
    - *Make $B$ 10×*: new time $= 0.70 + 0.30/10 = 0.70 + 0.03 = 0.73$.
      Speedup $= 1/0.73 \approx 1.37\times$.

    Making $A$ 2× gives the larger speedup. The counter-intuitive lesson is that
    **the dominant fraction matters more than the improvement factor**. A modest
    2× on the 70% hotspot beats a dramatic 10× on the 30% minor function.
    In Amdahl's form: $\text{Speedup} = 1 / ((1 - f) + f/k)$, where $f$ is the
    improved fraction and $k$ is the speedup factor within it.

    **Question 2**: At 85% of peak bandwidth there is only 15% headroom left on
    a single core. The next step is to measure bandwidth with increasing thread counts
    and find the point at which it saturates. With LIKWID: `likwid-perfctr -C 0-7 -g MEM1 ./vecadd`.
    Without LIKWID: set `OMP_NUM_THREADS=N` and compute bandwidth from the printed
    runtime at each $N$. If even 4 threads saturate the memory controller, adding
    cores helps only while you are below that saturation point — beyond it, you have
    reached the hardware ceiling and multithreading adds no more throughput. If
    bandwidth is still climbing with thread count, more threads are the right lever.

    **Question 3**: Both observations can be simultaneously true because the
    hardware prefetcher hides L1 miss latency for regular (stride-1 or stride-$k$)
    access patterns. The prefetcher issues read-ahead requests before the CPU
    actually demands the data. By the time a thread reaches a cache line, it is
    already in L2 or L3, so the miss is *counted* but not *stalled*. LIKWID's
    bandwidth counter measures actual bytes crossing the memory bus, which is near
    peak because the prefetcher keeps it busy. cachegrind's simulator counts
    misses but does not model whether the miss was hiding behind in-flight
    prefetches or causing a real stall.

    **Question 4**: Moving from FP32 to FP64 doubles the bytes per element (4 → 8)
    while leaving FLOPs per element unchanged. Arithmetic intensity = FLOPs / bytes
    → intensity **halves**, shifting the kernel **left** on the roofline into the
    bandwidth-bound region. Additionally, most consumer GPUs have 1/32 to 1/64 of
    the FP64 throughput they have for FP32, which also lowers the compute ceiling —
    the ridge point itself moves left. A kernel at the ridge point in FP32 can
    become deeply bandwidth-bound in FP64 from both effects simultaneously.

---

## References

### Reference materials

- [LIKWID](https://github.com/RRZE-HPC/likwid/wiki) — hardware-counter, topology, and thread-affinity tools.
- [perf wiki](https://perf.wiki.kernel.org/) — Linux kernel performance counters (`perf` command); covers `perf stat`, `perf record`, and `perf report`.
- [Roofline model](https://doi.org/10.1145/1498765.1498785) — Williams, Waterman & Patterson, *Communications of the ACM*, 2009.

For memory and threading correctness see [Module 10](10-valgrind.md) (Valgrind); for GPU kernel profiling see [Module 11](11-nsight.md) (Nsight).
