# Module 0: Parallel Computing Theory

## Learning Objectives

By the end of this module you will be able to:

- Explain why single-thread CPU performance plateaued around 2005
- Apply Amdahl's Law to predict strong-scaling limits
- Apply Gustafson-Barsis's Law to predict weak-scaling behavior
- Classify hardware and algorithms using Flynn's taxonomy
- Identify when a GPU is and is not the right tool for a problem
- Explain the performance portability problem and why abstraction layers exist

---

## Why Parallelism Matters

### The End of "Free" Performance Gains

For roughly three decades, programmers could rely on Moore's Law to deliver faster
hardware without changing their code. Clock frequencies doubled roughly every two
years; a program that ran in 10 seconds today ran in 5 seconds on next year's CPU.

That era ended around 2005. Clock frequency, single-thread performance, and power
consumption all plateaued simultaneously. The reasons are coupled:

- **Power wall**: Power dissipation scales as $P \propto C V^2 f$, where $C$ is
  capacitance, $V$ is supply voltage, and $f$ is clock frequency. Shrinking
  transistors lowers $C$ but not $V$ proportionally; attempting to raise $f$
  sends power density to levels that cannot be cooled in commodity packaging.
- **ILP wall**: Instruction-level parallelism (out-of-order execution, branch
  prediction, speculative execution) has diminishing returns; modern CPUs already
  exploit most of the available ILP in typical code.
- **Memory wall**: Memory latency has not kept pace with compute throughput,
  so a faster ALU often just stalls on data.

What *has* continued to grow: core counts. Modern server CPUs have 32–96 cores.
Modern GPUs have thousands of simple compute units. The hardware is parallel;
serial code cannot use it.

### The Cost of Writing Serial Code

Consider a single core on a modern 16-core CPU with hyperthreading and 256-bit
AVX vector units processing 64-bit doubles:

$$\text{Parallelism} = \underbrace{16}_{\text{cores}} \times \underbrace{2}_{\text{hyperthreads}} \times \underbrace{\frac{256\text{ bits}}{64\text{ bits}}}_{\text{vector width}} = 128\text{-way}$$

A serial scalar program exploits exactly 1 of those 128 lanes, achieving roughly
$1/128 \approx 0.8\%$ of available floating-point throughput — before even
considering that multiple sockets or GPU accelerators may be available.

### Energy Cost of Under-Utilization

Hardware efficiency translates directly to energy bills. As a concrete example:

| Configuration | Hardware | Count | TDP | Runtime | Energy |
|---|---|---|---|---|---|
| CPU-only | Intel Xeon E5-4660 (16-core) | 20 nodes | 120 W | 24 h | 57.6 kWh |
| GPU-accelerated | NVIDIA Tesla V100 | 4 GPUs | 300 W | 24 h | 28.8 kWh |

If the GPU configuration solves the same problem in the same wall-clock time using
half the energy, the choice is clear — provided the code is structured to exploit
GPU parallelism effectively.

!!! note
    These numbers assume the GPU run achieves the same time-to-solution as the
    CPU cluster. Achieving that requires both algorithmic suitability and careful
    implementation. The rest of this tutorial is about acquiring those skills.

---

## Amdahl's Law: Strong Scaling

### The Formula

Amdahl's Law describes *strong scaling*: what happens when you add processors to
solve a **fixed-size** problem. Let $S$ be the serial fraction of the work,
$P = 1 - S$ the parallelizable fraction, and $N$ the number of processors. The
maximum achievable speedup is:

$$\text{SpeedUp}(N) = \frac{1}{S + \dfrac{P}{N}}$$

### Implications

The serial fraction $S$ is an absolute ceiling on scalability. As $N \to \infty$:

$$\lim_{N \to \infty} \text{SpeedUp}(N) = \frac{1}{S}$$

| Serial fraction $S$ | Max speedup (any $N$) |
|---|---|
| 50% | 2× |
| 10% | 10× |
| 5% | 20× |
| 1% | 100× |

A program with 10% serial code can never exceed 10× speedup, no matter how many
processors you throw at it. Parallelizing 90% of the code leaves you one decimal
place of headroom.

!!! warning
    Amdahl's Law assumes the serial fraction $S$ is constant and cannot be
    parallelized. In practice, synchronization, I/O, initialization, and output
    all contribute to $S$. These costs sometimes *grow* with $N$ (due to
    communication overhead), making real scaling worse than Amdahl predicts.

### Strong-Scaling Curve

The speedup curve is concave and saturates rapidly:

- At $N = 10$ with $S = 0.1$: $\text{SpeedUp} = 1/(0.1 + 0.9/10) = 5.26\times$
- At $N = 100$ with $S = 0.1$: $\text{SpeedUp} = 1/(0.1 + 0.9/100) = 9.17\times$
- At $N = 1000$ with $S = 0.1$: $\text{SpeedUp} = 9.91\times$

The marginal return of adding processors falls off quickly. Doubling from 100 to
200 processors adds less than 0.5× speedup when $S = 0.1$.

---

## Gustafson-Barsis's Law: Weak Scaling

### The Insight

Amdahl's Law holds the problem size fixed. Gustafson and Barsis observed in 1988
that this is often the wrong model: in practice, scientists *choose problem size
based on available hardware*. With more processors, you run a bigger problem in
the same wall time — not the same problem faster.

Under weak scaling, the total work grows with $N$ while wall time remains constant.
If the serial work $s$ is fixed (does not grow with $N$), the scaled speedup is:

$$\text{SpeedUp}(N) = N - S(N - 1)$$

where $S$ is the serial fraction measured *on a single processor*. Rewritten:
$S \cdot 1 + P \cdot N$ units of work complete in the same time as $1$ unit on a
single processor — a speedup of $S + P \cdot N = N - S(N-1)$.

### Amdahl vs Gustafson-Barsis

| | Amdahl (Strong Scaling) | Gustafson-Barsis (Weak Scaling) |
|---|---|---|
| Problem size | Fixed | Grows with $N$ |
| What's measured | Speedup for same problem | Work done in same time |
| Serial bottleneck | Hard ceiling $1/S$ | Scales as $N - S(N-1)$ |
| Practical relevance | Memory-constrained runs | Production HPC simulations |

!!! tip
    When reporting scaling results, always state whether you ran a strong or weak
    scaling study. The two tell fundamentally different stories about your code's
    parallel efficiency.

---

## Flynn's Taxonomy

Michael Flynn's 1972 classification scheme categorizes computer architectures
along two axes: instruction stream and data stream, each either Single or Multiple.

| | Single Data | Multiple Data |
|---|---|---|
| **Single Instruction** | SISD | SIMD |
| **Multiple Instruction** | MISD | MIMD |

### SISD — Single Instruction, Single Data

A scalar serial processor. One instruction operates on one data element at a time.
A for-loop executing one iteration per cycle on a single core is SISD. This is the
model most programmers have in their heads.

### SIMD — Single Instruction, Multiple Data

One instruction operates on multiple data elements simultaneously using vector
registers. Modern CPUs expose this via AVX/AVX-512 intrinsics or auto-vectorization.
GPU warps (groups of 32 CUDA threads executing the same instruction) are also
SIMD at the warp level.

SIMD is the mechanism behind statements like "256-bit vector units process 4
doubles at once." The hardware broadcasts a single instruction to multiple ALUs
operating on adjacent memory.

### MISD — Multiple Instruction, Single Data

Uncommon in practice. Fault-tolerant systems that run the same data through
independent pipelines to cross-check results (e.g., flight-control computers) fit
this category.

### MIMD — Multiple Instruction, Multiple Data

Each processor runs its own instruction stream on its own data. This is the model
for multi-core CPUs and MPI-distributed clusters. OpenMP threads are MIMD;
MPI ranks are MIMD. Real workloads often combine MIMD at the node level with
SIMD at the core level.

!!! note
    Modern HPC codes often exploit all four levels of the hierarchy simultaneously:
    SIMD vector units inside MIMD OpenMP threads inside MIMD MPI ranks. Each level
    requires a different programming model and a different set of correctness
    guarantees.

---

## CPU vs GPU Architecture

### Design Philosophy

CPUs and GPUs reflect opposite design choices about the compute-latency tradeoff.

| Feature | CPU | GPU |
|---|---|---|
| Core count | 8–96 high-complexity cores | Thousands of simple shader/CUDA cores |
| Clock speed | 3–5 GHz | 1–2 GHz |
| Cache depth | L1 → L2 → L3 → DRAM | L1 → L2 → HBM/GDDR |
| Cache size (L3) | 16–256 MiB | 32–40 MiB (shared L2) |
| Branch prediction | Deep, speculative | Minimal (branch divergence is expensive) |
| Memory bandwidth | 50–200 GB/s (DDR5) | 900–3000 GB/s (HBM3) |
| Design target | Low latency, diverse workloads | High throughput, regular workloads |

A CPU minimizes latency for any single thread using deep caches and sophisticated
out-of-order execution. A GPU maximizes aggregate throughput by hiding latency:
when one warp stalls on a memory load, the hardware instantly switches to another
warp. This only works when there are thousands of independent warps in flight.

### When a GPU Is a Good Fit

A workload maps well to a GPU when:

- **Loops are order-independent**: any iteration can execute in any order without
  affecting correctness (no loop-carried dependencies).
- **Threads are independent and thread-safe**: no shared mutable state that
  requires synchronization across threads.
- **Memory access is contiguous**: threads in a warp access adjacent addresses
  (coalesced access). Random access causes one memory transaction per thread
  rather than one per warp.
- **Thread divergence is low**: all threads in a warp follow the same control
  flow path. Divergent branches serialize within the warp.
- **Arithmetic intensity is moderate to high**: enough floating-point work per
  byte loaded to keep the ALUs busy while data is in flight. Pure memory-copy
  kernels saturate bandwidth but do not stress compute.

### When a GPU Is a Poor Fit

| Pattern | Problem |
|---|---|
| Tree traversal / oct-trees / kd-trees | Irregular memory access; poor coalescing |
| Dynamic memory allocation in kernels | `malloc`/`new` on device is expensive and serializing |
| Monte Carlo with divergent paths | Thread divergence serializes warp execution |
| Globally implicit solvers (multi-GPU) | All-to-all communication across GPUs is slow |
| Small datasets | PCIe transfer overhead dominates compute time |

!!! warning
    Moving data to and from the GPU over PCIe costs roughly 10–15 GB/s. If your
    kernel takes less time than the data transfer, you have made the code slower,
    not faster. Profile the transfer cost before porting to GPU.

---

## The Performance Portability Problem

Writing GPU code in CUDA produces code that runs only on NVIDIA hardware. HIP
targets AMD GPUs. SYCL and oneAPI target Intel GPUs and CPUs. OpenMP offload
supports multiple vendors but with varying compiler maturity.

This creates a maintenance problem: to support multiple hardware targets, teams
historically maintained multiple codebases. Bugs fixed in one port were re-introduced
in another.

**Abstraction layers** solve this by providing a single source that compiles to
different backends:

| Framework | Backends supported |
|---|---|
| Kokkos | CUDA, HIP, SYCL, OpenMP, serial |
| RAJA | CUDA, HIP, OpenMP, sequential |
| OpenMP (target offload) | NVIDIA, AMD, Intel (compiler-dependent) |
| SYCL / DPC++ | Intel, NVIDIA (via Codeplay), AMD |

The tradeoff: abstraction layers impose a compilation and conceptual overhead. Code
written to Kokkos idioms is portable but requires the Kokkos runtime to be installed
and the developer to understand Kokkos's execution and memory space model.
[Module 8](08-kokkos.md) works through a Kokkos example to demonstrate the tradeoff.

---

## Discussion Questions

1. Your application has a serial fraction of 5%. What is the maximum achievable
   speedup with unlimited processors? At what processor count do you reach 90% of
   that maximum?

2. If doubling the problem size also doubles the parallel work but leaves the serial
   work unchanged, which scaling law is more appropriate to apply? What does that
   law predict about efficiency as $N$ grows?

3. Name two properties of your own code that would make it a strong candidate for
   GPU acceleration, and two properties that would make it a poor candidate. Be
   specific about which GPU-fit criteria from this module apply.

??? "Show solutions"
    **Question 1**: With $S = 0.05$, the maximum speedup is $1/0.05 = 20\times$.
    To reach 90% of that limit ($18\times$), solve $18 = 1/(0.05 + 0.95/N)$, giving
    $N = 0.95/(1/18 - 0.05) \approx 171$ processors. Beyond that, adding hardware
    delivers diminishing fractions of a speedup unit.

    **Question 2**: This is the Gustafson-Barsis regime. Problem size grows with
    $N$; only the serial work is fixed. The Gustafson-Barsis law predicts
    $\text{SpeedUp}(N) = N - S(N-1)$, which grows linearly in $N$ rather than
    saturating. Parallel efficiency — speedup divided by $N$ — approaches
    $(1 - S)$ as $N$ grows, not zero.

    **Question 3**: Open-ended; answers will vary. Examples of GPU-favorable
    properties: embarrassingly parallel loops with no dependencies, array-of-number
    data with stride-1 access, high arithmetic intensity (e.g., dense linear
    algebra). Examples of GPU-unfavorable properties: irregular sparse graph
    traversal, adaptive mesh refinement with frequent dynamic allocation, tightly
    coupled implicit solvers requiring global reductions at every timestep.
