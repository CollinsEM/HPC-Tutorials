# Module 11: GPU Profiling with Nsight

## Learning Objectives

By the end of this module you will be able to:

- Explain why CPU profilers cannot analyze GPU kernels
- Use **Nsight Systems** to see the system-wide timeline and the kernel-vs-transfer split
- Use **Nsight Compute** to analyze a single kernel's occupancy and memory throughput
- Read a GPU roofline and classify a kernel as memory- or compute-bound
- Profile GPU code under SLURM and handle counter-permission issues

---

## Why GPU Needs Its Own Profilers

LIKWID reads CPU hardware counters; Valgrind instruments CPU instructions. Neither can see inside a CUDA kernel — the GPU is a separate processor with its own memory, schedulers, and counters. To profile GPU code you need GPU-aware tools. NVIDIA's current suite is **Nsight**, which replaced the legacy `nvprof`/`nvvp` tools (deprecated on Volta and newer):

| Tool | Scope | Question it answers |
|------|-------|---------------------|
| [**Nsight Systems**](https://docs.nvidia.com/nsight-systems/) (`nsys`) | Whole application timeline | *Where does wall-clock time go?* Kernels, memcpys, API calls, gaps, CPU↔GPU overlap |
| [**Nsight Compute**](https://docs.nvidia.com/nsight-compute/) (`ncu`) | One kernel, in depth | *Why is this kernel slow?* Occupancy, memory throughput, instruction mix, roofline |

The workflow is top-down: **Nsight Systems first** to find which kernel (or which data transfer) dominates, then **Nsight Compute** to dissect that specific kernel. Profiling everything at the `ncu` level first is a common waste of time.

!!! note "Build for profiling"
    Compile with `-lineinfo` (nvcc) so the profilers can correlate metrics back to source lines, and profile an optimized (release) build. `-G` (full device debug) disables optimizations and distorts timing — use it for `cuda-gdb`, not for profiling.

---

## Nsight Systems — The Timeline

Nsight Systems traces the whole run: CUDA API calls, kernel launches, host↔device memory copies, and CPU activity on one timeline. It is low-overhead and the right first look.

```bash
cd examples/CUDA/Example1
make
nsys profile --stats=true -o triad_report ./stream_triad
```

- `--stats=true` prints summary tables to the terminal after profiling.
- `-o triad_report` writes `triad_report.nsys-rep`, which opens in the Nsight Systems GUI for the visual timeline.

For a terminal-only summary of an existing report:

```bash
nsys stats triad_report.nsys-rep
```

### What to look for

The summary breaks time down by category. For the Stream Triad you will see something striking — the same lesson [Module 4](04-cuda.md) reported numerically:

- **`cudaMemcpy` (HtoD/DtoH) dominates.** Moving the three 80 M-element arrays over PCIe dwarfs the kernel itself. The timeline shows long memcpy bars bracketing a thin kernel bar.
- **The kernel is tiny.** `StreamTriad` runs in a fraction of the transfer time.
- **Gaps and serialization.** Look for the GPU sitting idle while the CPU prepares the next launch, or transfers that could overlap with compute but do not.

This view is what turns "the GPU version isn't faster" into a concrete diagnosis: *you are transfer-bound, not compute-bound — fix the data movement, not the kernel.*

---

## Nsight Compute — Inside a Kernel

Once Nsight Systems identifies the kernel worth optimizing, Nsight Compute profiles it in depth. It replays the kernel many times to collect hardware counters, so it is slow — always target a specific kernel and limit launches.

```bash
ncu --set full -k StreamTriad -c 1 -o triad_kernel ./stream_triad
```

- `--set full` collects the complete metric set (use `--set basic` for a fast first pass).
- `-k StreamTriad` profiles only kernels whose name matches.
- `-c 1` profiles a single launch (the benchmark runs 16; one is enough).
- `-o triad_kernel` writes `triad_kernel.ncu-rep` for the `ncu-ui` GUI; omit `-o` to print to the terminal.

### Key sections to read

- **GPU Speed Of Light** — the headline: achieved percentage of peak *compute* and peak *memory* throughput. For the triad, memory will be high and compute low — the signature of a memory-bound kernel.
- **Memory Workload Analysis** — achieved DRAM bandwidth in GB/s; compare to the GPU's spec (e.g., ~900 GB/s on a V100, ~2 TB/s on an A100/H100). Near-peak here means the kernel is doing as well as the hardware allows.
- **Occupancy** — active warps versus the theoretical maximum. Low occupancy can hide memory latency poorly; but a memory-bound kernel at peak bandwidth is already optimal regardless of occupancy.
- **Roofline** — Nsight Compute plots the kernel on the GPU's roofline automatically, showing how close it sits to the bandwidth ceiling.

### The GPU roofline

The [roofline model](09-profiling.md) applies to GPUs with the GPU's own ceilings. The Stream Triad's arithmetic intensity (~0.08 FLOP/byte) places it far left, pinned to the bandwidth ceiling — so the only way to go faster is to move less data or use faster memory, never more FLOP/s. Contrast the [Kokkos matmul](08-kokkos.md), whose high intensity puts it near the compute ceiling where the GPU's thousands of FPUs dominate.

---

## Profiling Under SLURM

GPU profiling needs a GPU node, and Nsight Compute needs access to GPU performance counters — which clusters often restrict.

=== "Interactive (recommended)"
    ```bash
    salloc -N1 -n1 --gres=gpu:1 --time=00:30:00
    # on the granted node:
    module load cuda/12.0
    nsys profile --stats=true -o report ./stream_triad
    ncu --set basic -k StreamTriad -c 1 ./stream_triad
    ```

=== "Batch"
    Add the profiling command to a job script (model it on `examples/CUDA/Example1/run.slurm`), writing the report to a file you open later:
    ```bash
    nsys profile -o $SLURM_SUBMIT_DIR/report_%j ./stream_triad
    ```

!!! warning "ERR_NVGPUCTRPERM — counter permissions"
    Nsight Compute often fails for non-root users with *"The user does not have permission to access NVIDIA GPU Performance Counters."* This is a driver-level restriction (`NVreg_RestrictProfilingToAdminUsers`). The fix requires an administrator to relax it. Nsight **Systems** does *not* need these counters, so timeline profiling usually works even when `ncu` does not — another reason to start there.

---

## Assignment

1. **Timeline.** Build the CUDA Stream Triad and run `nsys profile --stats=true`. From the summary, record the total time in kernels versus in `cudaMemcpy`. What fraction is data transfer? Does it match the kernel-vs-transfer numbers the program prints itself (from [Module 4](04-cuda.md))?

2. **Kernel.** Run `ncu --set basic -k StreamTriad -c 1`. Read the GPU Speed Of Light section: what percentage of peak memory throughput does the kernel achieve, and what percentage of peak compute? Which one is the limiter?

3. **Bandwidth vs spec.** From Nsight Compute's Memory Workload Analysis, record the achieved DRAM bandwidth and compare it to your GPU's rated bandwidth. How close to peak is the triad?

4. **Roofline.** Locate the kernel on Nsight Compute's roofline chart. Confirm it sits on the bandwidth ceiling, and explain why increasing the block size or thread count will not move it up.

5. **(If `ncu` is blocked)** If you hit `ERR_NVGPUCTRPERM`, complete the analysis with Nsight Systems alone: what can you still conclude about the kernel-vs-transfer balance without kernel-level counters?

---

## Discussion Questions

1. Nsight Systems shows data transfer taking 20× the kernel time for the Stream Triad. List two concrete code changes that would cut the *transfer* cost, and explain why optimizing the kernel itself would be wasted effort here.

2. Nsight Compute reports the triad kernel at 90% of peak memory throughput but only 3% of peak compute. Is this kernel well-optimized or poorly optimized? Defend your answer using the roofline.

3. Why is the top-down order — Nsight Systems before Nsight Compute — more efficient than profiling every kernel with `ncu` from the start?

4. Occupancy for the triad kernel is moderate, yet it achieves near-peak bandwidth. Why is chasing higher occupancy pointless for this kernel, and for what kind of kernel *would* raising occupancy help?

---

## References

### Reference materials

- [Nsight Systems](https://docs.nvidia.com/nsight-systems/) — system-wide timeline profiler ([User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)).
- [Nsight Compute](https://docs.nvidia.com/nsight-compute/) — single-kernel profiler ([CLI reference](https://docs.nvidia.com/nsight-compute/NsightComputeCli/), [Kernel Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)).
