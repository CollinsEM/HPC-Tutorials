# Module 4: CUDA GPU Programming

## Learning Objectives

By the end of this module you will be able to:

- Explain the host/device memory model and why explicit data transfers exist
- Write a CUDA kernel and configure its grid and block dimensions
- Allocate GPU memory with `cudaMalloc` and transfer data with `cudaMemcpy`
- Check CUDA API error codes and kernel launch errors
- Measure kernel execution time separately from data transfer time
- Compute effective memory bandwidth and compare to GPU specs
- Map threads to a 2D problem with 2D grids and blocks
- Contrast a compute-bound kernel (matrix multiply) with a bandwidth-bound one (Stream Triad) on the roofline

---

## 1. GPU Execution Model

A GPU consists of thousands of simple cores organized into *Streaming Multiprocessors*
(SMs). Unlike a CPU, which has a small number of powerful, latency-optimized cores,
a GPU is designed to hide memory latency by switching between thousands of in-flight
threads. The hardware scheduler swaps a thread group out while it waits for memory
and immediately runs another.

When you launch a [CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) *kernel*, you specify two dimensions:

- A **grid** — the number of thread blocks to launch
- A **thread block** — the number of threads within each block

Every thread executes the same kernel function but with a unique index, making
CUDA a SIMT (Single Instruction, Multiple Thread) model — the GPU analogue of SIMD.

### Thread Indexing (1D case)

```
global_thread_index = blockIdx.x * blockDim.x + threadIdx.x
```

CUDA provides built-in variables for each thread:

| Variable | Meaning |
|----------|---------|
| `threadIdx.x` | Thread index within its block (0 to `blockDim.x - 1`) |
| `blockIdx.x` | Block index within the grid |
| `blockDim.x` | Number of threads per block |
| `gridDim.x` | Number of blocks in the grid |

Block size must be a multiple of 32 (the *warp* size — the hardware execution unit
that schedules 32 threads together). A common choice is 256 or 512 threads per block.
Grid size is computed to cover the array:

```c
int blocksize = 512;
int gridsize = (n + blocksize - 1) / blocksize;  // round up
```

!!! note "Bounds check is mandatory"
    When `n` is not evenly divisible by `blocksize`, the last block contains threads
    whose computed index exceeds the array bounds. Always guard with `if (i >= n) return;`
    at the start of the kernel before any array access.

---

## 2. Host and Device Memory

CPUs and GPUs have **separate memory spaces**. The GPU has its own DRAM (device
memory) connected via a high-bandwidth bus (hundreds of GB/s for HBM, 500–1000 GB/s
on modern data-center GPUs). CPU-to-GPU transfers go over PCIe (~16 GB/s) — a
significant bottleneck for small or frequently-transferred arrays.

The standard workflow for every GPU computation:

1. Allocate host (CPU) arrays with `malloc`
2. Allocate device (GPU) arrays with `cudaMalloc`
3. Copy input data host → device with `cudaMemcpy(..., cudaMemcpyHostToDevice)`
4. Launch kernel(s)
5. `cudaDeviceSynchronize()` — wait for the GPU to finish
6. Copy results device → host with `cudaMemcpy(..., cudaMemcpyDeviceToHost)`
7. `cudaFree` device arrays; `free` host arrays

Data transfer is often the dominant cost for memory-bound kernels. The Stream Triad
benchmark used here reports kernel time and total time (kernel + transfers) separately,
so you can quantify this overhead directly.

!!! tip "Naming convention"
    A common convention is to suffix device pointers with `_d` (e.g., `a_d`, `b_d`)
    to distinguish them from host pointers. Dereferencing a device pointer on the
    host (or vice versa) causes a segfault or silent memory corruption.

---

## 3. The CUDA Stream Triad Kernel

The Stream Triad computes:

$$c_i = a_i + s \cdot b_i \quad \forall \, i \in [0, N)$$

where $s$ is a scalar constant. The kernel touches each array element exactly once —
no reuse, no reduction — making it a pure **memory bandwidth** benchmark. The
effective bandwidth determines how fast the GPU can move data from DRAM to registers.

Below is the starter code from `examples/CUDA/Example1/StreamTriad.cu`. Locations
marked `TODO` are left for you to complete.

```cuda
__global__ void StreamTriad(
    const int n,
    const double scalar,
    const double *a,
    const double *b,
    double *c)
{
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= n) return;

  // TODO: c[i] = a[i] + scalar*b[i]
}

int main(int argc, char *argv[]){
  const int stream_array_size = 80000000;  // 80M elements
  double scalar = 3.0;

  double *a = (double *)malloc(stream_array_size*sizeof(double));
  double *b = (double *)malloc(stream_array_size*sizeof(double));
  double *c = (double *)malloc(stream_array_size*sizeof(double));

  for (int i=0; i<stream_array_size; i++) {
    a[i] = 1.0;  b[i] = 2.0;  c[i] = 0.0;
  }

  cudaError_t err;

  // TODO: Allocate a_d, b_d, c_d on the device with cudaMalloc
  double *a_d = nullptr, *b_d = nullptr, *c_d = nullptr;

  const int blocksize = 512;
  const int gridsize = (stream_array_size + blocksize - 1)/blocksize;

  for (int k = 0; k < NTIMES; k++){
    // TODO: Copy a and b to device (cudaMemcpyHostToDevice)
    cudaDeviceSynchronize();

    // TODO: Launch StreamTriad<<<gridsize, blocksize>>>(...)

    cudaDeviceSynchronize();

    // TODO: Copy c back to host (cudaMemcpyDeviceToHost)
  }

  if (a_d) cudaFree(a_d);
  if (b_d) cudaFree(b_d);
  if (c_d) cudaFree(c_d);
  free(a); free(b); free(c);
}
```

---

## 4. Assignment

Work through the TODOs in order. Build and test after each step to catch errors early.

### Step 1 — Complete the kernel

Inside `StreamTriad`, add the triad computation for thread `i`:

```cuda
c[i] = a[i] + scalar*b[i];
```

This is the only line inside the kernel body after the bounds check.

### Step 2 — Allocate device memory

After the host `malloc` calls, allocate three device arrays. Check every
`cudaMalloc` return value:

```c
err = cudaMalloc(&a_d, stream_array_size*sizeof(double));
if (err != cudaSuccess) {
  printf("Error allocating a_d: %s\n", cudaGetErrorString(err));
  return -1;
}
```

Repeat for `b_d` and `c_d`.

!!! warning "Always check cudaMalloc"
    GPU memory is limited. `cudaMalloc` silently succeeds even on some error paths
    on older drivers. Always check the return code and print the error string with
    `cudaGetErrorString(err)` — it gives a human-readable description.

### Step 3 — Copy host → device

Inside the benchmark loop, before the first `cudaDeviceSynchronize()`:

```c
cudaMemcpy(a_d, a, stream_array_size*sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(b_d, b, stream_array_size*sizeof(double), cudaMemcpyHostToDevice);
```

### Step 4 — Launch the kernel

After the first `cudaDeviceSynchronize()`:

```c
StreamTriad<<<gridsize, blocksize>>>(stream_array_size, scalar, a_d, b_d, c_d);
err = cudaGetLastError();
if (err != cudaSuccess) {
  printf("Kernel error: %s\n", cudaGetErrorString(err));
  return -1;
}
```

!!! note "Kernel launch errors are asynchronous"
    `cudaGetLastError()` catches configuration errors (invalid grid/block size,
    too much shared memory). Errors inside the kernel body (e.g., an out-of-bounds
    access causing a segfault) are not reported until the next synchronization point.
    Use `cuda-memcheck` or `compute-sanitizer` to catch runtime kernel errors.

### Step 5 — Copy device → host

After the second `cudaDeviceSynchronize()`:

```c
cudaMemcpy(c, c_d, stream_array_size*sizeof(double), cudaMemcpyDeviceToHost);
```

??? success "Show complete solution"
    The complete solution is at `examples/CUDA/Example1/solutions/StreamTriad.cu`.
    Key additions relative to the starter code:

    ```cuda
    // In the kernel body (after the bounds check):
    c[i] = a[i] + scalar*b[i];

    // Device allocation (repeat for b_d, c_d):
    err = cudaMalloc(&a_d, stream_array_size*sizeof(double));
    if (err != cudaSuccess) {
      printf("Error allocating a_d: %s\n", cudaGetErrorString(err));
      return -1;
    }

    // Host-to-device copies (inside the loop, before first sync):
    cudaMemcpy(a_d, a, stream_array_size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, stream_array_size*sizeof(double), cudaMemcpyHostToDevice);

    // Kernel launch (after first sync):
    StreamTriad<<<gridsize, blocksize>>>(stream_array_size, scalar, a_d, b_d, c_d);

    // Device-to-host copy (after second sync):
    cudaMemcpy(c, c_d, stream_array_size*sizeof(double), cudaMemcpyDeviceToHost);
    ```

---

## 5. Build and Run

=== "Workstation"
    The `Makefile` auto-detects your GPU's compute capability via `nvidia-smi`:

    ```bash
    cd examples/CUDA/Example1
    make
    ./stream_triad
    ```

    You need the CUDA Toolkit installed (`nvcc` on PATH). Verify with `nvidia-smi`.

=== "SLURM cluster"
    See `examples/CUDA/Example1/run.slurm`. Request a GPU node with `--gres=gpu:1`
    and load the appropriate CUDA module. Adjust the module name for your cluster's
    software stack.

---

## 6. Expected Output and Analysis

A successful run prints something like:

```
Average runtime is 1.234 msecs  data transfer is 45.678 msecs
```

### Memory Bandwidth

The Stream Triad reads arrays `a` and `b` and writes array `c` — three memory
operations per element. Effective GPU memory bandwidth is:

$$\text{BW} = \frac{3 \times N \times 8 \;\text{bytes}}{\text{kernel time (s)}} \;\text{GB/s}$$

For 80M doubles: $3 \times 80 \times 10^6 \times 8 = 1.92$ GB moved per kernel call.
Dividing by the kernel time in seconds gives GB/s. Compare this to your GPU's rated
HBM or GDDR bandwidth (available from `nvidia-smi --query-gpu=memory.total --format=csv`
or the GPU spec sheet).

!!! note "Transfer cost dominates"
    Data transfer time is typically 20–50$\times$ the kernel time for this example.
    The kernel is extremely fast on the GPU, but moving 1.92 GB over PCIe at
    ~16 GB/s takes ~120 ms. For small datasets, or kernels that are called only
    once, the GPU may not be worth the transfer overhead. [Module 11](11-nsight.md)
    shows this same kernel-versus-transfer split on the Nsight Systems timeline.

### Discussion Questions

1. The Makefile rounds `gridsize` up so every array element is covered. What would
   happen if `gridsize` were too small — say, `n / blocksize` without rounding up?

2. How does block size affect occupancy and performance? Try 128, 256, 512, and 1024
   threads per block and record the kernel time for each.

3. Data transfer time dominates total runtime. Propose a strategy to reduce it.
   (Hint: think about which arrays actually change between loop iterations.)

---

## 7. A Compute-Bound Kernel: Matrix Multiplication

The Stream Triad sits at the far *left* of the [roofline](02-autovec.md): with an
arithmetic intensity of ~0.08 FLOP/byte it is firmly **bandwidth-bound**. Dense
**matrix multiplication** lives at the opposite end. Multiplying two $N \times N$
matrices performs $2N^3$ floating-point operations on only $3N^2$ matrix elements,
so if every element were read from memory just once the arithmetic intensity would
be

$$I = \frac{2N^3}{3N^2 \times 8 \;\text{bytes}} = \frac{N}{12} \;\text{FLOP/byte}.$$

For $N = 1024$ that is ~85 FLOP/byte — far to the *right* of the ridge point, i.e.
**compute-bound**. Matrix multiply is the canonical compute-bound kernel, and it is
the same problem the [Kokkos and MATAR modules](08-kokkos.md) solve, which makes it
a useful bridge: here you write it in raw CUDA; there you write it once and let a
portability layer target any backend.

The second example, `examples/CUDA/Example2/matrix_multiply.cu`, computes
$C = A \times B$ for $N = 1024$ and verifies the first $3\times3$ block against a CPU
reference.

### Mapping threads to a 2D problem

The triad was one-dimensional, so a 1D grid sufficed. A matrix is naturally 2D, so
this kernel uses **2D blocks and a 2D grid**: each thread computes one output
element `C[row][col]`.

```cuda
#define BLOCK_SIZE 32   // 32 x 32 = 1024 threads per block

__global__ void matrixMultiply(real_t *A, real_t *B, real_t *C, int size) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < size && col < size) {
    real_t sum = 0.0;
    for (int i = 0; i < size; i++) {
      sum += A[row * size + i] * B[i * size + col];   // row of A . column of B
    }
    C[row * size + col] = sum;
  }
}
```

The launch configuration mirrors the 1D case, rounded up in *both* dimensions:

```cuda
dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
dim3 blocksPerGrid((size + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (size + BLOCK_SIZE - 1) / BLOCK_SIZE);
matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size);
```

!!! note "High arithmetic intensity is a property of the algorithm, not this kernel"
    The $N/12$ figure assumes each matrix element is read once. This **naive** kernel
    does not achieve that: every thread re-reads an entire row of $A$ and column of
    $B$ straight from global memory, so each element of $A$ and $B$ is fetched $N$
    times. Its *effective* arithmetic intensity is closer to $1/8$ FLOP/byte —
    memory-bound in practice — which is why it reaches only a fraction of the GPU's
    peak FLOP/s. The standard fix is **shared-memory tiling**: each block cooperatively
    loads a tile of $A$ and $B$ into fast on-chip shared memory and reuses it, raising
    the realized intensity toward the algorithmic ceiling. That optimization is the
    natural next step beyond this example.

### Build and run

=== "Workstation"
    The `Makefile` auto-detects your GPU's compute capability via `nvidia-smi`:

    ```bash
    cd examples/CUDA/Example2
    make
    ./matrix_multiply
    ```

=== "SLURM cluster"
    See `examples/CUDA/Example2/run.slurm`. Request a GPU node with `--gres=gpu:1`
    and load the appropriate CUDA module for your cluster.

A successful run reports the achieved throughput and the verification block:

```
Matrix Size: 1024 x 1024
Execution Time: 7.78 ms
Performance: 275.75 GFLOPS

Verification (showing first 3x3 elements of result):
C[0][0] = 257.26 (expected: 257.26)
...
```

The reported GFLOPS will vary with your GPU. Compare it to the card's peak FP64 (or
FP32) rate: the gap between the two is the headroom that shared-memory tiling and
other reuse optimizations recover.

### Assignment

The starter `examples/CUDA/Example2/matrix_multiply.cu` compiles as-is but does no
GPU work yet; complete the `TODO`s:

1. **Kernel body** — compute `sum` as the dot product of row `row` of `A` and column
   `col` of `B`, then store it to `C[row * size + col]`.
2. **Device memory** — allocate `d_A`, `d_B`, `d_C` with `cudaMalloc`.
3. **Transfers** — copy `h_A`/`h_B` to the device; the result copy back is already in
   place.
4. **Launch configuration** — set `threadsPerBlock` and `blocksPerGrid`, then launch
   the kernel.

??? success "Show complete solution"
    The complete program is at `examples/CUDA/Example2/solution/matrix_multiply.cu`.

Extensions to explore:

- Change the `real_t` typedef from `double` to `float` and re-measure. How does the
  throughput change, and why? (Hint: consumer GPUs have far more FP32 than FP64
  units.)
- Sketch (or implement) a shared-memory tiled version and predict its effect using
  the roofline.

### Discussion Questions

1. The triad and the matrix multiply use the *same* GPU yet land on opposite sides of
   the roofline. Which hardware resource limits each, and why does adding more
   floating-point units help one but not the other?
2. This kernel uses 32&times;32 = 1024 threads per block. What constrains the maximum
   block size, and what happens to occupancy if you also need shared memory per block?
3. The naive kernel re-reads $A$ and $B$ from global memory $N$ times. Estimate the
   total global-memory traffic with and without perfect reuse, and relate the two
   numbers to the arithmetic intensities quoted above.

---

## References

### Reference materials

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — the core language and execution-model reference.
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/) — reference for the runtime calls below.
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) — performance and correctness guidance.

### Runtime API calls used in this module

The memory-management and error-handling calls used here — `cudaMalloc`, `cudaMemcpy`, `cudaFree`, `cudaGetLastError`, `cudaGetErrorString`, and `cudaDeviceSynchronize` — are documented in the [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/) reference, under its Memory Management and Error Handling sections.
