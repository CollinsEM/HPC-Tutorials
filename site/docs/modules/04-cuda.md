# Module 4: CUDA GPU Programming

## Learning Objectives

By the end of this module you will be able to:

- Explain the host/device memory model and why explicit data transfers exist
- Write a CUDA kernel and configure its grid and block dimensions
- Allocate GPU memory with `cudaMalloc` and transfer data with `cudaMemcpy`
- Check CUDA API error codes and kernel launch errors
- Measure kernel execution time separately from data transfer time
- Compute effective memory bandwidth and compare to GPU specs

---

## 1. GPU Execution Model

A GPU consists of thousands of simple cores organized into *Streaming Multiprocessors*
(SMs). Unlike a CPU, which has a small number of powerful, latency-optimized cores,
a GPU is designed to hide memory latency by switching between thousands of in-flight
threads. The hardware scheduler swaps a thread group out while it waits for memory
and immediately runs another.

When you launch a CUDA *kernel*, you specify two dimensions:

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

???+ success "Show complete solution"
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
    once, the GPU may not be worth the transfer overhead.

### Discussion Questions

1. The Makefile rounds `gridsize` up so every array element is covered. What would
   happen if `gridsize` were too small — say, `n / blocksize` without rounding up?

2. How does block size affect occupancy and performance? Try 128, 256, 512, and 1024
   threads per block and record the kernel time for each.

3. Data transfer time dominates total runtime. Propose a strategy to reduce it.
   (Hint: think about which arrays actually change between loop iterations.)
