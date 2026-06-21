# Module 7: MPI — Distributed-Memory Parallelism

## Learning Objectives

By the end of this module you will be able to:

- Explain the SPMD execution model and the role of *rank*, *size*, and *communicator*
- Write, build, and launch a minimal MPI program with `mpiexec`/`srun`
- Exchange data with point-to-point `MPI_Send`/`MPI_Recv` and explain how ordering causes deadlock
- Avoid deadlock using ordered sends, `MPI_Sendrecv`, or non-blocking `MPI_Isend`/`MPI_Irecv`
- Distribute and collect data with collective operations (`MPI_Bcast`, `MPI_Scatter`, `MPI_Gather`)
- Reason about when communication cost outweighs the benefit of more processes

---

## A Different Axis of Parallelism

Every module so far has used **shared memory**: threads (OpenMP), vector lanes (SIMD), or GPU threads all read and write the *same* address space. [MPI](https://www.mpi-forum.org/docs/) is different. It is the tool for **distributed memory** — many separate processes, each with its own private memory, possibly on different physical nodes of a cluster, cooperating by *passing messages*.

This maps to the **MIMD** category of [Flynn's taxonomy](00-parallel-theory.md): Multiple Instruction, Multiple Data. It is how computation scales *beyond a single node* — past the point where shared-memory threading runs out of cores or memory.

!!! note "MPI and OpenMP are complementary, not competing"
    Large HPC codes are commonly *hybrid*: MPI between nodes, OpenMP (or CUDA) within each node. MPI handles the inter-node communication that shared-memory models cannot; threading exploits the cores inside each node. The two compose cleanly.

!!! warning "This module does not use the Stream Triad"
    The triad is a single-loop, memory-bandwidth benchmark — there is nothing to communicate, so it is a poor vehicle for MPI. Instead, Example 3 uses a **distributed matrix-vector multiply**, where the matrix is split across processes and partial results are gathered. This exposes the real subject of MPI: data distribution and communication.

---

## The SPMD Model

MPI programs follow the **Single Program, Multiple Data** (SPMD) pattern: you launch *N* copies of the *same* executable, and each copy discovers its identity at runtime and acts accordingly.

Four calls form the skeleton of every MPI program:

| Call | Purpose |
|------|---------|
| `MPI_Init(&argc, &argv)` | Start the MPI runtime; create the default communicator |
| `MPI_Comm_rank(comm, &rank)` | Get *this* process's ID (0 to size-1) |
| `MPI_Comm_size(comm, &size)` | Get the total number of processes |
| `MPI_Finalize()` | Tear down the runtime (required — skipping it leaks resources or segfaults) |

A **communicator** is a named group of processes that can talk to each other. `MPI_COMM_WORLD` is the default, created by `MPI_Init`, containing every process in the job. A **rank** is a process's zero-based index within a communicator — the distributed-memory analogue of `omp_get_thread_num()`.

---

## Example 1 — Minimal MPI Program

`examples/MPI/Example1/MinWorkExampleMPI.c`:

```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  printf("Rank %d of %d\n", rank, nprocs);

  MPI_Finalize();
  return 0;
}
```

### Building and launching

MPI programs are compiled with a *compiler wrapper* (`mpicc` for C, `mpicxx` for C++) that adds the MPI headers and libraries, then launched with `mpiexec`/`mpirun` (or `srun` under SLURM):

=== "Workstation"
    ```bash
    cd examples/MPI/Example1
    mpicc MinWorkExampleMPI.c -o min_mpi
    mpiexec -n 4 ./min_mpi
    ```
    Or with the provided CMake build:
    ```bash
    cmake . && make
    mpiexec -n 4 ./MinWorkExampleMPI
    ```

=== "SLURM cluster"
    See `examples/MPI/run.slurm`. The number of ranks comes from `--ntasks`; launch with `srun ./min_mpi` so SLURM places one rank per task.

**Expected output** (order is non-deterministic — ranks run concurrently):

```
Rank 2 of 4
Rank 0 of 4
Rank 3 of 4
Rank 1 of 4
```

!!! tip "`-n` sets the number of ranks"
    `mpiexec -n 4` launches 4 processes. On a workstation these oversubscribe the available cores if you ask for more ranks than cores; on a cluster, the count must fit the resources SLURM granted (`--ntasks`).

---

## Example 2 — Point-to-Point Communication and Deadlock

The most fundamental MPI operation is sending a message from one rank to another. The blocking calls are:

```c
MPI_Send(buf, count, MPI_DOUBLE, dest,   tag, comm);
MPI_Recv(buf, count, MPI_DOUBLE, source, tag, comm, MPI_STATUS_IGNORE);
```

The `tag` is a user-chosen integer that lets a receiver match a specific message; `dest`/`source` name the partner rank. Example 2 pairs up processes (0↔1, 2↔3, …) and has each pair exchange an array.

### The starter code deadlocks

`examples/MPI/Example2/SendRecv1.cpp` has **every** rank call `MPI_Recv` *before* `MPI_Send`:

```c
MPI_Recv(xrecv, count, MPI_DOUBLE, partner_rank, tag, comm, MPI_STATUS_IGNORE);
MPI_Send(xsend, count, MPI_DOUBLE, partner_rank, tag, comm);
```

`MPI_Recv` is **blocking**: it does not return until a matching message arrives. Every rank is now stuck waiting to receive, and *no rank ever reaches its `MPI_Send`*. The program hangs forever — a classic **deadlock**.

!!! warning "A hung MPI job is usually a deadlock"
    If `mpiexec` never returns and the CPUs sit at 100%, suspect a Send/Recv ordering deadlock. Kill it with `Ctrl+C` and inspect the order of your blocking calls.

### Four ways to fix it

The `solutions/` directory walks through progressively better fixes. Each is worth understanding because they trade off safety, clarity, and performance.

=== "1. Send-then-Recv (unsafe)"
    `SendRecv2.cpp` simply swaps the order — `MPI_Send` first:
    ```c
    MPI_Send(xsend, count, MPI_DOUBLE, partner_rank, tag, comm);
    MPI_Recv(xrecv, count, MPI_DOUBLE, partner_rank, tag, comm, MPI_STATUS_IGNORE);
    ```
    This *appears* to work for small messages but is **not correct**. It relies on the MPI library buffering the outgoing message (the "eager" protocol). For messages larger than the eager threshold, `MPI_Send` blocks until the matching receive is posted — and you deadlock again. Never rely on this.

=== "2. Ordered by parity (correct)"
    `SendRecv3.cpp` breaks the symmetry: even ranks send first, odd ranks receive first.
    ```c
    if (rank % 2 == 0) {
      MPI_Send(xsend, count, MPI_DOUBLE, partner_rank, tag, comm);
      MPI_Recv(xrecv, count, MPI_DOUBLE, partner_rank, tag, comm, MPI_STATUS_IGNORE);
    } else {
      MPI_Recv(xrecv, count, MPI_DOUBLE, partner_rank, tag, comm, MPI_STATUS_IGNORE);
      MPI_Send(xsend, count, MPI_DOUBLE, partner_rank, tag, comm);
    }
    ```
    Now every send has a receiver ready. Correct, but verbose and easy to get wrong as patterns grow.

=== "3. MPI_Sendrecv (clean)"
    `SendRecv4.cpp` uses the combined call, which the library schedules safely:
    ```c
    MPI_Sendrecv(xsend, count, MPI_DOUBLE, partner_rank, tag,
                 xrecv, count, MPI_DOUBLE, partner_rank, tag,
                 comm, MPI_STATUS_IGNORE);
    ```
    One call, no ordering to get wrong. Ideal for the common "swap with a neighbor" pattern (e.g., halo exchange).

=== "4. Non-blocking (most flexible)"
    `SendRecv5.cpp` posts both operations without blocking, then waits:
    ```c
    MPI_Request requests[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    MPI_Irecv(xrecv, count, MPI_DOUBLE, partner_rank, tag, comm, &requests[0]);
    MPI_Isend(xsend, count, MPI_DOUBLE, partner_rank, tag, comm, &requests[1]);
    MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
    ```
    `MPI_Isend`/`MPI_Irecv` return immediately, handing back a `MPI_Request` handle. `MPI_Waitall` blocks until both complete. This cannot deadlock, and — crucially — it lets you **overlap computation with communication**: do useful work between posting the requests and waiting on them.

### Assignment

Build and run each version with an even number of ranks. Confirm `SendRecv1` hangs (kill it after a few seconds), then verify each fix completes.

```bash
cd examples/MPI/Example2
mpicxx SendRecv1.cpp -o sr1 && mpiexec -n 4 ./sr1   # hangs — Ctrl+C
mpicxx solutions/SendRecv5.cpp -o sr5 && mpiexec -n 4 ./sr5
```

Then **stress-test version 1 of the fix** (`SendRecv2`, send-then-recv): increase `count` from 10 to 10,000,000 and rerun. Observe that it now deadlocks too, proving the eager-buffer dependency.

---

## Example 3 — Distributed Matrix-Vector Multiply

This capstone combines everything: a collective broadcast, point-to-point distribution, local computation, and a gather. The problem is $y = Mx$ for an $N \times N$ matrix $M$ and vector $x$, with $N = 10000$.

The strategy — **domain decomposition** — splits the matrix by rows. Each process owns a contiguous block of rows and computes the corresponding block of the result.

```text
            Matrix M (N rows)              Vector x      Result y
          +------------------+                +--+         +--+
 rank 0   |   rows 0..k-1    | x  ............ |  |  ---->  |  | rows 0..k-1
          +------------------+                |  |         +--+
 rank 1   |   rows k..2k-1   | x  ............ |  |  ---->  |  | rows k..2k-1
          +------------------+                |  |         +--+
 rank 2   |  rows 2k..3k-1   | x  ............ |  |  ---->  |  | rows 2k..3k-1
          +------------------+                +--+         +--+
        (distributed by rows)            (broadcast to all)  (gathered to rank 0)
```

### The four communication phases

The code is instrumented with `MPI_Wtime()` around each phase so you can see where time goes:

1. **Broadcast the vector** — every process needs the *whole* vector $x$:
   ```c
   MPI_Bcast(vector.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   ```
   `MPI_Bcast` is a **collective**: one-to-all. Rank 0 (the `root`) sends; all ranks receive. Every process must call it.

2. **Distribute matrix rows** — rank 0 sends each process its block of rows with point-to-point `MPI_Send`, keeping its own block:
   ```c
   for (int dest = 1; dest < size; dest++) {
     MPI_Send(&matrix[dest_start_row * N], dest_rows * N, MPI_DOUBLE, dest, 0, comm);
   }
   ```

3. **Local computation** — each process multiplies its rows by the vector, entirely in its own memory (no communication):
   ```c
   for (size_t i = 0; i < local_rows; i++)
     for (size_t j = 0; j < N; j++)
       local_result[i] += local_matrix[i * N + j] * vector[j];
   ```

4. **Gather results** — each non-root process sends its partial result back to rank 0:
   ```c
   MPI_Send(local_result.data(), local_rows, MPI_DOUBLE, 0, 1, comm);
   ```

### Optimization: collectives instead of hand-rolled loops

The manual `MPI_Send` loop in phase 2 is exactly what the collective `MPI_Scatter` does — and the library implementation is typically faster (it can use a tree algorithm instead of a serial loop from the root). `examples/MPI/Example3/vector_matrix_mpi_opt.cpp` replaces the distribution loop with:

```c
MPI_Scatter(matrix.data(), rows_per_proc * N, MPI_DOUBLE,
            local_matrix.data(), rows_per_proc * N, MPI_DOUBLE,
            0, MPI_COMM_WORLD);
```

!!! tip "Prefer collectives over point-to-point loops"
    Whenever you find yourself writing `for (dest = 1; dest < size; dest++) MPI_Send(...)` from the root, there is almost certainly a collective (`MPI_Scatter`, `MPI_Gather`, `MPI_Bcast`, `MPI_Allgather`) that expresses it more clearly *and* runs faster. Collectives also give the MPI runtime freedom to optimize for the network topology.

!!! note "MPI_Scatter requires even division"
    `MPI_Scatter` sends the *same* count to every rank. When `N` is not divisible by `size`, the last process's extra rows (handled by the `(rank == size-1)` logic in the manual version) need `MPI_Scatterv`, the variable-count variant. This is the subject of a discussion question below.

### Build and run

=== "Workstation"
    ```bash
    cd examples/MPI/Example3
    mpicxx vector_matrix_mpi.cpp -o vmm
    mpiexec -n 4 ./vmm
    ```

=== "SLURM cluster"
    See `examples/MPI/run.slurm`. Set `--ntasks` to the number of ranks; with $N = 10000$, 4–16 ranks is reasonable. Launch the binary with `srun`.

**Expected output:**

```
Matrix and vector initialized.
Initialization time: 0.42 seconds
Distribution time: 0.31 seconds
Computation time: 0.18 seconds
Gather time: 0.001 seconds
Result vector computed successfully.
First 5 elements of result: ...

Timing Summary:
...
Total time: 0.91 seconds
```

Look at the **distribution time versus computation time**. For this problem the matrix is large ($10000^2 \times 8$ bytes = 800 MB) and the per-row work is small, so moving the data can cost as much as the math — the distributed-memory echo of the "data transfer dominates" lesson from the [CUDA module](04-cuda.md).

---

## Discussion Questions

1. In Example 2, `SendRecv2` (send-then-receive) passes with `count = 10` but deadlocks with `count = 10000000`. Explain the role of the MPI eager/rendezvous protocol threshold. Why is depending on it a latent bug rather than a working optimization?

2. Example 3 distributes the matrix by *rows*. If instead you distributed by *columns*, what would change about the communication pattern and the final reduction? Which decomposition is better for matrix-vector multiply, and why?

3. The manual distribution loop was replaced by `MPI_Scatter`, but `MPI_Scatter` requires every rank to receive the same number of rows. Sketch how `MPI_Scatterv` handles the case where `N` is not divisible by the number of ranks. What extra arrays does it require?

4. Using the timing breakdown, suppose you double the number of ranks. Amdahl's Law aside, what happens to *distribution* time and *computation* time individually? At what point does adding ranks stop helping?

5. A production code would overlap communication and computation using the non-blocking pattern from Example 2. In the matrix-vector multiply, what work could a process do *while* its matrix rows are still arriving?

---

## References

### Reference materials

- [MPI Forum standard](https://www.mpi-forum.org/docs/) — the normative specification that defines the semantics of every MPI routine. The authoritative source, but dense.
- [Open MPI documentation](https://www.open-mpi.org/doc/) — implementation docs and per-function man pages; the source for the function links below.
- [MPICH documentation](https://www.mpich.org/static/docs/latest/) — man pages for the other major open-source implementation. The API semantics match the standard; the two implementations differ only in implementation-specific notes.
- [rookieHPC MPI guide](https://rookiehpc.org/mpi/) — community-maintained, beginner-friendly pages with plain-language parameter explanations and a full runnable example for each call.

### Functions used in this module

Links point to the Open MPI man pages.

**Environment and timing:**
[MPI_Init](https://www.open-mpi.org/doc/current/man3/MPI_Init.3.php),
[MPI_Finalize](https://www.open-mpi.org/doc/current/man3/MPI_Finalize.3.php),
[MPI_Comm_rank](https://www.open-mpi.org/doc/current/man3/MPI_Comm_rank.3.php),
[MPI_Comm_size](https://www.open-mpi.org/doc/current/man3/MPI_Comm_size.3.php),
[MPI_Wtime](https://www.open-mpi.org/doc/current/man3/MPI_Wtime.3.php)

**Point-to-point communication:**
[MPI_Send](https://www.open-mpi.org/doc/current/man3/MPI_Send.3.php),
[MPI_Recv](https://www.open-mpi.org/doc/current/man3/MPI_Recv.3.php),
[MPI_Sendrecv](https://www.open-mpi.org/doc/current/man3/MPI_Sendrecv.3.php),
[MPI_Isend](https://www.open-mpi.org/doc/current/man3/MPI_Isend.3.php),
[MPI_Irecv](https://www.open-mpi.org/doc/current/man3/MPI_Irecv.3.php),
[MPI_Waitall](https://www.open-mpi.org/doc/current/man3/MPI_Waitall.3.php)

**Collective communication:**
[MPI_Bcast](https://www.open-mpi.org/doc/current/man3/MPI_Bcast.3.php),
[MPI_Scatter](https://www.open-mpi.org/doc/current/man3/MPI_Scatter.3.php),
[MPI_Scatterv](https://www.open-mpi.org/doc/current/man3/MPI_Scatterv.3.php),
[MPI_Gather](https://www.open-mpi.org/doc/current/man3/MPI_Gather.3.php),
[MPI_Allgather](https://www.open-mpi.org/doc/current/man3/MPI_Allgather.3.php)
