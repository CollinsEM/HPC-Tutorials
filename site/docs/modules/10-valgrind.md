# Module 10: Valgrind — Memory & Threading Debugging

## Learning Objectives

By the end of this module you will be able to:

- Explain what Valgrind instruments and why it is slow but thorough
- Find leaks, invalid accesses, and uninitialized reads with **memcheck**
- Detect data races and lock errors with **helgrind** and **DRD**
- Profile cache behavior and hotspots with **cachegrind** and **callgrind**
- Track heap growth with **massif**
- Apply best practices: build flags, suppressions, and a fix-the-first-error discipline

---

## What Valgrind Is

[Valgrind](https://valgrind.org/docs/manual/) is a *dynamic binary instrumentation* framework. It runs your program on a synthetic CPU, watching every memory access and instruction. That makes it 10–50× slower than native execution, but it needs no special privileges and catches bugs that only manifest at runtime — the ones that cause wrong answers, sporadic crashes, or the nondeterministic results that masquerade as "performance" problems.

Valgrind is a collection of tools selected with `--tool=`:

| Tool | Finds |
|------|-------|
| `memcheck` (default) | leaks, invalid/uninitialized memory access |
| `helgrind` | data races, lock-ordering errors |
| `drd` | data races (lower overhead, different heuristics) |
| `cachegrind` | cache misses, branch mispredictions (simulated) |
| `callgrind` | call-graph hotspots (+ optional cache sim) |
| `massif` | heap memory growth over time |

!!! warning "Valgrind is CPU-only"
    Valgrind instruments host CPU instructions. It cannot see inside a CUDA/HIP kernel or measure GPU memory. For GPU code use the Nsight tools in [Module 11](11-nsight.md); for native CPU hardware counters use LIKWID or `perf` as described in [Module 9](09-profiling.md).

!!! tip "If Valgrind is not installed — compiler sanitizers"
    GCC ≥ 4.8 and Clang ≥ 3.1 ship built-in sanitizers that cover the most
    common Valgrind use cases at a fraction of the overhead. No external tool
    is needed — just recompile:

    | Valgrind tool | Compiler sanitizer | Flag | Overhead |
    |---------------|--------------------|------|----------|
    | `memcheck` (invalid access, leaks) | AddressSanitizer (ASan) | `-fsanitize=address -fno-omit-frame-pointer` | ~2× |
    | `memcheck --track-origins` (uninitialized reads) | MemorySanitizer (MSan) | `-fsanitize=memory` | ~3× (Clang only) |
    | `helgrind` / `drd` (data races) | ThreadSanitizer (TSan) | `-fsanitize=thread` | ~5–15× |
    | `cachegrind` (cache misses) | No direct equivalent; use `perf stat -e cache-misses` | — | — |
    | `callgrind` (hotspots) | `gprof` (`-pg`) or `perf record`/`perf report` | — | — |

    Sanitizers are usually the first choice on a developer workstation because
    they are faster, easier to set up, and produce actionable error messages.
    Valgrind's `cachegrind` and `callgrind` remain useful on clusters where
    compiler versions are older or `perf` access requires elevated permissions.

    ASan example:
    ```bash
    gcc -g -O1 -fsanitize=address -fno-omit-frame-pointer stream_triad.c timer.c -o stream_triad_asan
    ./stream_triad_asan
    ```
    TSan example (for a threaded program):
    ```bash
    gcc -g -O1 -fsanitize=thread your_omp_program.c -fopenmp -o prog_tsan
    ./prog_tsan
    ```
    ASan and TSan cannot be used together in the same binary. Run separate
    builds for memory and threading checks.

---

## memcheck — Memory Correctness

The default tool. Build with debug symbols (`-g`) so reports carry file and line numbers, and prefer a light optimization level — `-O1` keeps line mapping honest while staying realistic, whereas `-O0` changes behavior and `-O2`/`-O3` can inline away the lines you need:

```bash
gcc -g -O1 stream_triad.c timer.c -o stream_triad
valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./stream_triad
```

Key options:

- `--leak-check=full` — report each leaking allocation with its stack trace.
- `--show-leak-kinds=all` — include *definitely*, *indirectly*, *possibly*, and *still-reachable* leaks.
- `--track-origins=yes` — for uninitialized-value errors, trace back to where the bad value came from (slower, but worth it).

### Reading the output

memcheck reports each error with the offending stack and a category. The ones you will see most:

- **Invalid read/write of size N** — out-of-bounds access (heap overrun, off-by-one).
- **Conditional jump depends on uninitialised value** — you read memory you never wrote (a classic: an accumulator declared but not zeroed).
- **N bytes definitely lost** — a leak: allocated, never freed, no surviving pointer.
- **Mismatched free() / delete** — `free` on `new`, or `delete` on `malloc`.

!!! tip "Fix the first error first"
    Memory errors cascade — one bad write corrupts state and spawns dozens of downstream reports. Always fix the *first* reported error, then re-run. Half your error list often vanishes.

The leak summary distinguishes *definitely lost* (no pointer remains — a real bug) from *still reachable* (a pointer exists at exit — usually a global never freed, often benign). Focus on the former.

---

## helgrind & DRD — Data Races

Memory correctness is only half the story for parallel code; the other half is **threading correctness**. A data race — two threads accessing the same location concurrently with at least one write, and no synchronization — produces nondeterministic, backend-dependent results. That is exactly the failure mode of the [MATAR `FOR_ALL` race in Module 8](08-kokkos.md), where parallelizing the `k` reduction made multiple threads update `C(i,j)` at once.

```bash
# Compile the OpenMP/threaded program, then:
valgrind --tool=helgrind ./your_threaded_program
```

helgrind reports "Possible data race during write of size N" with the two conflicting stacks — the two threads and the unsynchronized location. DRD is an alternative with lower memory overhead:

```bash
valgrind --tool=drd ./your_threaded_program
```

!!! note "Threading runtimes generate noise"
    OpenMP and pthread runtimes do their own lock-free synchronization that helgrind cannot always reason about, so you will see false positives from inside `libgomp`/`libpthread`. Suppress them (below) so the real races in *your* code stand out. For OpenMP specifically, compiling with a helgrind-aware runtime or annotating with `ANNOTATE_*` macros reduces the noise.

This is the tool to reach for when an OpenMP loop ([Module 3](03-openmp.md)) gives different answers on different runs, or when you suspect a missing `reduction`/`critical` clause.

---

## cachegrind & callgrind — Where the Time Goes

### callgrind — hotspots and call graph

```bash
valgrind --tool=callgrind --callgrind-out-file=callgrind.out ./stream_triad
callgrind_annotate callgrind.out
```

`callgrind_annotate` ranks functions by instruction count with per-line detail; [**KCachegrind**](https://kcachegrind.github.io/) opens `callgrind.out` as an interactive call graph. Add `--cache-sim=yes --branch-sim=yes` to also model cache and branch behavior in the same run.

### cachegrind — cache and branch simulation

```bash
valgrind --tool=cachegrind ./stream_triad
cg_annotate cachegrind.out.<pid>
```

cachegrind reports I1/D1/LL miss rates per function and per line — the tool to confirm a suspected cache problem, such as a strided access or an [array-of-structs layout](01-hardware.md) thrashing the cache.

!!! warning "Simulated, not measured"
    cachegrind models a *generic* two-level cache and ignores hardware prefetchers and out-of-order execution. Use it for *relative* comparisons (this loop misses far more than that one); use LIKWID's hardware counters ([Module 9](09-profiling.md)) for *absolute* numbers on real silicon.

---

## massif — Heap Profiling

When a program's memory footprint grows unexpectedly, massif samples the heap over time:

```bash
valgrind --tool=massif ./your_program
ms_print massif.out.<pid>
```

`ms_print` draws an ASCII graph of heap usage against time and lists the call sites responsible for the peak. Use it to find a leak that *isn't* technically lost (memory that keeps growing but is still reachable), or to right-size allocations.

---

## Best Practices

- **Build with `-g` and `-O1`.** Debug symbols give line numbers; `-O1` keeps them trustworthy. Avoid `-O0` (unrealistic) and heavy `-O2`/`-O3` inlining (loses lines) while debugging.
- **Shrink the problem.** Valgrind's slowdown makes full-scale runs painful. Reproduce the bug with the smallest input that still triggers it — drop `STREAM_ARRAY_SIZE`, `NumPatients`, or iteration counts.
- **Use suppressions for known noise.** MPI, OpenMP, and CUDA runtimes trip false positives. Capture them once with `--gen-suppressions=all` and feed them back via `--suppressions=my.supp` so real bugs are not buried.
- **Fix top-down.** The first error often causes the rest; re-run after each fix.
- **Make it routine, not a last resort.** A clean memcheck run before committing catches the off-by-one that would otherwise become a production crash.

---

## Common Bug Patterns

| Symptom | Likely cause | Tool |
|---------|--------------|------|
| Wrong result, "uninitialised value" | accumulator declared but not zeroed | memcheck `--track-origins` |
| Sporadic crash near an array | off-by-one heap overrun | memcheck |
| Memory grows until OOM | leak in an error/retry path | memcheck `--leak-check=full`, massif |
| Result changes run-to-run (threads) | missing `reduction`/lock; data race | helgrind / DRD |
| Use-after-free / double free | freeing in a loop or destructor twice | memcheck |

---

## Assignment

Tasks 1–3 can be completed with either Valgrind or compiler sanitizers. Where
both options are shown, try both and compare the error messages.

1. **Clean baseline.** Build `examples/autovec/stream_triad.c` with `-g -O1` and check for memory errors. It should be clean — confirm there are no invalid accesses, and note the "still reachable" total (the static arrays).
   ```bash
   # With Valgrind:
   valgrind --leak-check=full --show-leak-kinds=all ./stream_triad

   # With AddressSanitizer (no Valgrind needed):
   gcc -g -O1 -fsanitize=address -fno-omit-frame-pointer stream_triad.c timer.c -o stream_triad_asan
   ./stream_triad_asan
   ```

2. **Inject a bug.** Change the triad loop bound from `i < STREAM_ARRAY_SIZE` to `i <= STREAM_ARRAY_SIZE`. Find the out-of-bounds write report and confirm it points at the triad line. Revert.
   ```bash
   # Either tool will catch this; ASan is faster:
   valgrind ./stream_triad           # "Invalid write of size 8"
   ./stream_triad_asan               # "heap-buffer-overflow" or "global-buffer-overflow"
   ```

3. **Uninitialized value.** In a copy of the code, remove the initialization of `b[]`. Trace the uninitialized-value error back to the missing init loop.
   ```bash
   valgrind --track-origins=yes ./stream_triad
   # Note: ASan does not detect uninitialized reads; use Valgrind or
   # Clang's MemorySanitizer (-fsanitize=memory) for this case.
   ```

4. **Hotspot.** Confirm that the triad loop dominates execution time.
   ```bash
   # With Valgrind callgrind:
   valgrind --tool=callgrind --callgrind-out-file=callgrind.out ./stream_triad
   callgrind_annotate callgrind.out

   # With gprof (no Valgrind needed):
   gcc -g -O1 -pg stream_triad.c timer.c -o stream_triad_pg
   ./stream_triad_pg
   gprof stream_triad_pg gmon.out | head -30
   ```

5. **(Threads)** Build an OpenMP example ([Module 3](03-openmp.md)) and check for data races. Identify which reports come from the runtime (noise) versus your code.
   ```bash
   # With Valgrind helgrind:
   valgrind --tool=helgrind ./your_omp_program

   # With ThreadSanitizer (much faster, no Valgrind needed):
   gcc -g -O1 -fsanitize=thread -fopenmp your_omp_program.c -o prog_tsan
   ./prog_tsan
   ```

---

## Discussion Questions

1. memcheck reports "still reachable: 19,200,000 bytes" for the Stream Triad but zero "definitely lost." Is this a leak you should fix? Why or why not?

2. Why must you build with `-g` for useful Valgrind output, and why is `-O1` usually a better choice than either `-O0` or `-O3` while debugging?

3. helgrind flags a "race" inside `libgomp` that you did not write. How do you decide whether it is a real bug in your use of OpenMP versus benign runtime noise — and what do you do about it either way?

4. You have a wrong-answer bug that only appears with 8 threads, never with 1. Which Valgrind tool do you reach for, and why would memcheck alone not find it?

??? "Show solutions"
    **Question 1**: "Still reachable" means the allocations are still pointed to
    by live pointers at program exit — they are not lost, just never explicitly
    freed. The Stream Triad's three arrays ($3 \times 80\,\text{M} \times 8\,\text{bytes}
    = 1.92\,\text{GB}$... actually $3 \times 80\,000\,000 \times 8 = 1,920,000,000$
    bytes, but the question says 19,200,000 which corresponds to smaller example
    arrays) simply fall out of scope at `return 0`. The OS reclaims all process
    memory on exit regardless. "Still reachable" is not a leak — no data has been
    lost and there is no memory corruption. It is not worth fixing unless the code
    runs as a long-lived server where memory growth would accumulate, or security
    policy requires zeroing sensitive data before deallocation. "Definitely lost"
    (no pointer to the block exists anywhere) is the category that warrants
    immediate attention.

    **Question 2**: `-g` embeds DWARF debugging information — source file names,
    line numbers, function names — that Valgrind uses to produce human-readable
    stack traces. Without it, you get only raw memory addresses. `-O0` disables
    all optimization, which sometimes hides bugs that only manifest under
    optimization (register allocation, loop transforms, inlining can expose or
    obscure memory errors). `-O3` enables aggressive inlining and loop transforms
    that can produce confusing stack traces, eliminate source-level variables
    (they become registers), and change memory-access patterns enough to alter
    which errors Valgrind detects. `-O1` is the pragmatic middle ground: enough
    optimization to reproduce most real-world bugs, while preserving most
    debug structure and keeping stack traces readable.

    **Question 3**: First, examine the full stack trace. If the flagged race is
    *entirely inside* OpenMP runtime code (e.g., inside `libgomp.so`) with none
    of your code in the trace, it is almost certainly a known benign pattern:
    OpenMP runtimes use lock-free work-stealing or lazy initialization that
    appears as a race to a happens-before detector but is deliberately correct.
    You can confirm by checking helgrind's suppression file for known OpenMP
    suppressions (`--gen-suppressions=all` generates a suppressions entry you
    can add to `~/.valgrind.supp`). If *your* code appears anywhere in the
    trace — even in a function that calls an OpenMP routine — investigate as a
    real bug. Either way, never silently suppress a warning without understanding
    whether your code is in the call path.

    **Question 4**: Reach for **helgrind** (or DRD), or for **ThreadSanitizer**
    (`-fsanitize=thread`) if Valgrind is not available. A wrong-answer bug that
    appears only with multiple threads is almost certainly a **data race**: two
    threads read and write the same memory location without proper synchronization,
    producing a result that depends on scheduling order. memcheck detects memory
    errors — reads from uninitialized memory, out-of-bounds accesses, invalid frees.
    A race that writes the correct type to the correct address is invisible to
    memcheck; the write is well-typed and in-bounds. helgrind and TSan both track
    concurrent accesses and flag when two threads reach the same location without a
    happens-before relationship between them. TSan is typically faster (5–15×
    overhead vs. 50–100× for helgrind) and produces cleaner output for OpenMP code.

---

## References

### Reference materials

- [Valgrind manual](https://valgrind.org/docs/manual/) — top-level documentation.
- [KCachegrind](https://kcachegrind.github.io/) — interactive viewer for callgrind output.

### Valgrind tool manuals

- [Memcheck](https://valgrind.org/docs/manual/mc-manual.html) — memory errors and leaks.
- [Helgrind](https://valgrind.org/docs/manual/hg-manual.html) — data races and locking errors.
- [DRD](https://valgrind.org/docs/manual/drd-manual.html) — alternative data-race detector.
- [Cachegrind](https://valgrind.org/docs/manual/cg-manual.html) — cache and branch-prediction simulation.
- [Callgrind](https://valgrind.org/docs/manual/cl-manual.html) — call-graph and instruction profiling.
- [Massif](https://valgrind.org/docs/manual/ms-manual.html) — heap profiler.

### Compiler sanitizers (alternatives)

- [AddressSanitizer](https://clang.llvm.org/docs/AddressSanitizer.html) — Clang/LLVM documentation; same flags work with GCC.
- [MemorySanitizer](https://clang.llvm.org/docs/MemorySanitizer.html) — uninitialized-read detection (Clang only).
- [ThreadSanitizer](https://clang.llvm.org/docs/ThreadSanitizer.html) — data-race detection; works with GCC and Clang.
- [UndefinedBehaviorSanitizer](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html) — catches integer overflow, null dereference, and other UB.
