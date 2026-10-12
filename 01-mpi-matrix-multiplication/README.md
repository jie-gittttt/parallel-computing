# MPI Matrix Multiplication — D = A×B + C

[English](README.md) | [中文](README.zh-CN.md)

A distributed matrix computation over MPI, where the decisive speedup came from reordering
the loops for cache locality rather than from adding processes.

## 1. Overview

The task is to compute `D = A×B + C` for three 1023×1023 double-precision matrices using
message passing only — no shared memory, no threads, no BLAS. Process 0 reads the input,
distributes the work, collects the result, and writes it back.

The assignment has two goals, and the second is the interesting one:

1. **Decomposition.** Split a dense linear algebra kernel across processes with MPI
   collectives, handling a matrix dimension that does not divide evenly.
2. **Memory behavior.** Measure how much of the achievable speedup comes from *how the
   inner loop walks memory* rather than from how many cores are running it.

The workload is the canonical BLAS-3 kernel, so the loop-order lesson transfers directly to
any dense linear algebra or stencil code.

## 2. Key Features / Contributions

- **Master-worker decomposition with row-block partitioning** — process 0 owns I/O and
  orchestration; workers only compute.
- **Variable-count collectives throughout.** 1023 divides evenly by neither 2 nor 4, so the
  code uses `MPI_Scatterv` for both A and C and `MPI_Gatherv` for D, with `displs` derived
  from a running row offset. B is broadcast in full with `MPI_Bcast`.
- **Two serial optimizations implemented and benchmarked independently** — i-k-j loop
  interchange, and pre-transposing B — so their contribution is separable from the
  parallel one.
- **Correctness verified on every run** against the instructor's `verify_` checker
  (`比对成功！`).
- **Raw timing logs preserved** from three separate sessions on the shared course server,
  including the ones that don't flatter the headline numbers.

## 3. Methodology / Architecture

### Data distribution

| Matrix | Shape | Method | Why |
| :--- | :--- | :--- | :--- |
| **A** | M×N, row-split | `MPI_Scatterv` | 1023 is not divisible by 2 or 4 — uneven blocks are mandatory |
| **C** | M×K, row-split | `MPI_Scatterv` | Same, and must align with A's row blocks |
| **B** | N×K, full | `MPI_Bcast` | Every output row of D needs the *entire* B; it cannot be split |
| **D** | M×K, gathered | `MPI_Gatherv` | Symmetric with Scatterv — reassembles unequal blocks |

```
                    ┌─────────────┐
   process 0 ──────▶│  read input │  RandMatrix.txt  (A, B, C)
   (master)         └──────┬──────┘
                           │  Scatterv A (rows)   sendcounts = {256,256,256,255} × 1023
                           │  Scatterv C (rows)
                           │  Bcast    B (full, 1023×1023 ≈ 8.4 MB)
                           ▼
        ┌──────────┬──────────┬──────────┬──────────┐
        │  P0      │  P1      │  P2      │  P3      │   local D_block += A_block × B
        └────┬─────┴────┬─────┴────┬─────┴────┬─────┘
             └──────────┴────┬─────┴──────────┘
                             │  Gatherv D
                             ▼
                    ┌─────────────┐
                    │ write output│  result.txt
                    └─────────────┘
```

**Why row blocks and not column blocks.** Matrices are stored row-major, so a row block is
one contiguous span that can be handed to a collective directly. A column block would have to
be packed across a stride first, which costs more than the communication it saves.

### Serial optimization 1 — i-k-j loop interchange

The naive triple loop is i-j-k:

```cpp
for (int i = 0; i < M; ++i)
    for (int j = 0; j < K; ++j)
        for (int k = 0; k < N; ++k)
            D[i][j] += A[i][k] * B[k][j];   // B[k][j] strides by K
```

The innermost index is `k`, so `B[k][j]` jumps a full row (K elements) each iteration. Of a
64-byte cache line only 8 bytes are used before it is evicted — roughly **1/8 effective
cache utilization**.

Interchanging to i-k-j:

```cpp
for (int i = 0; i < M; ++i)
    for (int k = 0; k < N; ++k)
        for (int j = 0; j < K; ++j)
            D[i][j] += A[i][k] * B[k][j];   // both stride by 1
```

Now `B[k][j]` and `D[i][j]` both advance contiguously, and `A[i][k]` is loop-invariant with
respect to the inner loop, so the compiler can keep it in a register. Every cache line is
used in full. This is not a pure reordering of `for` statements — `A[i][k]` becoming
invariant changes the arithmetic, and the accumulator has to be handled accordingly.

### Serial optimization 2 — pre-transposing B

```cpp
// transpose once: Bt is K×N, so Bt[j][k] is row-contiguous
for (int i = 0; i < M; ++i)
    for (int j = 0; j < K; ++j) {
        double s = 0;
        for (int k = 0; k < N; ++k)
            s += A[i][k] * Bt[j][k];        // both operands contiguous
        D[i][j] = s + C[i][j];
    }
```

The transpose costs O(N·K) extra traffic, but against a main loop of 1023³ ≈ 1.07 billion
multiply-adds that is negligible.

## 4. Experiments and Results

Course shared server, 1023×1023, `MPI_Wtime` around the compute phase:

| Processes | Naive i-j-k | After loop interchange | Speedup (optimized) |
| :-: | --: | --: | --: |
| 1 | 10.494 s | 7.710 s | 1.00× |
| 2 | 6.659 s | 3.946 s | 1.95× |
| 4 | 3.398 s | **1.225 s** | **6.29×** |

- **Serial optimization alone: 1.36×** (10.494 s → 7.710 s) with no extra hardware. This is
  purely improved memory access.
- **Parallel speedup at 4 processes: 3.09× → 6.29×** after the transformation.
- **End to end: 10.494 s → 1.225 s, an 8.57× overall improvement.**

Every run was checked with the instructor's verification program:

```
$ ./verify_ RandMatrix.txt result.txt
比对成功！
```

### Why is the 4-process speedup superlinear?

6.29× on 4 processes exceeds linear scaling. The most plausible explanation is cache: in the
serial case the full 1023×1023 array of doubles (≈8.4 MB) does not fit in L2, whereas each of
the 4 processes handles only 256 rows, so the working set per core shrinks enough to stay
resident. Combined with the locality gain from the loop interchange, the two effects
compound past linearity.

This is an inference from the timing data, not something measured with hardware counters — see
the Limitations section.

### Timing variance

Absolute numbers differ between sessions on the same code. Three logged sessions:

| | 1 process | 2 processes | 4 processes |
| :--- | --: | --: | --: |
| Session A | 7.640 s | 3.836 s | 2.762 s |
| Session B | 5.400 s | 4.135 s | 3.033 s |
| Session C (table above) | 10.494 s | 6.659 s | 3.398 s (naive) |

The server is shared with other students' jobs, so clock frequency and cache contention vary.
The headline table uses the last session with a complete verification record; the raw output
of all sessions is in [`results/run-log.txt`](results/run-log.txt) and
[`results/timing-sessions.txt`](results/timing-sessions.txt).

## 5. Tech Stack

- **Language**: C++11
- **Parallelism**: MPI (`mpic++`, MPICH/OpenMPI on the course server)
- **Timing**: `MPI_Wtime`
- **Build**: single translation unit, no build system required

## 6. Installation and Usage

**Requirements** — an MPI implementation (`mpic++` on `PATH`) and a 1023×1023
`RandMatrix.txt` containing A, B and C, in the format described in
[`report/io-spec.pdf`](report/io-spec.pdf).

```bash
mpic++ -o Matrix src/Matrix.cpp -std=c++11
mpirun -np 4 ./Matrix        # reads RandMatrix.txt, writes result.txt
```

The process count is a runtime argument; `sendcounts` and `displs` are recomputed for
any number of processes, so `-np 2`, `-np 8`, etc. all work.

**Expected output** — timing lines per process, followed by a total:

```
------------------------------------------
Job Complete.
Total execution time with 4 processes: 1.225 seconds
------------------------------------------
```

## 7. Limitations

- **The superlinear result is unexplained at the hardware level.** The cache working-set
  argument is consistent with the data but was not verified with performance counters.
  A rigorous version would measure L2/L3 miss rates at each process count.
- **Scaling stops at 4 processes.** No strong or weak scaling study was performed, so the
  point at which communication overtakes computation is unknown.
- **Shared-server noise.** Runs are not isolated, so the absolute timings — and possibly the
  shape of the speedup curve — carry load-dependent error. This is why the raw logs are kept.
- **One matrix size only.** All conclusions are drawn at N = 1023. Nothing here demonstrates
  how the tradeoff shifts as the matrix outgrows cache entirely.
- **B is broadcast in full, every run.** At 8.4 MB this is the dominant communication cost and
  it does not shrink with process count. A production implementation would use a
  decomposition that partitions B as well (SUMMA, Cannon) or overlap communication with
  computation via `MPI_Isend`/`MPI_Irecv`.
- **Blocking collectives only.** No computation/communication overlap was attempted.

## Repository Structure

```
01-mpi-matrix-multiplication/
├── src/Matrix.cpp                  # main program (loop-interchange version)
├── report/
│   ├── report.pdf                  # lab report (Chinese)
│   └── io-spec.pdf                 # I/O format specification (instructor-provided)
├── slides/optimization.pdf         # optimization presentation (Chinese)
├── results/
│   ├── run-log.txt                 # full terminal output, three runs
│   └── timing-sessions.txt         # per-session timing summary
└── docs/
    ├── mpi-loop-interchange.html   # interactive: cache line reuse under each loop order
    └── mpi-transpose.html          # interactive: original vs. transposed B
```

Reports and slides are written in Chinese.
