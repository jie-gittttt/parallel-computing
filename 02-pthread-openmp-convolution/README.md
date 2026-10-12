# Parallel Image Convolution — Pthread vs. OpenMP

[English](README.md) | [中文](README.zh-CN.md)

A 5×5 Gaussian convolution of BMP images implemented twice — once with raw Pthreads and once
with OpenMP — where cutting the per-pixel work in half by exploiting kernel separability
reverses which scheduler is the better choice.

## 1. Overview

The task is to convolve a BMP image with a fixed 5×5 Gaussian kernel (σ = 1.5) using both
shared-memory paradigms, then compare them. The program reads a BMP, convolves it, and writes
a BMP of identical dimensions.

Two goals:

1. **Compare the paradigms.** Pthreads gives explicit control over thread creation, work
   partitioning and synchronization; OpenMP expresses the same decomposition as compiler
   directives. The assignment is to see what each costs you in code and in performance.
2. **Optimize the algorithm, not just the parallelism.** A 5×5 Gaussian is *separable*, and
   exploiting that halves the arithmetic — independently of how many cores are used.

Borders are handled with **zero padding** rather than cropping, so the output matches the
input dimensions exactly.

## 2. Key Features / Contributions

- **Four complete implementations** — Pthread and OpenMP, each in a direct 2D version and a
  separable version — so the effect of the algorithmic change and the effect of the
  parallelism stay separable.
- **Hand-written barrier** in the Pthread separable version. The vertical pass reads rows
  produced by the horizontal pass, so all threads must finish stage one before any starts
  stage two. OpenMP gets this for free from the implicit barrier at the end of `parallel for`;
  Pthreads needs `pthread_barrier_t`.
- **A float intermediate buffer** in the separable versions. The direct versions truncate to
  8-bit after every pass — twice per pixel — while the separable versions keep the
  intermediate in `float` and truncate once. This costs one extra image-sized buffer and
  yields a measurably lower MSE.
- **A documented performance reversal** between the two paradigms depending on iteration
  granularity (see Results).
- **Timing data and a paradigm comparison table** preserved as CSV under `data/`.

## 3. Methodology / Architecture

### Separable convolution

A 2D Gaussian is the outer product of two 1D Gaussians: `G2D(x, y) = G1D(x) · G1D(y)`. So a
5×5 convolution can be run as two 1D passes:

| | Multiply-adds per pixel | Intermediate |
| :--- | --: | :--- |
| Direct 2D | 5 × 5 = **25** | — |
| Separable: horizontal 1×5 + vertical 5×1 | 5 + 5 = **10** | one float image |

```
 input (8-bit)                    stage 1                    stage 2                output
   ┌────────┐   1×5 horizontal   ┌────────┐   5×1 vertical  ┌────────┐  truncate  ┌───────┐
   │  H×W   │ ─────────────────▶ │ float  │ ──────────────▶ │ float  │ ─────────▶ │ 8-bit │
   └────────┘   (row-independent)│  H×W   │  (reads ±2 rows)└────────┘            └───────┘
                                 └────────┘
                                  ▲ all threads must reach here before stage 2 starts
```

Arithmetic drops to 40% of the direct version, predicting a 2.5× single-threaded speedup.
The two passes require a **barrier** between them.

### Pthread: explicit thread management

```cpp
pthread_create(&threads[i], NULL, worker, &args[i]);
pthread_join(threads[i], NULL);
```

`pthread_create` accepts a single `void*`, so a struct packs the image pointers, the row range
and the kernel. Work is split **statically** — each thread gets one contiguous band of rows,
computed by hand from `start_row` / `end_row`.

### OpenMP: compiler directives

```cpp
#pragma omp parallel for collapse(2) schedule(dynamic)
for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) { ... }
```

| Directive | Effect |
| :--- | :--- |
| `parallel for` | Distributes the following loop across threads (fork-join) |
| `collapse(2)` | Flattens the `y` and `x` loops into one iteration space before splitting. Without it the split can only happen at row granularity, so the parallel unit is a row; with it, the unit is a single pixel |
| `schedule(dynamic)` | Threads claim the next chunk when they finish (work stealing), which addresses the long tail caused by differing cache behavior between image regions |

OpenMP infers `y` and `x` as `private` and the image pointers as `shared`. Matching the
dynamic scheduling behavior in Pthreads would require a hand-built task queue and mutex.

## 4. Experiments and Results

Same test image, seconds, mean of 3 measurements:

| Implementation | 1 thread | 2-thread speedup | 4 threads | 4-thread speedup |
| :--- | --: | --: | --: | --: |
| Pthread, direct 2D | 1.043 | 1.93× | 0.323 | 3.23× |
| Pthread, separable | 0.400 | 1.96× | 0.128 | 3.12× |
| OpenMP, direct 2D | 1.053 | 1.53× | 0.370 | 2.84× |
| OpenMP, separable | 0.392 | 1.97× | 0.111 | **3.52×** |

**Effect of separability alone:** Pthread 1.043 s → 0.400 s (**2.61×**), OpenMP 1.053 s →
0.392 s (**2.69×**). This matches the predicted 2.5× and slightly exceeds it; the extra margin
comes from the float intermediate buffer removing one round of 8-bit truncation per pixel.

### The reversal

On the **direct** version, Pthread's static partitioning wins at 4 threads (3.23× vs. 2.84×).
On the **separable** version, OpenMP's dynamic scheduling wins (3.52× vs. 3.12×).

The explanation is iteration granularity. The direct version does 25 multiply-adds per pixel,
so each iteration is substantial work and the atomic operation behind `schedule(dynamic)` to
claim the next chunk is pure overhead. The separable version drops to 10 multiply-adds per
pixel, **which doubles scheduling overhead as a fraction of total time**, and at that point
the load-balancing benefit — especially the cache long tail in the vertical pass — outweighs it.

The takeaway: **no scheduling policy is universally better; it depends on the size of one
iteration.** Keeping both implementations and measuring each at both granularities is what
makes this visible.

> Full dimension-by-dimension comparison (paradigm, code volume, work partitioning,
> synchronization, portability): [`data/comparison.csv`](data/comparison.csv).

## 5. Tech Stack

- **Language**: C++ with POSIX threads and OpenMP
- **Compiler**: `g++ -O2`, `-lpthread` and `-fopenmp`
- **Image I/O**: hand-written BMP reader/writer (no external image library)
- **Measurement**: `std::chrono`, 3 runs averaged

## 6. Installation and Usage

**Requirements** — `g++` with OpenMP support. The program reads a BMP from the working
directory; no external libraries are needed.

```bash
g++ -O2 -o pthread_conv src/pthread/pthread.cpp -lpthread
g++ -O2 -fopenmp -o omp_conv src/openmp/omp.cpp

./pthread_conv 4          # thread count as argument
./omp_conv                # thread count via OMP_NUM_THREADS
```

The separable versions build the same way:

```bash
g++ -O2 -o pthread_conv_sep src/pthread/pthread_sep.cpp -lpthread
g++ -O2 -fopenmp -o omp_conv_sep src/openmp/omp_sep.cpp
```

```
OMP_NUM_THREADS=4 ./omp_conv_sep
```

The 5×5 Gaussian kernel is fixed in [`src/gaussian_kernel.txt`](src/gaussian_kernel.txt)
(σ = 1.5, radius 2), as specified by the assignment.

**Verification** — see [`figures/pthread-verify.png`](figures/pthread-verify.png) for the
correctness check against the reference output.

## 7. Limitations

- **Tested at up to 4 threads only.** No scaling study beyond 4 cores, so where the
  synchronization and scheduling overheads start to dominate is unknown.
- **A single test image.** Cache behavior depends heavily on image size relative to L2/L3;
  the Pthread/OpenMP reversal in particular may not hold for a much larger or much smaller image.
- **Static vs. dynamic scheduling is compared across paradigms, not within one.** Pthread is
  always static and OpenMP always dynamic here, so the comparison confounds the paradigm with
  the scheduling policy. Isolating the two would require a dynamic Pthread version.
- **The barrier cost is not separately measured.** The hand-written Pthread barrier is
  presumed more expensive than OpenMP's implicit one, but no experiment isolates it.
- **Zero padding is fixed.** No comparison against mirror or replicate padding, which would
  change both edge quality and the memory access pattern.
- **Only one kernel.** Separability is a property of the Gaussian specifically; the
  O(K²) → O(2K) argument does not generalize to arbitrary 5×5 kernels.

## Repository Structure

```
02-pthread-openmp-convolution/
├── src/
│   ├── pthread/
│   │   ├── pthread.cpp             # direct 2D, static row partitioning
│   │   └── pthread_sep.cpp         # separable, hand-written barrier
│   ├── openmp/
│   │   ├── omp.cpp                 # direct 2D, collapse(2) + schedule(dynamic)
│   │   └── omp_sep.cpp             # separable, implicit barrier
│   └── gaussian_kernel.txt         # 5×5 Gaussian kernel (σ = 1.5), instructor-provided
├── report/
│   ├── pthread.pdf                 # lab report (Chinese)
│   └── openmp.pdf                  # lab report (Chinese)
├── slides/presentation.pdf         # presentation (Chinese)
├── data/
│   ├── pthread-results.csv         # raw timing, 1/2/4 threads
│   ├── openmp-results.csv          # raw timing
│   └── comparison.csv              # paradigm comparison across dimensions
├── figures/                        # output images per core count, plus correctness check
└── docs/
    ├── convolution.html            # interactive: the sliding convolution window
    ├── convolution-optimized.html  # interactive: 2D → two 1D passes, stage by stage
    └── convolution-parallel.html   # interactive: row partitioning and load balancing
```

Reports and slides are written in Chinese.
