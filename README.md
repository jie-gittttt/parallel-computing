# Parallel and Distributed Systems — Course Projects

[English](README.md) | [中文](README.zh-CN.md)

Three projects covering the three standard parallel paradigms — **MPI** message passing,
**Pthread / OpenMP** shared memory, and **CUDA** heterogeneous computing.

Each one is built around the same question: *how far does the obvious optimization actually
get you, and where is the real bottleneck?* Answering that honestly is the point of the
project; two of the three end with a result that contradicts the naive expectation.

| # | Project | Parallel model | Key optimization | Headline result |
| :-: | :--- | :--- | :--- | :--- |
| [01](01-mpi-matrix-multiplication/) | Matrix multiplication `D = A×B + C`, 1023×1023 | MPI message passing | i-k-j loop interchange, B transpose | **8.57×** end-to-end (10.494 s → 1.225 s) |
| [02](02-pthread-openmp-convolution/) | 5×5 Gaussian convolution of BMP images | Pthread / OpenMP | Separable convolution, O(K²) → O(2K) | **2.6×** single-threaded from separability alone |
| [03](03-cuda-convolution-pooling/) | Convolution + 2×2 max pooling, 4096×2304 | CUDA | Kernel fusion, constant memory | Kernel **1.27×** faster — but only 0.03% of runtime |

## Results at a glance

**01 · MPI** — memory access pattern matters more than you'd expect

| Processes | Naive i-j-k | After loop interchange |
| :-: | --: | --: |
| 1 | 10.494 s | 7.710 s |
| 2 | 6.659 s | 3.946 s |
| 4 | 3.398 s | **1.225 s** |

The serial optimization alone is a 1.36× win, purely from cache behavior. The 4-process
speedup goes from 3.09× to **6.29×** — superlinear, because splitting the matrix shrinks each
process's working set. Verification passed the instructor's `verify_` checker on every run.

**02 · Pthread / OpenMP** — the faster algorithm also changes which scheduler wins

| Implementation | 1 thread | 4 threads | 4-thread speedup |
| :--- | --: | --: | --: |
| Pthread, direct 2D | 1.043 s | 0.323 s | 3.23× |
| Pthread, separable | 0.400 s | 0.128 s | 3.12× |
| OpenMP, direct 2D | 1.053 s | 0.370 s | 2.84× |
| OpenMP, separable | 0.392 s | 0.111 s | **3.52×** |

Separable convolution cuts 25 multiply-adds per pixel to 10, matching the predicted 2.5×.
The interesting part is the reversal: Pthread's static partitioning wins on the heavier
direct version, OpenMP's dynamic scheduling wins once the per-iteration work drops.

**03 · CUDA** — a correct optimization that changes nothing

| Version | Kernel only | Excluding I/O | Full pipeline |
| :--- | --: | --: | --: |
| Baseline (two kernels) | 0.230 ms | 64.6 ms | 187.6 ms |
| Fused (single kernel) | **0.180 ms** | 68.6 ms | 191.1 ms |

Kernel fusion eliminates a 32 MB round trip to global memory and makes the kernel 27%
faster. It is also invisible end-to-end: 0.05 ms out of 187.6 ms. BMP parsing and disk I/O
dominate the pipeline, which is the actual conclusion of the assignment.

## Interactive visualizations

Nine self-written HTML pages step through the algorithms frame by frame — cache line reuse,
sliding convolution windows, thread-to-pixel mapping, warp-level access patterns.

**→ [jie-gittttt.github.io/parallel-computing](https://jie-gittttt.github.io/parallel-computing/)**

These are static pages and need no server, but they load Tailwind, React and Babel from
public CDNs, so they do require a network connection to render.

## Running the code

```bash
# 01 · MPI
mpic++ -o Matrix src/Matrix.cpp -std=c++11
mpirun -np 4 ./Matrix

# 02 · Pthread / OpenMP
g++ -O2 -o pthread_conv src/pthread/pthread.cpp -lpthread
g++ -O2 -fopenmp -o omp_conv src/openmp/omp.cpp
./pthread_conv 4

# 03 · CUDA
nvcc -O2 -o cuda_conv_pool src/cuda_conv_pool.cu
cd 03-cuda-convolution-pooling/data && ../../cuda_conv_pool
```

Each subdirectory README has full environment requirements, parameters and expected output.

## Repository layout

```
parallel-computing/
├── README.md                           # this file
├── index.html                          # landing page for the interactive demos
├── 01-mpi-matrix-multiplication/       # MPI · D = A×B + C
├── 02-pthread-openmp-convolution/      # Pthread / OpenMP · 5×5 Gaussian convolution
└── 03-cuda-convolution-pooling/        # CUDA · convolution + max pooling
```

## Notes on the material

- **Provided by the instructor, not included**: the `verify_` result checker (01) and the
  test suite with black/white block images (02).
- **Provided by the instructor, included**: the standard Gaussian kernel
  (`src/gaussian_kernel.txt`) and the test image (`data/timg.bmp`) — both are required
  inputs, and the code will not run without them.
- Timing for project 01 comes from a shared course server. Absolute numbers vary between
  sessions with machine load; the raw output of three separate runs is preserved in
  `01-mpi-matrix-multiplication/results/run-log.txt`.
- All source code, reports, slides and visualizations are my own work.
