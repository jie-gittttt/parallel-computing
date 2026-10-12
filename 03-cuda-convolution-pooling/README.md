# CUDA Convolution + Max Pooling

[English](README.md) | [中文](README.zh-CN.md)

A 5×5 Gaussian convolution followed by 2×2 max pooling on a 4096×2304 image, where kernel
fusion makes the GPU kernel 27% faster and changes the end-to-end runtime not at all.

## 1. Overview

The task is to implement convolution + pooling on the GPU and then optimize it. The image is
4096×2304 BMP, the kernel is a fixed 5×5 Gaussian, and pooling is 2×2 max — so the output is
half the width and half the height of the input.

The assignment deliberately pairs two goals that turn out to be in tension:

1. **Understand the GPU thread hierarchy** — how `grid`/`block`/`thread` map onto image
   coordinates, why the block shape matters for memory coalescing, and where constant memory
   helps.
2. **Measure honestly.** Kernel fusion is a textbook-correct optimization here. The
   interesting result is what it does to the overall runtime, which is where Amdahl's law
   stops being an abstraction.

## 2. Key Features / Contributions

- **Three implementations at increasing levels of optimization**, all in the repository:
  a two-kernel baseline, a single fused kernel, and a separable variant — so the contribution
  of each idea can be read off independently.
- **Operator fusion** that keeps the intermediate convolution result in registers and never
  writes it to global memory, eliminating a 32 MB round trip.
- **Constant memory for the kernel weights** (`__constant__ float d_KERNEL_2D[25]` with
  `cudaMemcpyToSymbol`), so the warp-wide broadcast of the same weights costs effectively
  nothing.
- **A measured negative result.** `__ldg()` was tried and removed; `const Pixel* __restrict__`
  performed better. The repository documents what was tried and rejected, not only what worked.
- **A pipeline-level breakdown** that separates kernel time, host-side conversion and disk
  I/O, which is what makes the Amdahl's law conclusion quantitative rather than hand-waving.

## 3. Methodology / Architecture

### Grid and block configuration

```cpp
filterSize = 5;
dim3 block(16, 16);                                   // 256 threads/block = 8 warps
dim3 grid((out_width  + block.x - 1) / block.x,
          (out_height + block.y - 1) / block.y);      // round up to cover the border
```

A 2D 16×16 block is used instead of a flat 256 so that threads stay **contiguous in x**. The
32 threads of a warp then read adjacent pixels and the accesses coalesce into a single memory
transaction. Laying threads out linearly and recovering x/y by division and modulo tends to
break that contiguity.

Each thread produces **one output pixel**: it performs the 5×5 convolution, then folds the
result into its pooling window. The grid is sized from the **output** dimensions (half the
input, since pooling halves each axis), so no thread is launched for work that pooling
would discard.

### Baseline: two kernels

```
  input ──▶ convolution_kernel ──▶ d_conv (16 MB, global) ──▶ max_pool_kernel ──▶ output
```

The convolution kernel writes the entire 4096×2304 intermediate image to global memory, and
the pooling kernel reads all of it back. That is **32 MB of pure global memory traffic for
data that is used exactly once**.

### Optimized: fused into one kernel

```
  input ──▶ fused_conv2d_pool_kernel ──▶ output (¼ the pixels)
                    │
                    └── convolution accumulator lives in a register;
                        the intermediate never reaches global memory
```

```cpp
__global__ void fused_conv2d_pool_kernel(const Pixel* __restrict__ input, ...) {
    // 1. 5×5 convolution, accumulator in a register
    float sum = 0;
    for (int ky = 0; ky < 5; ++ky)
        for (int kx = 0; kx < 5; ++kx)
            sum += input[conv_y * width + conv_x] * d_KERNEL_2D[ky * 5 + kx];

    // 2. pooling done in place — the intermediate never leaves the register
    atomicMax(&output[...], (Pixel)sum);
}
```

Memory traffic for the intermediate drops from `read 1 + write 1 + read 1 + write 0.25` to
`read 1 + write 0.25`, and the output image is already a quarter of the input size.

### Constant memory

```cpp
CHECK(cudaMemcpyToSymbol(d_KERNEL_2D, h_KERNEL_2D, 25 * sizeof(float)));
```

Constant memory has a dedicated constant cache: when every thread in a warp reads the **same
address**, the hardware broadcasts it in a single transaction — effectively zero latency. The
convolution weights are exactly this pattern, since all threads read the same 25 values.
Held in global memory instead, it would become 256 independent loads per block.

### What was tried and rejected: `__ldg`

An early version used `__ldg()` to route reads explicitly through the read-only cache.
Measured, it was slower than letting the compiler decide. The final code uses
`const Pixel* __restrict__`, which gives the compiler the aliasing guarantee it needs and
lets it choose the access path itself.

### The separable variant

The same idea as project 02: the 5×5 Gaussian is separable, so 25 multiply-adds per pixel
become 10.

```cpp
__constant__ float d_KERNEL_1D[5];
row_conv_kernel<<<...>>>(d_src,   d_buffer,     width, height);  // horizontal
col_conv_kernel<<<...>>>(d_buffer, d_conv_final, width, height); // vertical
max_pool_kernel<<<...>>>(d_conv_final, d_pool, ...);
```

On a GPU the accounting changes. The intermediate needs a full-size float buffer
(4096 × 2304 × 4 = 37.7 MB) plus an extra kernel launch and its own round trip to global
memory. Compute drops 60%, but **memory traffic goes up** — which is the wrong direction on a
bandwidth-bound workload. This is why separability is not the final approach for this
assignment, in contrast to project 02.

## 4. Experiments and Results

Mean of 3 measurements, milliseconds:

| Version | Kernel only | Excluding I/O | Full pipeline |
| :--- | --: | --: | --: |
| Baseline (two kernels) | 0.230 | 64.6 | 187.6 |
| Optimized (fused kernel) | **0.180** | 68.6 | 191.1 |

**The kernel itself is 1.27× faster** (0.230 → 0.180 ms), from eliminating the 32 MB
round trip.

**The overall time barely moves** — and within noise, it is slightly worse. The reason is
straightforward:

```
Kernel       0.23 ms   ← the 0.05 ms that was optimized away
Host side   64.6  ms   ← reading the 27 MB BMP, Pixel struct conversion, allocation
Disk I/O   123.0  ms   ← reading and writing the 27 MB file
──────────────────────
Full       187.6  ms
```

The 0.05 ms saved is **0.03% of the full pipeline**. This is a textbook Amdahl's law result:
make a 0.1% component infinitely fast and the total improves by at most 0.1%.

**That result is the conclusion.** Kernel fusion is the correct optimization — it really does
cut kernel time by 27%. But in a real pipeline, **what determines performance is the 99.9%
around it**. Further speedup requires attacking BMP parsing and host-side conversion, not
polishing the kernel.

> Bandwidth perspective: [cuda-bandwidth.html](https://jie-gittttt.github.io/parallel-computing/03-cuda-convolution-pooling/docs/cuda-bandwidth.html) —
> how arithmetic intensity decides whether a kernel is compute-bound or bandwidth-bound.

## 5. Tech Stack

- **Language**: CUDA C++
- **Build**: `nvcc -O2`, single translation unit
- **GPU**: NVIDIA (single device)
- **Image I/O**: hand-written BMP reader/writer — no image library, so `data/timg.bmp` is
  kept in its original format
- **Timing**: device timing via CUDA events, host timing via wall clock

## 6. Installation and Usage

**Requirements** — a CUDA toolkit with `nvcc` and an NVIDIA GPU. The program reads
`timg.bmp` from the working directory and writes a same-size BMP.

```bash
nvcc -O2 -o cuda_conv_pool src/cuda_conv_pool.cu
cd data && ../cuda_conv_pool
```

From the repository root, for the fused version:

```bash
nvcc -O2 -o cuda_fused src/cuda_fused.cu
cd 03-cuda-convolution-pooling/data && ../../cuda_fused
```

The program must run from inside `data/` because it loads `timg.bmp` by relative path — the
BMP loader is hand-written and takes a fixed filename.

**Expected output** — per-stage timing lines (kernel, host-side, I/O) followed by the total,
and an output BMP written to the working directory.

**Verification** — [`figures/verified.png`](figures/verified.png) shows the fused output
against the baseline output; [`figures/baseline.png`](figures/baseline.png) and
[`figures/optimized.png`](figures/optimized.png) are the two results side by side.

## 7. Limitations

- **The optimization is real but negligible end-to-end.** The 0.03% figure is the honest
  summary. The pipeline is dominated by BMP parsing and disk I/O, and none of that was
  addressed — which was outside the assignment's scope but is where the actual headroom is.
- **`atomicMax` on the output is not the best pooling implementation.** Every thread in a
  pooling window issues an atomic operation to the same global address. A block-level
  reduction, or warp shuffle reduction followed by one atomic per warp, would reduce
  contention substantially.
- **No shared-memory tiling.** The convolution re-reads a 5×5 neighborhood from global memory
  per thread; an overlapping tile in shared memory would cut redundant global reads. Not
  implemented.
- **No asynchronous copy or streams.** The host-side read → convert → upload sequence is fully
  serialized, which is precisely the part that dominates. `cudaMemcpyAsync` plus pinned memory
  and overlapping computation would target the right bottleneck.
- **Single GPU, single image size.** No multi-GPU or multi-stream measurement, and the tile
  and block configuration was not tuned beyond the initial 16×16 choice.
- **The separable variant was not benchmarked as thoroughly** as the two main versions; its
  timing is recorded in the report but is not part of the headline comparison.
- **Kernel times are close to measurement resolution.** 0.18 ms and 0.23 ms are small enough
  that launch overhead and event-timing granularity are a meaningful fraction of the number.

## Repository Structure

```
03-cuda-convolution-pooling/
├── src/
│   ├── cuda_conv_pool.cu           # baseline: convolution and pooling as two kernels
│   ├── cuda_fused.cu               # optimized: single fused kernel
│   └── cuda_sep_opt.cu             # separable: 1×5 + 5×1 + pooling
├── report/report.pdf               # lab report (Chinese)
├── slides/presentation.pdf         # presentation (Chinese)
├── data/
│   ├── results.csv                 # timing data (mean of 3 runs)
│   └── timg.bmp                    # test image 4096×2304 (instructor-provided)
├── figures/
│   ├── baseline.png                # baseline output
│   ├── optimized.png               # fused output
│   └── verified.png                # correctness check against the baseline
└── docs/
    ├── cuda-pipeline.html          # interactive: host ↔ device data flow
    ├── cuda-mapping.html           # interactive: blockIdx/threadIdx → global coordinates
    ├── cuda-fusion.html            # interactive: how fusion cuts memory traffic 4× → 1×
    ├── cuda-bandwidth.html         # interactive: bandwidth vs. arithmetic intensity
    └── media/                      # screen recordings of the pages above
        ├── cuda-mapping.mp4
        └── cuda-fusion.mp4
```

Reports and slides are written in Chinese.
