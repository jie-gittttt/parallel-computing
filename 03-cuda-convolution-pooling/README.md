# 作业 03 · 基于 CUDA 的 BMP 图像卷积池化

在 NVIDIA CUDA 上实现 5×5 高斯卷积 + 2×2 最大池化，图像分辨率 4096×2304。
重点在于理解 **GPU 的线程层次结构（Grid / Block / Thread）**，以及
**算子融合**如何突破显存带宽墙。

```bash
nvcc -O2 -o cuda_conv_pool src/cuda_conv_pool.cu
cd data && ../cuda_conv_pool        # 读 timg.bmp（4096×2304），输出同尺寸 BMP
```

三种算法实现：

| 文件 | 结构 | 显存中的中间结果 |
| :--- | :--- | :--- |
| [cuda_conv_pool.cu](src/cuda_conv_pool.cu) | 基础版，**两次 kernel 调用** | 完整的 4096×2304 卷积图 |
| [cuda_fused.cu](src/cuda_fused.cu) | **单核融合**，卷积 + 池化一个 kernel 完成 | 不落显存，全程留在寄存器 |
| [cuda_sep_opt.cu](src/cuda_sep_opt.cu) | **可分离卷积**（1×5 → 5×1）+ 池化 | 一张 float 中间图 + 最终卷积图 |

---

## 线程结构

```cpp
filterSize = 5;
dim3 block(16, 16);                                   // 256 线程 / block = 8 个 warp
dim3 grid((out_width  + block.x - 1) / block.x,
          (out_height + block.y - 1) / block.y);      // 向上取整，覆盖边界
```

用 16×16 的二维 block 而不是一维 256，是为了让线程在 **x 方向连续**——
这样同一个 warp 的 32 个线程访问的是相邻像素，可以合并成一次显存事务
（coalesced access）。如果按一维编号铺开再手工除模，很容易破坏这个连续性。

每个线程负责输出图的**一个像素**：先做 5×5 卷积，再把结果写进池化窗口。
网格按**输出**尺寸（池化后 = 原图的一半）计算，避免多算一半的线程。

---

## 优化一：融合 kernel

### 基础版的问题

```
convolution_kernel  ──写入──▶  d_conv（16 MB）  ──读出──▶  max_pool_kernel
```

卷积 kernel 把整张 4096×2304 的中间图写进**全局显存**，池化 kernel 再完整读回来。
这一来一回是 16 MB × 2 = 32 MB 的纯显存流量，而这些数据**只需要用一次**。

### 融合版

```cpp
__global__ void fused_conv2d_pool_kernel(const Pixel* __restrict__ input, ...) {
    // 1. 5×5 卷积，累加器放在寄存器里
    float sum = 0;
    for (int ky = 0; ky < 5; ++ky)
        for (int kx = 0; kx < 5; ++kx)
            sum += input[conv_y * width + conv_x] * d_KERNEL_2D[ky * 5 + kx];

    // 2. 直接比较，池化结果就地算出——中间值从未离开寄存器
    atomicMax(&output[...], (Pixel)sum);
}
```

中间结果全程待在**寄存器**里，只在最后写一次输出（尺寸还只有原图的 1/4）。
显存流量从 `读1 + 写1 + 读1 + 写0.25` 降到 `读1 + 写0.25`。

> 微观机制演示：[cuda-fusion.html](docs/cuda-fusion.html)

### 顺带解决的两件事

**常量内存**：卷积核权重放在 `__constant__ float d_KERNEL_2D[25]`，
用 `cudaMemcpyToSymbol` 从主机拷入。

```cpp
CHECK(cudaMemcpyToSymbol(d_KERNEL_2D, h_KERNEL_2D, 25 * sizeof(float)));
```

常量内存有专用的 **constant cache**：同一个 warp 内所有线程读**同一个地址**时，
硬件一次广播就完成，等价于零延迟。卷积核正好符合这个模式——
每个线程都在读同一组 25 个权重。放在全局内存里则会变成 256 次独立的访存。

**去掉 `__ldg`**：早期版本用 `__ldg()` 显式走只读缓存，实测反而不如让编译器自己决定。
最终改成 `const Pixel* __restrict__`，把「这个指针不会指向别处」的承诺交给编译器，
它自己就会选择最优的访存路径。

---

## 优化二：可分离卷积（cuda_sep_opt.cu）

和作业 02 是同一个思路：5×5 二维高斯核可分离，25 次乘加降到 10 次。

```cpp
__constant__ float d_KERNEL_1D[5];    // 一维核，5 个权重
row_conv_kernel<<<...>>>(d_src,   d_buffer,     width, height);  // 水平
col_conv_kernel<<<...>>>(d_buffer, d_conv_final, width, height); // 垂直
max_pool_kernel<<<...>>>(d_conv_final, d_pool, ...);
```

在 GPU 上做可分离要多算一笔账：中间结果需要一张**与图像等大的 float 缓冲**
（4096×2304×4 = 37.7 MB），加上多一次 kernel 启动和往返显存。
计算量省了 60%，但访存量**涨了**——在显存带宽受限的场景下不一定划算。
这正是作业 03 里没有把可分离列为最终方案的取舍。

---

## 结果

计时数据（3 次测量取平均，单位毫秒）：

| 版本 | Kernel 纯计算 | 整体（除 I/O） | 全流程（含 I/O） |
| :--- | --: | --: | --: |
| 基础版（两次 kernel） | 0.230 | 64.6 | 187.6 |
| 优化版（融合 kernel） | **0.180** | 68.6 | 191.1 |

> 上表中「优化版」为融合 kernel。可分离实现的单独测量记录在
> [report/report.pdf](report/report.pdf) 中。

**Kernel 本身提速 1.27×**（0.230 → 0.180 ms），来自省掉的 32 MB 显存往返。

但**整体时间和全流程几乎没有变化**，甚至在噪声范围内略微上升。原因很直接：

```
Kernel      0.23 ms   ← 优化掉 0.05 ms
主机侧       64.6 ms   ← 读取 27 MB BMP、Pixel 结构转换、内存分配
磁盘 I/O    123.0 ms   ← 读入 + 写出 27 MB 文件
──────────────────────
全流程      187.6 ms
```

优化掉的 0.05 ms 占全流程的 **0.03%**。这是 Amdahl 定律的一个教科书式例子：
把 0.1% 的部分做到无穷快，整体最多也就快 0.1%。

**这个结果本身就是结论**：算子融合是正确的优化（kernel 时间确实降了 27%），
但在真实管道中，**决定性能的是占 99.9% 的那部分**。要想真正提速，得先解决
BMP 的读写与主机侧转换，而不是继续打磨 kernel。

> 带宽视角的分析：[cuda-bandwidth.html](docs/cuda-bandwidth.html) ——
> 算术强度（Arithmetic Intensity）如何决定一个 kernel 是计算受限还是带宽受限。

---

## 文件结构

```
03-cuda-convolution-pooling/
├── src/
│   ├── cuda_conv_pool.cu           # 基础版：卷积、池化两次 kernel 调用
│   ├── cuda_fused.cu               # 优化版：单核融合，中间值不落显存
│   └── cuda_sep_opt.cu             # 可分离版：1×5 + 5×1 + 池化
├── report/report.pdf               # 实验报告
├── slides/presentation.pdf         # 汇报 PPT
├── data/
│   ├── results.csv                 # 计时数据（3 次测量取平均）
│   └── timg.bmp                    # 测试图像 4096×2304（教师提供）
├── figures/
│   ├── baseline.png                # 基础版输出
│   ├── optimized.png               # 优化版输出
│   └── verified.png                # 正确性验证图
└── docs/
    ├── cuda-pipeline.html          # 异构计算全流程：Host ↔ Device 数据流转
    ├── cuda-mapping.html           # blockIdx / threadIdx → 全局内存坐标（含 Warp）
    ├── cuda-fusion.html            # 算子融合微观机制：访存 4× → 1×
    ├── cuda-bandwidth.html         # 显存带宽 vs 算术强度
    └── media/                      # 两份录屏（上述页面的屏幕录制）
        ├── cuda-mapping.mp4
        └── cuda-fusion.mp4
```
