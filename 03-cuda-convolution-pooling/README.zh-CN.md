# CUDA 卷积 + 最大池化

[English](README.md) | [中文](README.zh-CN.md)

在 4096×2304 的图像上做 5×5 高斯卷积 + 2×2 最大池化。算子融合让 GPU kernel 快了 27%，
而端到端耗时**一点没变**。

## 1. 概述

任务是在 GPU 上实现卷积 + 池化并做优化。图像为 4096×2304 的 BMP，核是固定的 5×5 高斯核，
池化为 2×2 最大池化——所以输出图的宽高各是输入的一半。

这次作业有意把两个最终互相冲突的目标放在一起：

1. **理解 GPU 的线程层次结构** —— `grid`/`block`/`thread` 如何映射到图像坐标、
   为什么 block 的形状会影响访存合并、常量内存在哪里帮得上忙。
2. **诚实地测量。** 算子融合在这里是教科书式正确的优化。有意思的结果是它对整体耗时做了什么——
   而这正是 Amdahl 定律不再是抽象概念的地方。

## 2. 核心贡献

- **三份递进的实现**全部收录在仓库里：两次 kernel 的基础版、单核融合版、可分离版——
  每个想法的贡献都能单独看出。
- **算子融合**，把卷积的中间结果留在寄存器里、从不写入全局显存，省掉 32 MB 的往返流量。
- **用常量内存放卷积核权重**（`__constant__ float d_KERNEL_2D[25]` + `cudaMemcpyToSymbol`），
  同一个 warp 内广播同一组权重，代价接近于零。
- **一个被量化的否定结果。** 试过 `__ldg()` 并把它去掉了；`const Pixel* __restrict__` 表现更好。
  仓库里记录的是试过并否决了什么，而不只是最后用了什么。
- **管道级的耗时拆解**，把 kernel、主机侧转换、磁盘 I/O 分开——这才让 Amdahl 定律的结论是定量的，
  而不是一句口号。

## 3. 方法与架构

### 网格与线程块配置

```cpp
filterSize = 5;
dim3 block(16, 16);                                   // 256 线程 / block = 8 个 warp
dim3 grid((out_width  + block.x - 1) / block.x,
          (out_height + block.y - 1) / block.y);      // 向上取整，覆盖边界
```

用 16×16 的二维 block 而不是一维 256，是为了让线程在 **x 方向连续**——
这样同一个 warp 的 32 个线程访问的是相邻像素，可以合并成一次显存事务。
按一维编号铺开再手工除模，很容易破坏这个连续性。

每个线程负责输出图的**一个像素**：先做 5×5 卷积，再把结果折进池化窗口。
网格按**输出**尺寸（池化后 = 原图的一半）计算，避免启动那些池化时会被丢弃的线程。

### 基础版：两次 kernel

```
  输入 ──▶ convolution_kernel ──▶ d_conv（16 MB，全局显存）──▶ max_pool_kernel ──▶ 输出
```

卷积 kernel 把整张 4096×2304 的中间图写进**全局显存**，池化 kernel 再完整读回来。
这一来一回是 **32 MB 的纯显存流量**，而这些数据**只需要用一次**。

### 优化版：融合成一个 kernel

```
  输入 ──▶ fused_conv2d_pool_kernel ──▶ 输出（像素数为 1/4）
                    │
                    └── 卷积累加器留在寄存器里，中间值从不落到全局显存
```

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

中间的显存流量从 `读1 + 写1 + 读1 + 写0.25` 降到 `读1 + 写0.25`，
而且输出图本身就只有原图的四分之一大。

### 常量内存

```cpp
CHECK(cudaMemcpyToSymbol(d_KERNEL_2D, h_KERNEL_2D, 25 * sizeof(float)));
```

常量内存有专用的 **constant cache**：同一个 warp 内所有线程读**同一个地址**时，
硬件一次广播就完成，等价于零延迟。卷积核正好符合这个模式——
每个线程都在读同一组 25 个权重。放在全局内存里则会变成每个 block 256 次独立的访存。

### 试过但否决的：`__ldg`

早期版本用 `__ldg()` 显式走只读缓存，实测反而不如让编译器自己决定。
最终改成 `const Pixel* __restrict__`，把「这个指针不会指向别处」的承诺交给编译器，
它自己就会选择最优的访存路径。

### 可分离版

和作业 02 是同一个思路：5×5 二维高斯核可分离，25 次乘加降到 10 次。

```cpp
__constant__ float d_KERNEL_1D[5];
row_conv_kernel<<<...>>>(d_src,   d_buffer,     width, height);  // 水平
col_conv_kernel<<<...>>>(d_buffer, d_conv_final, width, height); // 垂直
max_pool_kernel<<<...>>>(d_conv_final, d_pool, ...);
```

在 GPU 上做可分离要多算一笔账：中间结果需要一张**与图像等大的 float 缓冲**
（4096×2304×4 = 37.7 MB），加上多一次 kernel 启动和它自己的一次显存往返。
计算量省了 60%，但**访存量涨了**——在带宽受限的场景下这是反方向。
这正是作业 03 里没有把可分离列为最终方案的取舍，与作业 02 的结论相反。

## 4. 实验与结果

3 次测量取平均，单位毫秒：

| 版本 | 仅 Kernel | 除 I/O 外 | 全流程 |
| :--- | --: | --: | --: |
| 基础版（两次 kernel） | 0.230 | 64.6 | 187.6 |
| 优化版（融合 kernel） | **0.180** | 68.6 | 191.1 |

**Kernel 本身提速 1.27×**（0.230 → 0.180 ms），来自省掉的 32 MB 显存往返。

**整体时间几乎没有变化**——在噪声范围内甚至略微上升。原因很直接：

```
Kernel      0.23 ms   ← 优化掉的那 0.05 ms
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

> 带宽视角的分析：[cuda-bandwidth.html](https://jie-gittttt.github.io/parallel-computing/03-cuda-convolution-pooling/docs/cuda-bandwidth.html) ——
> 算术强度（Arithmetic Intensity）如何决定一个 kernel 是计算受限还是带宽受限。

## 5. 技术栈

- **语言**：CUDA C++
- **构建**：`nvcc -O2`，单编译单元
- **GPU**：NVIDIA 单卡
- **图像 I/O**：自写的 BMP 读写——不依赖图像库，所以 `data/timg.bmp` 保持原始格式
- **计时**：设备侧用 CUDA event，主机侧用 wall clock

## 6. 安装与运行

**环境要求** —— 带 `nvcc` 的 CUDA 工具链与一块 NVIDIA GPU。程序从工作目录读取
`timg.bmp`，输出同尺寸的 BMP。

```bash
nvcc -O2 -o cuda_conv_pool src/cuda_conv_pool.cu
cd data && ../cuda_conv_pool
```

融合版（从仓库根目录）：

```bash
nvcc -O2 -o cuda_fused src/cuda_fused.cu
cd 03-cuda-convolution-pooling/data && ../../cuda_fused
```

**必须在 `data/` 目录内运行**，因为程序按相对路径加载 `timg.bmp`——
BMP 加载器是手写的，文件名固定。

**预期输出** —— 分阶段计时（kernel、主机侧、I/O）与总时间，并在工作目录写出结果 BMP。

**正确性验证** —— [`figures/verified.png`](figures/verified.png) 是融合版输出与基础版输出的比对；
[`figures/baseline.png`](figures/baseline.png) 与
[`figures/optimized.png`](figures/optimized.png) 是两个结果并排。

## 7. 局限性

- **优化是真实的，但在端到端上可以忽略。** 0.03% 这个数字就是诚实的总结。
  管道由 BMP 解析和磁盘 I/O 主导，而这些都没有被触及——虽然超出本次作业范围，
  但那才是真正的优化空间所在。
- **用 `atomicMax` 写输出不是最好的池化实现。** 池化窗口内的每个线程都对同一个全局地址发原子操作。
  改成 block 级归约，或先做 warp shuffle 归约再每 warp 发一次原子操作，能大幅降低竞争。
- **没有用共享内存做分块（tiling）。** 卷积时每个线程都从全局显存重新读取 5×5 邻域；
  在共享内存里放一块带重叠的 tile 可以减少冗余的全局读取。未实现。
- **没有异步拷贝或 stream。** 主机侧的「读取 → 转换 → 上传」完全是串行的，
  而这正是占主导的部分。`cudaMemcpyAsync` 配合锁页内存做重叠计算，才是打在正确瓶颈上的手段。
- **单卡、单一图像尺寸。** 没有多 GPU 或多 stream 的测量，block 与 tile 配置也没有在初始的
  16×16 之外做过调优。
- **可分离版的测量不如另外两版充分**，它的计时记录在报告里，但没有进入主要对比表。
- **kernel 耗时接近测量分辨率。** 0.18 ms 与 0.23 ms 都足够小，
  启动开销和 event 计时精度在其中的占比不可忽略。

## 目录结构

```
03-cuda-convolution-pooling/
├── src/
│   ├── cuda_conv_pool.cu           # 基础版：卷积、池化两次 kernel 调用
│   ├── cuda_fused.cu               # 优化版：单核融合
│   └── cuda_sep_opt.cu             # 可分离版：1×5 + 5×1 + 池化
├── report/report.pdf               # 实验报告
├── slides/presentation.pdf         # 汇报 PPT
├── data/
│   ├── results.csv                 # 计时数据（3 次测量取平均）
│   └── timg.bmp                    # 测试图像 4096×2304（教师提供）
├── figures/
│   ├── baseline.png                # 基础版输出
│   ├── optimized.png               # 优化版输出
│   └── verified.png                # 与基础版的一致性验证
└── docs/
    ├── cuda-pipeline.html          # 交互演示：异构计算全流程，Host ↔ Device 数据流转
    ├── cuda-mapping.html           # 交互演示：blockIdx/threadIdx → 全局坐标
    ├── cuda-fusion.html            # 交互演示：融合如何把访存 4× 降到 1×
    ├── cuda-bandwidth.html         # 交互演示：显存带宽 vs. 算术强度
    └── media/                      # 上述页面的录屏
        ├── cuda-mapping.mp4
        └── cuda-fusion.mp4
```
