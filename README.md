# 并行与分布式系统 · 课程实验

> Three course projects on parallel computing, covering three different parallel paradigms:
> **MPI** message passing, **Pthread / OpenMP** shared memory, and **CUDA** heterogeneous computing.

三次实验，三种并行范式：从分布式消息传递，到共享内存多线程，再到 GPU 异构计算。
每份作业包含源码、实验报告、汇报 PPT、计时数据，以及可交互的原理演示页面。

---

## 实验一览

| # | 主题 | 并行模型 | 关键优化 |
| :-: | :--- | :--- | :--- |
| [01](01-mpi-matrix-multiplication/) | 矩阵运算 `D = A×B + C`（1023×1023） | MPI 消息传递 | i-k-j 循环交换、B 转置 |
| [02](02-pthread-openmp-convolution/) | BMP 图像高斯卷积（5×5, σ=1.5） | Pthread / OpenMP | 可分离卷积，复杂度 O(K²) → O(2K) |
| [03](03-cuda-convolution-pooling/) | 卷积 + 2×2 最大池化（4096×2304） | CUDA | 算子融合、常量内存、`__ldg` |

## 结果概览

**01 · MPI 矩阵运算**（1023×1023，课程远程服务器）

| 进程数 | 优化前 | 优化后 |
| :-: | --: | --: |
| 1 | 10.494 s | 7.710 s |
| 2 | 6.659 s | 3.946 s |
| 4 | 3.398 s | **1.225 s** |

4 进程加速比由 **3.09×** 提升至 **6.29×**；相对优化前单进程的端到端总提升 **8.57×**。
每次运行均通过助教提供的 `verify_` 校验程序（`比对成功！`）。

**02 · Pthread / OpenMP 卷积**（4 线程）

| 实现 | 单线程 | 4 线程 | 4 线程加速比 |
| :--- | --: | --: | --: |
| Pthread 基础版 | 1.043 s | 0.323 s | 3.23× |
| Pthread 可分离版 | 0.400 s | 0.128 s | 3.12× |
| OpenMP 基础版 | 1.053 s | 0.370 s | 2.84× |
| OpenMP 可分离版 | 0.392 s | 0.111 s | 3.52× |

可分离卷积把每像素 25 次乘加降到 10 次，**单线程即提速 2.6×**，与理论值 2.5× 吻合。

**03 · CUDA 卷积池化**（NVIDIA GPU，4096×2304）

| 版本 | Kernel 纯计算耗时 |
| :--- | --: |
| 基础版（卷积 + 池化两次访存） | 0.230 ms |
| 优化版（融合 kernel + 常量内存） | **0.180 ms** |

算子本身提速约 **1.27×**。全流程 170–210 ms 由 BMP 读写主导，端到端差异落在测量噪声内——
这一点在 [作业3 的 README](03-cuda-convolution-pooling/) 里有说明。

> **关于计时**：作业 01 运行在课程共享远程服务器上，不同时段负载不同。`results/run-log.txt`
> 保留了同一份代码在三次不同会话中的完整输出，可以看到绝对值存在波动；表中取的是最后一次
> 带校验记录的完整运行。

---

## 可交互原理演示

10 个自写的 HTML 页面，用 SVG / Canvas 把三次实验的算法过程逐帧展开。
纯静态、不需要服务器，克隆下来**双击即可打开**；但页面依赖几个公开 CDN
（Tailwind CSS、React 18、Babel standalone、Lucide 图标），**需要联网才能渲染**：

| 演示 | 说明 |
| :--- | :--- |
| [mpi-loop-interchange](01-mpi-matrix-multiplication/docs/mpi-loop-interchange.html) | i-k-j 循环交换为什么比 i-j-k 快：逐格展示访存顺序 |
| [mpi-transpose](01-mpi-matrix-multiplication/docs/mpi-transpose.html) | 7×8 缩微矩阵，对比原版与 B 转置版的计算流程 |
| [convolution](02-pthread-openmp-convolution/docs/convolution.html) | 卷积过程：像素窗口如何在图像上滑动 |
| [convolution-optimized](02-pthread-openmp-convolution/docs/convolution-optimized.html) | 二维卷积拆成水平 + 垂直两趟，逐阶段对照 |
| [convolution-parallel](02-pthread-openmp-convolution/docs/convolution-parallel.html) | 按行划分给多个线程，展示负载均衡问题 |
| [cuda-pipeline](03-cuda-convolution-pooling/docs/cuda-pipeline.html) | 异构计算全流程：Host ↔ Device 数据流转 |
| [cuda-mapping](03-cuda-convolution-pooling/docs/cuda-mapping.html) | `blockIdx` / `threadIdx` 如何映射到全局内存坐标，含 Warp 可视化 |
| [cuda-fusion](03-cuda-convolution-pooling/docs/cuda-fusion.html) | 算子融合微观机制：寄存器里完成卷积 + 池化，访存 4× → 1× |
| [cuda-bandwidth](03-cuda-convolution-pooling/docs/cuda-bandwidth.html) | 显存带宽 vs 算术强度：为什么融合能突破带宽墙 |

另外两份演示录屏（[cuda-mapping.mp4](03-cuda-convolution-pooling/docs/media/cuda-mapping.mp4)、
[cuda-fusion.mp4](03-cuda-convolution-pooling/docs/media/cuda-fusion.mp4)）是上述页面的屏幕录制，
放在 `docs/media/` 下作为存档。

> 想在线点开它们，在 **Settings → Pages** 里把 Source 设为当前分支的根目录即可，
> 之后每个页面就能通过 `https://<用户名>.github.io/parallel-computing/<路径>` 直接访问。

---

## 文件结构

```
parallel-computing/
├── README.md
├── index.html                          # 演示导航页（GitHub Pages 入口）
│
├── 01-mpi-matrix-multiplication/       # MPI 消息传递 · D = A×B + C（1023×1023）
│   ├── src/                            # Matrix.cpp（含循环变换优化）
│   ├── report/                         # 实验报告 + 输入输出格式说明
│   ├── slides/                         # 优化方案 PPT（PDF）
│   ├── results/                        # 三次会话的完整终端输出与计时汇总
│   └── docs/                           # 2 个交互演示
│
├── 02-pthread-openmp-convolution/      # Pthread / OpenMP · 5×5 高斯卷积
│   ├── src/pthread/                    # pthread.cpp、pthread_sep.cpp
│   ├── src/openmp/                     # omp.cpp、omp_sep.cpp
│   ├── src/gaussian_kernel.txt         # 作业指定的高斯核（σ = 1.5）
│   ├── report/                         # Pthread、OpenMP 两份实验报告
│   ├── slides/                         # 汇报 PPT（PDF）
│   ├── data/                           # 计时数据与范式对比表（CSV）
│   ├── figures/                        # 各核数下的输出图与正确性验证图
│   └── docs/                           # 3 个交互演示
│
└── 03-cuda-convolution-pooling/        # CUDA · 卷积 + 2×2 最大池化（4096×2304）
    ├── src/                            # cuda_conv_pool.cu、cuda_fused.cu、cuda_sep_opt.cu
    ├── report/                         # 实验报告
    ├── slides/                         # 汇报 PPT（PDF）
    ├── data/                           # 计时数据（CSV）、测试图像 timg.bmp
    ├── figures/                        # 基础版 / 优化版输出与正确性验证图
    └── docs/                           # 4 个交互演示 + media/ 录屏
```

## 运行方法

**作业 01 · MPI**

```bash
mpic++ -o Matrix src/Matrix.cpp -std=c++11
mpirun -np 4 ./Matrix          # 读 RandMatrix.txt，输出 result.txt
```

**作业 02 · Pthread / OpenMP**

```bash
g++ -O2 -o pthread_conv src/pthread/pthread.cpp -lpthread
g++ -O2 -fopenmp -o omp_conv src/openmp/omp.cpp
./pthread_conv 4                # 参数为线程数
```

**作业 03 · CUDA**

```bash
nvcc -O2 -o cuda_conv_pool src/cuda_conv_pool.cu
cd 03-cuda-convolution-pooling/data && ../../cuda_conv_pool   # 需在 data/ 下运行，读 timg.bmp
```

## 关于素材来源

- **教师提供、未包含**：作业 01 的 `verify_` 结果校验程序、作业 02 的测试程序套件
  （`测试程序/`，含 `test.cpp` / `test_blur.py` 与黑白色块测试图）。
- **教师提供、已包含**：标准高斯核（`src/gaussian_kernel.txt`）与测试图像（`data/timg.bmp`）——
  这两份是程序运行必需的输入，缺了代码跑不起来。
- 所有源码、实验报告、汇报 PPT、计时数据、图表与 HTML 演示均为本人独立完成。
