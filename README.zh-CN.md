# 并行与分布式系统 · 课程实验

[English](README.md) | [中文](README.zh-CN.md)

三次实验，覆盖三种标准并行范式：**MPI** 消息传递、**Pthread / OpenMP** 共享内存、
**CUDA** 异构计算。

三次实验都围绕同一个问题展开：*那个看起来最显然的优化，实际能带来多少收益？真正的瓶颈在哪里？*
诚实地回答这个问题才是作业的重点——其中两次实验的结论与直觉相反。

| # | 项目 | 并行模型 | 关键优化 | 主要成果 |
| :-: | :--- | :--- | :--- | :--- |
| [01](01-mpi-matrix-multiplication/) | 矩阵运算 `D = A×B + C`，1023×1023 | MPI 消息传递 | i-k-j 循环交换、B 转置 | 端到端 **8.57×**（10.494 s → 1.225 s） |
| [02](02-pthread-openmp-convolution/) | BMP 图像 5×5 高斯卷积 | Pthread / OpenMP | 可分离卷积，O(K²) → O(2K) | 仅靠可分离就单线程提速 **2.6×** |
| [03](03-cuda-convolution-pooling/) | 卷积 + 2×2 最大池化，4096×2304 | CUDA | 算子融合、常量内存 | Kernel 快 **1.27×**，但只占全流程 0.03% |

## 结果速览

**01 · MPI** —— 访存模式的影响比想象中大得多

| 进程数 | 朴素 i-j-k | 循环变换后 |
| :-: | --: | --: |
| 1 | 10.494 s | 7.710 s |
| 2 | 6.659 s | 3.946 s |
| 4 | 3.398 s | **1.225 s** |

串行优化本身就有 1.36× 的收益，纯粹来自缓存行为改善。4 进程加速比从 3.09× 提升到 **6.29×**——
超线性，因为矩阵被切开后每个进程的工作集变小了。每次运行都通过了助教的 `verify_` 校验程序。

**02 · Pthread / OpenMP** —— 换了更快的算法，赢的调度策略也跟着换了

| 实现 | 1 线程 | 4 线程 | 4 线程加速比 |
| :--- | --: | --: | --: |
| Pthread 基础版 | 1.043 s | 0.323 s | 3.23× |
| Pthread 可分离版 | 0.400 s | 0.128 s | 3.12× |
| OpenMP 基础版 | 1.053 s | 0.370 s | 2.84× |
| OpenMP 可分离版 | 0.392 s | 0.111 s | **3.52×** |

可分离卷积把每像素 25 次乘加降到 10 次，与理论值 2.5× 吻合。有意思的是那个反转：
基础版上 Pthread 的静态划分赢，可分离版一旦把单次迭代的活儿变轻，OpenMP 的动态调度就反超了。

**03 · CUDA** —— 一个正确的优化，但什么都没改变

| 版本 | 仅 Kernel | 除 I/O 外 | 全流程 |
| :--- | --: | --: | --: |
| 基础版（两次 kernel） | 0.230 ms | 64.6 ms | 187.6 ms |
| 优化版（融合 kernel） | **0.180 ms** | 68.6 ms | 191.1 ms |

算子融合省掉了 32 MB 的显存往返，kernel 快了 27%。但它在端到端里完全看不见：
187.6 ms 里的 0.05 ms。BMP 解析和磁盘 I/O 主导了整个管道——这才是这次作业真正的结论。

## 交互式原理演示

九个自写的 HTML 页面，把算法过程逐帧展开——缓存行复用、卷积窗口滑动、线程到像素的映射、
warp 级访存模式。

**→ [jie-gittttt.github.io/parallel-computing](https://jie-gittttt.github.io/parallel-computing/)**

这些是静态页面，不需要服务器；但会从公开 CDN 加载 Tailwind、React 和 Babel，
所以需要联网才能渲染。

## 运行

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

各子目录的 README 有完整的环境要求、参数说明与预期输出。

## 目录结构

```
parallel-computing/
├── README.md                           # 本文件（英文）
├── README.zh-CN.md                     # 中文版
├── index.html                          # 交互演示导航页
├── 01-mpi-matrix-multiplication/       # MPI · D = A×B + C
├── 02-pthread-openmp-convolution/      # Pthread / OpenMP · 5×5 高斯卷积
└── 03-cuda-convolution-pooling/        # CUDA · 卷积 + 最大池化
```

## 关于素材

- **教师提供、未包含**：作业 01 的 `verify_` 结果校验程序；作业 02 的测试程序套件
  （含 `test.cpp` / `test_blur.py` 与黑白色块测试图）。
- **教师提供、已包含**：标准高斯核（`src/gaussian_kernel.txt`）与测试图像（`data/timg.bmp`）——
  这两份是程序运行必需的输入，缺了代码跑不起来。
- 作业 01 的计时来自课程共享服务器。绝对值会随机器负载波动，三次不同会话的原始输出
  完整保留在 `01-mpi-matrix-multiplication/results/run-log.txt`。
- 所有源码、报告、幻灯片与可视化演示均为本人独立完成。
