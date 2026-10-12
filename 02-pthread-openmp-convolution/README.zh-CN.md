# 共享内存并行图像卷积 —— Pthread vs. OpenMP

[English](README.md) | [中文](README.zh-CN.md)

对 BMP 图像做 5×5 高斯卷积，用 Pthread 和 OpenMP 各实现一遍。当利用核的可分离性把每像素计算量减半后，
**哪种调度策略更好这件事发生了反转**。

## 1. 概述

任务是用两种共享内存并行范式对 BMP 图像做固定的 5×5 高斯卷积（σ = 1.5），并比较二者。
程序读入 BMP，卷积后写回同尺寸的 BMP。

两个目标：

1. **对比两种范式。** Pthread 把线程创建、任务划分、同步全部交给程序员显式控制；
   OpenMP 把同样的划分表达成编译器指令。作业要看的是两者分别让你在代码量和性能上付出什么。
2. **优化算法本身，而不只是并行度。** 5×5 高斯核是**可分离**的，利用这一点能把计算量减半——
   这与用多少个核无关。

边界采用**零填充**（Zero Padding）而非裁剪，因此输出图与输入图尺寸完全一致。

## 2. 核心贡献

- **四份完整实现** —— Pthread 和 OpenMP 各有一个直接二维版和一个可分离版，
  这样算法改变的贡献与并行的贡献可以分开衡量。
- **手写 barrier**（Pthread 可分离版）。垂直卷积要读取水平卷积产出的行，
  因此必须等所有线程都完成第一阶段才能开始第二阶段。OpenMP 靠 `parallel for` 结束处的
  **隐式屏障**自动获得这一点；Pthread 需要自己用 `pthread_barrier_t`。
- **float 中间缓冲**（可分离版）。直接二维版每趟都截断成 8-bit，**每像素截断两次**；
  可分离版把中间结果保存在 `float` 里，只在最后一趟截断。代价是多一张与图像等大的内存，
  换来可测量的 MSE 下降。
- **记录了一个性能反转现象**：两种范式谁更快取决于单次迭代的粒度（见第 4 节）。
- **保留计时数据与范式对比表**（`data/` 下的 CSV）。

## 3. 方法与架构

### 可分离卷积

二维高斯可以写成两个一维高斯的乘积：`G2D(x, y) = G1D(x) · G1D(y)`。
于是 5×5 卷积可以拆成两趟一维卷积：

| | 每像素乘加次数 | 中间结果 |
| :--- | --: | :--- |
| 直接二维卷积 | 5 × 5 = **25** | — |
| 可分离：水平 1×5 + 垂直 5×1 | 5 + 5 = **10** | 一张 float 中间图 |

```
 输入（8-bit）                  第一趟                     第二趟                输出
   ┌────────┐   1×5 水平卷积   ┌────────┐   5×1 垂直卷积  ┌────────┐  截断一次  ┌───────┐
   │  H×W   │ ───────────────▶ │ float  │ ──────────────▶ │ float  │ ────────▶ │ 8-bit │
   └────────┘   （行间独立）    │  H×W   │  （读取上下 ±2 行）└────────┘           └───────┘
                               └────────┘
                                 ▲ 所有线程必须都到过这里，第二趟才能开始
```

计算量降到直接版的 40%，理论上单线程就能提速 2.5×。两趟之间必须同步。

### Pthread：显式线程管理

```cpp
pthread_create(&threads[i], NULL, worker, &args[i]);
pthread_join(threads[i], NULL);
```

`pthread_create` 只接受一个 `void*` 参数，所以需要自定义结构体来打包图像指针、行范围和高斯核。
任务划分靠**手工计算 `start_row` 和 `end_row`**——按行**静态**均分，每个线程负责一段连续的行。

### OpenMP：编译指令

```cpp
#pragma omp parallel for collapse(2) schedule(dynamic)
for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) { ... }
```

| 指令 | 作用 |
| :--- | :--- |
| `parallel for` | 把紧跟着的 for 循环分配给多个线程，Fork-Join 模型 |
| `collapse(2)` | 把 `y` 和 `x` 两层循环**合并成一个扁平迭代空间**再划分。不 collapse 的话只能按行切，并行粒度是「一行」；collapse 后细化到「一个像素」 |
| `schedule(dynamic)` | 动态调度：线程做完手头一块就去领下一块（抢单模式）。解决图像不同区域缓存行为不同造成的长尾 |

OpenMP 会自动推断 `y`、`x` 为 `private`、图像指针为 `shared`。
Pthread 版本要做同样的动态负载均衡，得自己维护任务队列加互斥锁。

## 4. 实验与结果

同一张测试图，计时单位秒（3 次测量取平均）：

| 实现 | 1 线程 | 2 线程加速比 | 4 线程 | 4 线程加速比 |
| :--- | --: | --: | --: | --: |
| Pthread 基础版 | 1.043 | 1.93× | 0.323 | 3.23× |
| Pthread 可分离版 | 0.400 | 1.96× | 0.128 | 3.12× |
| OpenMP 基础版 | 1.053 | 1.53× | 0.370 | 2.84× |
| OpenMP 可分离版 | 0.392 | 1.97× | 0.111 | **3.52×** |

**仅可分离性带来的效果**：Pthread 从 1.043 s 降到 0.400 s（**2.61×**），
OpenMP 从 1.053 s 降到 0.392 s（**2.69×**）。与理论值 2.5× 吻合，甚至略好；
多出来的一点来自 float 中间缓冲减少的那一轮 8-bit 截断。

### 一个有意思的反转

在**基础版**上，Pthread 的静态划分反而赢（4 线程 3.23× vs. 2.84×）；
到了**可分离版**，OpenMP 的动态调度反超（3.52× vs. 3.12×）。

原因在于每像素的工作量变了。基础版每像素 25 次乘加，每次迭代的活儿足够重，
`schedule(dynamic)` 每次「领任务」的原子操作开销纯属浪费。
可分离版把每次迭代降到 10 次乘加，**调度开销在总时间里的占比翻了倍**，
而动态调度换来的负载均衡收益开始盖过它——尤其是第二趟垂直卷积存在明显的 cache 长尾。

换句话说：**调度策略没有绝对优劣，取决于单次迭代的粒度**。
这也是作业里同时保留两套实现、分别在两种粒度下测量的价值所在。

> 完整的 Pthread vs. OpenMP 逐维度对比（编程范式、代码量、任务划分、同步机制、可移植性）见
> [`data/comparison.csv`](data/comparison.csv)。

## 5. 技术栈

- **语言**：C++，POSIX 线程与 OpenMP
- **编译器**：`g++ -O2`，链接 `-lpthread` 与 `-fopenmp`
- **图像 I/O**：自写的 BMP 读写（不依赖任何图像库）
- **计时**：`std::chrono`，3 次取平均

## 6. 安装与运行

**环境要求** —— 支持 OpenMP 的 `g++`。程序从工作目录读取 BMP，不需要任何外部库。

```bash
g++ -O2 -o pthread_conv src/pthread/pthread.cpp -lpthread
g++ -O2 -fopenmp -o omp_conv src/openmp/omp.cpp

./pthread_conv 4          # 参数为线程数
./omp_conv                # 线程数由 OMP_NUM_THREADS 控制
```

可分离版的构建方式相同：

```bash
g++ -O2 -o pthread_conv_sep src/pthread/pthread_sep.cpp -lpthread
g++ -O2 -fopenmp -o omp_conv_sep src/openmp/omp_sep.cpp
```

```
OMP_NUM_THREADS=4 ./omp_conv_sep
```

5×5 高斯核固定在 [`src/gaussian_kernel.txt`](src/gaussian_kernel.txt)（σ = 1.5，半径 2），由作业指定。

**正确性验证** —— 与参考输出的比对结果见
[`figures/pthread-verify.png`](figures/pthread-verify.png)。

## 7. 局限性

- **最多只测到 4 线程。** 没有超过 4 核的扩展性研究，因此同步与调度开销何时开始占主导是未知的。
- **只有一张测试图。** 缓存行为高度依赖图像尺寸相对于 L2/L3 的大小；
  尤其是那个 Pthread/OpenMP 反转，换成大得多或小得多的图未必成立。
- **静态与动态调度是跨范式比较，不是同一范式内的比较。** 这里 Pthread 永远是静态划分、
  OpenMP 永远是动态调度，所以比较结果把「范式」和「调度策略」混在了一起。
  要分离两者，需要再写一个动态调度的 Pthread 版本。
- **barrier 的开销没有被单独测量。** 推测手写 Pthread barrier 比 OpenMP 的隐式屏障更贵，
  但没有实验把它单独隔离出来。
- **零填充是固定的。** 没有与镜像填充或复制填充做对比，而那会同时改变边缘质量和访存模式。
- **只有一种核。** 可分离性是高斯核特有的性质；O(K²) → O(2K) 的论证不能推广到任意 5×5 核。

## 目录结构

```
02-pthread-openmp-convolution/
├── src/
│   ├── pthread/
│   │   ├── pthread.cpp             # 基础版：二维直接卷积，静态按行划分
│   │   └── pthread_sep.cpp         # 可分离版：手写 barrier
│   ├── openmp/
│   │   ├── omp.cpp                 # 基础版：collapse(2) + schedule(dynamic)
│   │   └── omp_sep.cpp             # 可分离版：隐式屏障
│   └── gaussian_kernel.txt         # 5×5 高斯核（σ = 1.5），教师提供
├── report/
│   ├── pthread.pdf                 # Pthread 实验报告
│   └── openmp.pdf                  # OpenMP 实验报告
├── slides/presentation.pdf         # 汇报 PPT
├── data/
│   ├── pthread-results.csv         # Pthread 计时原始数据（1/2/4 线程）
│   ├── openmp-results.csv          # OpenMP 计时原始数据
│   └── comparison.csv              # 两种范式的逐维度对比表
├── figures/                        # 各核数输出图与正确性验证图
└── docs/
    ├── convolution.html            # 交互演示：卷积窗口如何滑动
    ├── convolution-optimized.html  # 交互演示：二维 → 两趟一维，逐阶段对照
    └── convolution-parallel.html   # 交互演示：按行划分与负载均衡
```
