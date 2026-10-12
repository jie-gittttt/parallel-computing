# 作业 02 · 基于共享内存的 BMP 图像并行卷积

对 BMP 图像做 5×5 高斯卷积（σ = 1.5，半径 2），分别用 **Pthread** 和 **OpenMP** 两种
共享内存并行方式实现，并对比不同并行范式的差异。

```bash
g++ -O2 -o pthread_conv src/pthread/pthread.cpp -lpthread
g++ -O2 -fopenmp -o omp_conv src/openmp/omp.cpp

./pthread_conv 4        # 参数为线程数
./omp_conv              # 线程数由 OMP_NUM_THREADS 控制
```

程序读入 BMP，卷积后写回同尺寸的 BMP。边界采用**零填充**（Zero Padding）——
越界像素视为 0，这样输出图与输入图尺寸一致，不会产生边界收缩。

---

## 核心优化：可分离卷积

5×5 高斯核是**可分离**的——一个二维高斯可以写成两个一维高斯的乘积：

```
G2D(x, y)  =  G1D(x) · G1D(y)
```

于是二维卷积可以拆成两趟一维卷积：

| | 每像素乘加次数 | 中间结果 |
| :--- | --: | :--- |
| 直接二维卷积 | 5 × 5 = **25** | — |
| 可分离：水平 1×5 + 垂直 5×1 | 5 + 5 = **10** | 需一张 float 中间图 |

**计算量降到 40%**，理论上单线程就能提速 2.5×。

两趟之间必须同步：垂直卷积要读取水平卷积的**上下相邻行**，必须等整个水平阶段完成。
Pthread 版本用手写 barrier 实现，OpenMP 版本则利用 `parallel for` 结束处的**隐式屏障**。

### 精度上的意外收益

基础版把每趟结果都截断成 8-bit 整数，**两次截断**会累积误差。
可分离版把中间结果保存在 `float` 缓冲里，只在最后一趟才截断——
多花一张与图像等大的内存，换到更低的 MSE（图像质量更好）。

> 逐阶段对照：[convolution-optimized.html](https://jie-gittttt.github.io/parallel-computing/02-pthread-openmp-convolution/docs/convolution-optimized.html)

---

## 两种并行范式

### Pthread：显式线程管理

```cpp
pthread_create(&threads[i], NULL, worker, &args[i]);
pthread_join(threads[i], NULL);
```

`pthread_create` 只接受一个 `void*` 参数，所以需要自己定义结构体来打包
图像指针、行范围、和高斯核。任务划分靠**手工计算 `start_row` 和 `end_row`**——
按行静态均分，每个线程负责一段连续的行。

### OpenMP：编译指令

```cpp
#pragma omp parallel for collapse(2) schedule(dynamic)
for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) { ... }
```

| 指令 | 作用 |
| :--- | :--- |
| `parallel for` | 把紧跟着的 for 循环自动分配给多个线程，Fork-Join 模型 |
| `collapse(2)` | 把 `y` 和 `x` 两层循环**合并成一个扁平迭代空间**再划分。不 collapse 的话只能按行切，线程数超过图像行数就没法再分了；collapse 后并行粒度从「行」细化到「像素」 |
| `schedule(dynamic)` | 动态调度：线程做完手头一块就去领下一块（抢单模式）。解决「最慢的线程拖累整体」——图像边缘与中心的像素计算量相同，但缓存行为不同，静态划分容易出现长尾 |

OpenMP 会自动推断 `y`、`x` 为 `private`、图像指针为 `shared`，不需要手写。
Pthread 版本要做同样的动态负载均衡，得自己维护一个任务队列加互斥锁。

---

## 结果

同一张测试图，计时单位秒（3 次测量取平均）：

| 实现 | 1 线程 | 2 线程加速比 | 4 线程 | 4 线程加速比 |
| :--- | --: | --: | --: | --: |
| Pthread 基础版 | 1.043 | 1.93× | 0.323 | 3.23× |
| Pthread 可分离版 | 0.400 | 1.96× | 0.128 | 3.12× |
| OpenMP 基础版 | 1.053 | 1.53× | 0.370 | 2.84× |
| OpenMP 可分离版 | 0.392 | 1.97× | 0.111 | **3.52×** |

**可分离卷积的效果**：Pthread 从 1.043 s 降到 0.400 s（2.61×），
OpenMP 从 1.053 s 降到 0.392 s（2.69×）——与理论值 2.5× 吻合，甚至略好，
多出来的一点来自 float 中间缓冲减少的整数截断开销。

### 一个有意思的反转

在**基础版**上，Pthread 的静态划分反而赢（4 线程 3.23× vs 2.84×）；
到了**可分离版**，OpenMP 的动态调度反超（3.52× vs 3.12×）。

原因在于每像素的工作量变了。基础版每像素 25 次乘加，每次迭代的活儿足够重，
`schedule(dynamic)` 每次「领任务」的原子操作开销纯属浪费，静态划分更划算。
可分离版把每次迭代降到 10 次乘加，**调度开销在总时间里的占比翻了倍**，
而动态调度换来的负载均衡收益开始盖过它——尤其是第二趟垂直卷积存在明显的 cache 长尾。

换句话说：**调度策略没有绝对优劣，取决于单次迭代的粒度**。
这也是作业里同时保留两套实现、分别在两种粒度下测量的价值所在。

> 完整的 Pthread vs OpenMP 逐维度对比（编程范式、代码量、任务划分、同步机制、
> 可移植性）见 [data/comparison.csv](data/comparison.csv)。

---

## 文件结构

```
02-pthread-openmp-convolution/
├── src/
│   ├── pthread/
│   │   ├── pthread.cpp             # 基础版：二维 5×5 卷积
│   │   └── pthread_sep.cpp         # 可分离版：水平 + 垂直两趟，手写 barrier
│   ├── openmp/
│   │   ├── omp.cpp                 # 基础版：collapse(2) + schedule(dynamic)
│   │   └── omp_sep.cpp             # 可分离版：靠 parallel for 的隐式屏障同步
│   └── gaussian_kernel.txt         # 作业指定的 5×5 高斯核（σ = 1.5）
├── report/
│   ├── pthread.pdf                 # Pthread 实验报告
│   └── openmp.pdf                  # OpenMP 实验报告
├── slides/presentation.pdf         # 汇报 PPT
├── data/
│   ├── pthread-results.csv         # Pthread 计时原始数据（1/2/4 线程）
│   ├── openmp-results.csv          # OpenMP 计时原始数据
│   └── comparison.csv              # 两种范式的逐维度对比表
├── figures/                        # 各核数输出图与正确性验证图（9 张 PNG）
└── docs/
    ├── convolution.html            # 卷积窗口如何在图像上滑动
    ├── convolution-optimized.html  # 二维 → 两趟一维的逐阶段对照
    └── convolution-parallel.html   # 按行划分与负载均衡问题
```
