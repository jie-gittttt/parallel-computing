#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <string>
#include <chrono> 
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ================= 配置区域 =================
const std::string STUDENT_ID = "YOUR_STUDENT_ID";
const std::string OUTPUT_FILENAME = "Cuda_DirectFused_" + STUDENT_ID + ".bmp";

// ================= CUDA 错误检查宏 =================
#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", " \
                  << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

// ================= 结构定义 =================
#pragma pack(push, 1)
struct BMPFileHeader {
    uint16_t bfType; uint32_t bfSize; uint16_t bfReserved1; uint16_t bfReserved2; uint32_t bfOffBits;
};
struct BMPInfoHeader {
    uint32_t biSize; int32_t biWidth; int32_t biHeight; uint16_t biPlanes; uint16_t biBitCount;
    uint32_t biCompression; uint32_t biSizeImage; int32_t biXPelsPerMeter; int32_t biYPelsPerMeter;
    uint32_t biClrUsed; uint32_t biClrImportant;
};
#pragma pack(pop)

struct Pixel { unsigned char b, g, r; };

// ================= 常量内存 (2D 卷积核) =================
__constant__ float d_KERNEL_2D[25];

// 高斯核 Sigma=1.5
const float h_KERNEL_2D[25] = {
    0.01441881f, 0.02808402f, 0.03507270f, 0.02808402f, 0.01441881f,
    0.02808402f, 0.05470020f, 0.06831229f, 0.05470020f, 0.02808402f,
    0.03507270f, 0.06831229f, 0.08531173f, 0.06831229f, 0.03507270f,
    0.02808402f, 0.05470020f, 0.06831229f, 0.05470020f, 0.02808402f,
    0.01441881f, 0.02808402f, 0.03507270f, 0.02808402f, 0.01441881f
};
const int K_RADIUS = 2;

// ================= 核心优化 Kernel (去掉了 __ldg) =================
// 策略：单核融合 (Fused Standard Conv + Pooling)
// 每个线程计算 1 个输出像素 (即原图 2x2 区域的卷积+最大值)
__global__ void fused_conv2d_pool_kernel(const Pixel* __restrict__ input, 
                                         Pixel* output, 
                                         int in_width, int in_height, 
                                         int out_width) {
    // 1. 确定当前线程负责的【输出像素】坐标
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Grid 是按输出图像大小分配的，所以只需检查输出边界
    if (x >= out_width || y >= (in_height / 2)) return;

    // 2. 映射回【输入图像】的起始坐标 (因为是 2x2 池化，所以乘 2)
    int src_start_x = x * 2;
    int src_start_y = y * 2;

    float max_r = 0.0f, max_g = 0.0f, max_b = 0.0f;

    // 3. 局部循环：遍历 2x2 的池化窗口
    #pragma unroll
    for (int dy = 0; dy < 2; ++dy) {
        #pragma unroll
        for (int dx = 0; dx < 2; ++dx) {
            
            // 当前卷积中心点
            int cx = src_start_x + dx;
            int cy = src_start_y + dy;

            float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;

            // 4. 执行标准 5x5 卷积
            if (cx >= 0 && cx < in_width && cy >= 0 && cy < in_height) {
                #pragma unroll
                for (int ky = -K_RADIUS; ky <= K_RADIUS; ++ky) {
                    #pragma unroll
                    for (int kx = -K_RADIUS; kx <= K_RADIUS; ++kx) {
                        int nx = cx + kx;
                        int ny = cy + ky;

                        // 边界检查
                        if (nx >= 0 && nx < in_width && ny >= 0 && ny < in_height) {
                            
                            // 【修改处】直接使用标准数组访问，替代 __ldg
                            // 加上 const __restrict__ 后，编译器依然会尽可能优化缓存
                            Pixel p = input[ny * in_width + nx];
                            
                            float w = d_KERNEL_2D[(ky + K_RADIUS) * 5 + (kx + K_RADIUS)];
                            sum_r += p.r * w;
                            sum_g += p.g * w;
                            sum_b += p.b * w;
                        }
                    }
                }
            }

            // 5. 每次卷积后，立即进行池化比较
            if (sum_r > max_r) max_r = sum_r;
            if (sum_g > max_g) max_g = sum_g;
            if (sum_b > max_b) max_b = sum_b;
        }
    }

    // 6. 写入结果 (直接写入 Global Memory)
    int out_idx = y * out_width + x;
    output[out_idx].r = (unsigned char)fminf(fmaxf(max_r, 0.0f), 255.0f);
    output[out_idx].g = (unsigned char)fminf(fmaxf(max_g, 0.0f), 255.0f);
    output[out_idx].b = (unsigned char)fminf(fmaxf(max_b, 0.0f), 255.0f);
}

// ================= Host: BMP 类 (保持不变) =================
class BMPImage {
public:
    int width, height;
    std::vector<Pixel> data;
    BMPFileHeader fileHeader;
    BMPInfoHeader infoHeader;
    std::vector<unsigned char> rawHeader;

    bool load(const char* filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) return false;
        file.read((char*)&fileHeader, sizeof(fileHeader));
        file.read((char*)&infoHeader, sizeof(infoHeader));
        if (fileHeader.bfType != 0x4D42) return false;
        width = infoHeader.biWidth;
        height = std::abs(infoHeader.biHeight);
        int offset = fileHeader.bfOffBits;
        int headerSize = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);
        if (offset > headerSize) {
            rawHeader.resize(offset - headerSize);
            file.read((char*)rawHeader.data(), rawHeader.size());
        }
        int padding = (4 - (width * 3) % 4) % 4;
        data.resize(width * height);
        file.seekg(offset, std::ios::beg);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                file.read((char*)&data[y * width + x], 3);
            }
            file.ignore(padding);
        }
        return true;
    }

    bool save(const char* filename, int new_w, int new_h, const std::vector<Pixel>& new_data) {
        std::ofstream file(filename, std::ios::binary);
        if (!file) return false;
        BMPFileHeader outFH = fileHeader;
        BMPInfoHeader outIH = infoHeader;
        outIH.biWidth = new_w;
        outIH.biHeight = new_h;
        int padding = (4 - (new_w * 3) % 4) % 4;
        uint32_t dataSize = (new_w * 3 + padding) * std::abs(new_h);
        outIH.biSizeImage = dataSize;
        outFH.bfSize = outFH.bfOffBits + dataSize;
        file.write((char*)&outFH, sizeof(outFH));
        file.write((char*)&outIH, sizeof(outIH));
        if (!rawHeader.empty()) file.write((char*)rawHeader.data(), rawHeader.size());
        unsigned char padBytes[3] = {0, 0, 0};
        for (int y = 0; y < std::abs(new_h); ++y) {
            for (int x = 0; x < new_w; ++x) {
                file.write((char*)&new_data[y * new_w + x], 3);
            }
            file.write((char*)padBytes, padding);
        }
        return true;
    }
};

int main() {
    auto start_io = std::chrono::high_resolution_clock::now();

    BMPImage img;
    if (!img.load("timg.bmp")) {
        std::cerr << "Error: Cannot load timg.bmp" << std::endl;
        return -1;
    }
    int width = img.width;
    int height = img.height;
    int out_width = width / 2;
    int out_height = height / 2;
    
    // 2. 显存分配
    Pixel *d_src, *d_pool;
    size_t img_bytes = width * height * sizeof(Pixel);
    size_t out_bytes = out_width * out_height * sizeof(Pixel);

    // --- 算法核心计时开始 ---
    auto start_total = std::chrono::high_resolution_clock::now();

    CHECK(cudaMalloc(&d_src, img_bytes));
    CHECK(cudaMalloc(&d_pool, out_bytes));

    // 3. 拷贝数据
    CHECK(cudaMemcpy(d_src, img.data.data(), img_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(d_KERNEL_2D, h_KERNEL_2D, 25 * sizeof(float)));

    // 4. Grid 配置
    dim3 block(16, 16);
    dim3 grid((out_width + block.x - 1) / block.x, (out_height + block.y - 1) / block.y);

    // --- Kernel 计时开始 ---
    cudaEvent_t start_k, stop_k;
    cudaEventCreate(&start_k);
    cudaEventCreate(&stop_k);

    cudaEventRecord(start_k);

    // 启动 Fused Kernel
    fused_conv2d_pool_kernel<<<grid, block>>>(d_src, d_pool, width, height, out_width);

    CHECK(cudaGetLastError());
    cudaEventRecord(stop_k);
    cudaEventSynchronize(stop_k);
    
    float kernel_ms = 0;
    cudaEventElapsedTime(&kernel_ms, start_k, stop_k);

    // 5. 拷贝回结果
    std::vector<Pixel> result_data(out_width * out_height);
    CHECK(cudaMemcpy(result_data.data(), d_pool, out_bytes, cudaMemcpyDeviceToHost));

    CHECK(cudaDeviceSynchronize());
    // --- 算法核心计时结束 ---
    auto end_total = std::chrono::high_resolution_clock::now();

    // 6. 资源释放
    cudaFree(d_src);
    cudaFree(d_pool);
    cudaEventDestroy(start_k);
    cudaEventDestroy(stop_k);

    if (img.save(OUTPUT_FILENAME.c_str(), out_width, out_height, result_data)) {
        std::cout << "Saved: " << OUTPUT_FILENAME << std::endl;
    }

    auto end_io = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_ms = end_total - start_total;
    std::chrono::duration<double, std::milli> io_ms = end_io - start_io;

    // ================= 报表 =================
    std::cout << "\n=== Direct Fused Optimization Report (" << STUDENT_ID << ") ===" << std::endl;
    std::cout << "1. Kernel Time (Fused 2D+Pool)      : " << kernel_ms << " ms" << std::endl;
    std::cout << "2. Total Algo Time (H2D + Calc + D2H): " << total_ms.count() << " ms" << std::endl;
    std::cout << "3. End-to-End Time (with Disk I/O)   : " << io_ms.count() << " ms" << std::endl;
    std::cout << "=================================================" << std::endl;

    return 0;
}