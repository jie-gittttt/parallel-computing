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
const std::string OUTPUT_FILENAME = "Cuda_Opt_" + STUDENT_ID + ".bmp";

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
    uint16_t bfType;
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
};

struct BMPInfoHeader {
    uint32_t biSize;
    int32_t  biWidth;
    int32_t  biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    int32_t  biXPelsPerMeter;
    int32_t  biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
};
#pragma pack(pop)

struct Pixel {
    unsigned char b, g, r;
};

// 中间结果使用 Float，防止两次截断导致精度丢失
struct PixelFloat {
    float b, g, r;
};

// ================= 常量内存 (1D 卷积核) =================
// 1D Kernel, Radius=2
__constant__ float d_KERNEL_1D[5];

// 对应 Sigma=1.5 的 1D 核
const float h_KERNEL_1D[5] = {0.120078f, 0.233880f, 0.292082f, 0.233880f, 0.120078f};
const int K_RADIUS = 2;

// ================= Kernel 1: 行卷积 (Row Convolution) =================
// 输入: uchar (Pixel) -> 输出: float (PixelFloat)
__global__ void row_conv_kernel(Pixel* input, PixelFloat* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;

    // 1D 水平遍历
    #pragma unroll
    for (int k = -K_RADIUS; k <= K_RADIUS; ++k) {
        int nx = x + k;
        // Zero Padding 
        if (nx >= 0 && nx < width) {
            Pixel p = input[y * width + nx];
            float w = d_KERNEL_1D[k + K_RADIUS];
            sum_r += p.r * w;
            sum_g += p.g * w;
            sum_b += p.b * w;
        }
    }

    // 存入中间 Buffer (不截断)
    int idx = y * width + x;
    output[idx].r = sum_r;
    output[idx].g = sum_g;
    output[idx].b = sum_b;
}

// ================= Kernel 2: 列卷积 (Column Convolution) =================
// 输入: float (PixelFloat) -> 输出: uchar (Pixel) (此处我们直接输出卷积结果，不做池化，池化单独做)
// 注意：为了保持模块化，这里先输出卷积后的图像。
// 实际上为了极致性能，可以将列卷积和池化融合(Kernel Fusion)，但为了代码清晰度和复用性，这里保持三步走。
__global__ void col_conv_kernel(PixelFloat* input, Pixel* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;

    // 1D 垂直遍历
    #pragma unroll
    for (int k = -K_RADIUS; k <= K_RADIUS; ++k) {
        int ny = y + k;
        if (ny >= 0 && ny < height) {
            PixelFloat p = input[ny * width + x];
            float w = d_KERNEL_1D[k + K_RADIUS];
            sum_r += p.r * w;
            sum_g += p.g * w;
            sum_b += p.b * w;
        }
    }

    // 最终截断
    int idx = y * width + x;
    output[idx].r = (unsigned char)fminf(fmaxf(sum_r, 0.0f), 255.0f);
    output[idx].g = (unsigned char)fminf(fmaxf(sum_g, 0.0f), 255.0f);
    output[idx].b = (unsigned char)fminf(fmaxf(sum_b, 0.0f), 255.0f);
}

// ================= Kernel 3: 2x2 Max Pooling =================
__global__ void max_pool_kernel(Pixel* input, Pixel* output, int in_width, int in_height, int out_width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_width || y >= (in_height / 2)) return;

    int start_x = x * 2;
    int start_y = y * 2;

    unsigned char max_r = 0, max_g = 0, max_b = 0;

    #pragma unroll
    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            int nx = start_x + dx;
            int ny = start_y + dy;

            if (nx < in_width && ny < in_height) {
                Pixel p = input[ny * in_width + nx];
                if (p.r > max_r) max_r = p.r;
                if (p.g > max_g) max_g = p.g;
                if (p.b > max_b) max_b = p.b;
            }
        }
    }

    int out_idx = y * out_width + x;
    output[out_idx].r = max_r;
    output[out_idx].g = max_g;
    output[out_idx].b = max_b;
}

// ================= Host: BMP 类 =================
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
    // --- 【计时开始 1：端到端 IO 时间】(包含硬盘读写) ---
    auto start_io = std::chrono::high_resolution_clock::now();

    // 1. 读取图片 (硬盘 IO)
    BMPImage img;
    if (!img.load("timg.bmp")) {
        std::cerr << "Error: Cannot load timg.bmp" << std::endl;
        return -1;
    }
    int width = img.width;
    int height = img.height;
    int out_width = width / 2;
    int out_height = height / 2;

    // --- 【计时开始 2：整体运行时间】(算法核心: 传输+计算) ---
    auto start_total = std::chrono::high_resolution_clock::now();

    // 2. 准备 Device 内存
    Pixel *d_src, *d_conv_final, *d_pool;
    PixelFloat *d_buffer; 

    size_t img_bytes = width * height * sizeof(Pixel);
    size_t buffer_bytes = width * height * sizeof(PixelFloat); 
    size_t out_bytes = out_width * out_height * sizeof(Pixel);

    CHECK(cudaMalloc(&d_src, img_bytes));
    CHECK(cudaMalloc(&d_buffer, buffer_bytes));
    CHECK(cudaMalloc(&d_conv_final, img_bytes)); 
    CHECK(cudaMalloc(&d_pool, out_bytes));

    // 3. 拷贝数据 Host -> Device (PCIe 传输)
    CHECK(cudaMemcpy(d_src, img.data.data(), img_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(d_KERNEL_1D, h_KERNEL_1D, 5 * sizeof(float)));

    // 4. 配置 Grid
    dim3 block(16, 16);
    dim3 grid_full((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    dim3 grid_pool((out_width + block.x - 1) / block.x, (out_height + block.y - 1) / block.y);

    // --- 【计时开始 3：并行部分时间】(纯 GPU 计算) ---
    cudaEvent_t start_k, stop_k;
    cudaEventCreate(&start_k);
    cudaEventCreate(&stop_k);
    cudaEventRecord(start_k);

    // Kernel 1: 行卷积
    row_conv_kernel<<<grid_full, block>>>(d_src, d_buffer, width, height);
    
    // Kernel 2: 列卷积
    col_conv_kernel<<<grid_full, block>>>(d_buffer, d_conv_final, width, height);
    
    // Kernel 3: 池化
    max_pool_kernel<<<grid_pool, block>>>(d_conv_final, d_pool, width, height, out_width);
    
    // 检查 Kernel 启动错误
    CHECK(cudaGetLastError());

    // --- 【计时结束 3：并行部分时间】 ---
    cudaEventRecord(stop_k);
    cudaEventSynchronize(stop_k);
    float kernel_time_ms = 0;
    cudaEventElapsedTime(&kernel_time_ms, start_k, stop_k);

    // 5. 拷贝回 Host (PCIe 传输)
    std::vector<Pixel> result_data(out_width * out_height);
    CHECK(cudaMemcpy(result_data.data(), d_pool, out_bytes, cudaMemcpyDeviceToHost));

    // 确保设备端所有操作完成
    CHECK(cudaDeviceSynchronize());

    // --- 【计时结束 2：整体运行时间】 ---
    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_total = end_total - start_total;

    // 6. 释放显存资源 (通常计入算法开销，也可以放在Total之后，影响不大)
    cudaFree(d_src);
    cudaFree(d_buffer);
    cudaFree(d_conv_final);
    cudaFree(d_pool);
    cudaEventDestroy(start_k);
    cudaEventDestroy(stop_k);

    // 7. 保存图片 (硬盘 IO)
    if (img.save(OUTPUT_FILENAME.c_str(), out_width, out_height, result_data)) {
        std::cout << "Image saved to " << OUTPUT_FILENAME << std::endl;
    }

    // --- 【计时结束 1：端到端 IO 时间】 ---
    auto end_io = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_io = end_io - start_io;

    // ================= 输出三组时间 =================
    std::cout << "\n=== Performance Report (" << STUDENT_ID << ") ===" << std::endl;
    std::cout << "1. 并行部分时间 (GPU Kernel) : " << kernel_time_ms  << " ms" << std::endl;
    std::cout << "2. 整体运行时间 (H2D+Comp+D2H): " << diff_total.count() * 1000 << " ms" << std::endl;
    std::cout << "3. 端到端时间 (含硬盘 IO)    : " << diff_io.count() * 1000 << " ms" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}