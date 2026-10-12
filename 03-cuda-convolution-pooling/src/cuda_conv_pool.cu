#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <string>
#include <chrono> // 用于CPU端整体计时
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ================= 配置区域 =================
const std::string STUDENT_ID = "YOUR_STUDENT_ID";
const std::string OUTPUT_FILENAME = "Cuda_卷积池化_" + STUDENT_ID + ".bmp";

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

// ================= BMP 结构定义 =================
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

// ================= 常量内存定义 (卷积核) =================
__constant__ float d_KERNEL[25];

const float h_KERNEL[5][5] = {
    {0.01441881f, 0.02808402f, 0.03507270f, 0.02808402f, 0.01441881f},
    {0.02808402f, 0.05470020f, 0.06831229f, 0.05470020f, 0.02808402f},
    {0.03507270f, 0.06831229f, 0.08531173f, 0.06831229f, 0.03507270f},
    {0.02808402f, 0.05470020f, 0.06831229f, 0.05470020f, 0.02808402f},
    {0.01441881f, 0.02808402f, 0.03507270f, 0.02808402f, 0.01441881f}
};

const int K_RADIUS = 2;

// ================= Device Kernel: 卷积 =================
__global__ void convolution_kernel(Pixel* input, Pixel* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;

    // 卷积计算
    for (int ky = -K_RADIUS; ky <= K_RADIUS; ++ky) {
        for (int kx = -K_RADIUS; kx <= K_RADIUS; ++kx) {
            int ny = y + ky;
            int nx = x + kx;

            if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                Pixel p = input[ny * width + nx];
                float w = d_KERNEL[(ky + K_RADIUS) * 5 + (kx + K_RADIUS)];
                sum_r += p.r * w;
                sum_g += p.g * w;
                sum_b += p.b * w;
            }
        }
    }

    int idx = y * width + x;
    output[idx].r = (unsigned char)fminf(fmaxf(sum_r, 0.0f), 255.0f);
    output[idx].g = (unsigned char)fminf(fmaxf(sum_g, 0.0f), 255.0f);
    output[idx].b = (unsigned char)fminf(fmaxf(sum_b, 0.0f), 255.0f);
}

// ================= Device Kernel: 2x2 Max Pooling =================
__global__ void max_pool_kernel(Pixel* input, Pixel* output, int in_width, int in_height, int out_width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_width || y >= (in_height / 2)) return;

    int start_x = x * 2;
    int start_y = y * 2;

    unsigned char max_r = 0, max_g = 0, max_b = 0;

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
        if (!rawHeader.empty()) {
            file.write((char*)rawHeader.data(), rawHeader.size());
        }

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

    // 1. 读取 Host 端图片
    BMPImage img;
    if (!img.load("timg.bmp")) {
        std::cerr << "Error: Cannot load timg.bmp" << std::endl;
        return -1;
    }
    std::cout << "Original Image: " << img.width << "x" << img.height << std::endl;

    int width = img.width;
    int height = img.height;
    int out_width = width / 2;
    int out_height = height / 2;

    // --- 【计时开始 2：整体运行时间】(算法核心: 传输+计算) ---
    auto start_total = std::chrono::high_resolution_clock::now();

    // 2. 准备 Device 内存
    Pixel *d_src, *d_conv, *d_pool;
    size_t img_size = width * height * sizeof(Pixel);
    size_t out_size = out_width * out_height * sizeof(Pixel);

    CHECK(cudaMalloc(&d_src, img_size));
    CHECK(cudaMalloc(&d_conv, img_size));
    CHECK(cudaMalloc(&d_pool, out_size));

    // 3. 拷贝数据 Host -> Device
    CHECK(cudaMemcpy(d_src, img.data.data(), img_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(d_KERNEL, h_KERNEL, 25 * sizeof(float)));

    // 4. 配置 Grid 和 Block
    dim3 block(16, 16);
    dim3 grid_conv((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    dim3 grid_pool((out_width + block.x - 1) / block.x, (out_height + block.y - 1) / block.y);

    // --- 【计时开始 3：并行部分时间】(纯 GPU 计算) ---
    cudaEvent_t start_kernel, stop_kernel;
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);

    cudaEventRecord(start_kernel); // 记录并行开始

    // Step 1: 卷积
    convolution_kernel<<<grid_conv, block>>>(d_src, d_conv, width, height);
    CHECK(cudaGetLastError()); // 检查启动错误
    
    // Step 2: 池化
    max_pool_kernel<<<grid_pool, block>>>(d_conv, d_pool, width, height, out_width);
    CHECK(cudaGetLastError());

    cudaEventRecord(stop_kernel); // 记录并行结束
    cudaEventSynchronize(stop_kernel); // 必须等待 GPU 完成才能算时间

    // --- 【计时计算：并行部分】 ---
    float kernel_milliseconds = 0;
    cudaEventElapsedTime(&kernel_milliseconds, start_kernel, stop_kernel);

    // 5. 拷贝结果 Device -> Host
    std::vector<Pixel> result_data(out_width * out_height);
    CHECK(cudaMemcpy(result_data.data(), d_pool, out_size, cudaMemcpyDeviceToHost));

    // --- 【计时结束 2：整体运行时间】 ---
    // 为了公平起见，这里的 sync 确保所有内存拷贝也完成了
    CHECK(cudaDeviceSynchronize());
    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_total = end_total - start_total;

    // 6. 释放显存资源 (放在 Total 计时后，Save 之前)
    cudaFree(d_src);
    cudaFree(d_conv);
    cudaFree(d_pool);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);

    // 7. 保存结果
    if (img.save(OUTPUT_FILENAME.c_str(), out_width, out_height, result_data)) {
        std::cout << "Image saved to " << OUTPUT_FILENAME << " (" << out_width << "x" << out_height << ")" << std::endl;
    }

    // --- 【计时结束 1：端到端 IO 时间】 ---
    auto end_io = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_io = end_io - start_io;

    // 8. 输出时间报表 (与优化版格式对齐)
    std::cout << "\n=== Basic CUDA Performance Report (" << STUDENT_ID << ") ===" << std::endl;
    std::cout << "Grid Shape (Conv): " << grid_conv.x << "x" << grid_conv.y << std::endl;
    std::cout << "Grid Shape (Pool): " << grid_pool.x << "x" << grid_pool.y << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    std::cout << "1. 并行部分时间 (GPU Kernel) : " << kernel_milliseconds  << " ms" << std::endl;
    std::cout << "2. 整体运行时间 (H2D+Comp+D2H): " << diff_total.count() * 1000<< " ms" << std::endl;
    std::cout << "3. 端到端时间 (含硬盘 IO)    : " << diff_io.count() * 1000 << " ms" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}


