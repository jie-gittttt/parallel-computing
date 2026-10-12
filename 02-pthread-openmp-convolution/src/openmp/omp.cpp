#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <omp.h>      // OpenMP 头文件
#include <cstring>
#include <algorithm>
#include <string>

// ================= 配置区域 =================
// 请在此处修改您的学号
const std::string STUDENT_ID = "YOUR_STUDENT_ID"; 
const std::string OUTPUT_FILENAME = "OpenMP+" + STUDENT_ID + ".bmp";
const std::string MATRIX_FILENAME = "matrix.txt";

// ================= BMP 处理模块 (复用之前的稳定代码) =================
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

class BMPImage {
public:
    int width, height;
    std::vector<Pixel> data;
    std::vector<unsigned char> headerInfo;

    bool load(const char* filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) return false;
        BMPFileHeader fileHeader;
        BMPInfoHeader infoHeader;
        file.read((char*)&fileHeader, sizeof(fileHeader));
        file.read((char*)&infoHeader, sizeof(infoHeader));
        if (fileHeader.bfType != 0x4D42) return false;
        width = infoHeader.biWidth;
        height = std::abs(infoHeader.biHeight);
        file.seekg(0, std::ios::beg);
        headerInfo.resize(fileHeader.bfOffBits);
        file.read((char*)headerInfo.data(), fileHeader.bfOffBits);
        int padding = (4 - (width * 3) % 4) % 4;
        data.resize(width * height);
        file.seekg(fileHeader.bfOffBits, std::ios::beg);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                file.read((char*)&data[y * width + x], 3);
            }
            file.ignore(padding);
        }
        return true;
    }

    bool save(const char* filename) {
        std::ofstream file(filename, std::ios::binary);
        if (!file) return false;
        file.write((char*)headerInfo.data(), headerInfo.size());
        int padding = (4 - (width * 3) % 4) % 4;
        unsigned char padBytes[3] = {0, 0, 0};
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                file.write((char*)&data[y * width + x], 3);
            }
            file.write((char*)padBytes, padding);
        }
        return true;
    }
};

// ================= 卷积核 (5x5 Gaussian) =================
const int K_SIZE = 5;
const int K_RADIUS = 2;
const float KERNEL[5][5] = {
    {0.01441881f, 0.02808402f, 0.03507270f, 0.02808402f, 0.01441881f},
    {0.02808402f, 0.05470020f, 0.06831229f, 0.05470020f, 0.02808402f},
    {0.03507270f, 0.06831229f, 0.08531173f, 0.06831229f, 0.03507270f},
    {0.02808402f, 0.05470020f, 0.06831229f, 0.05470020f, 0.02808402f},
    {0.01441881f, 0.02808402f, 0.03507270f, 0.02808402f, 0.01441881f}
};

int main(int argc, char** argv) {
    // 1. 设置线程数 (默认4核，或从命令行读取)
    int num_threads = 4;
    if (argc > 1) {
        num_threads = std::stoi(argv[1]);
    }
    omp_set_num_threads(num_threads);

    // 2. 读取图片
    const char* input_file = "timg.bmp";
    BMPImage img;
    if (!img.load(input_file)) {
        std::cerr << "Error: Cannot load " << input_file << ". Please confirm timg.bmp exists." << std::endl;
        return 1;
    }

    BMPImage outImg = img; // 复制头信息
    int height = img.height;
    int width = img.width;

    std::cout << "Starting OpenMP convolution with " << num_threads << " threads..." << std::endl;

    // 3. 并行计算核心部分
    double start_time = omp_get_wtime(); // OpenMP 开始计时

    // #pragma omp parallel for: 自动将 for 循环的任务分配给多个线程
    // collapse(2): 将外层(y)和内层(x)循环合并为一个大循环进行划分，增加粒度
    // schedule(dynamic): 动态调度，解决潜在的负载不均衡问题
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;

            // 卷积核计算
            for (int ky = -K_RADIUS; ky <= K_RADIUS; ++ky) {
                for (int kx = -K_RADIUS; kx <= K_RADIUS; ++kx) {
                    int ny = y + ky;
                    int nx = x + kx;

                    // Zero Padding 边界处理 (Border Constant = 0)
                    if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                        // 多个线程同时读取 img.data 是安全的（只读）
                        Pixel& p = img.data[ny * width + nx];
                        float w = KERNEL[ky + K_RADIUS][kx + K_RADIUS];
                        sum_r += p.r * w;
                        sum_g += p.g * w;
                        sum_b += p.b * w;
                    }
                }
            }

            // 结果写入 (互不冲突，因为每个线程负责不同的 idx)
            int idx = y * width + x;
            outImg.data[idx].r = (unsigned char)std::min(std::max(sum_r, 0.0f), 255.0f);
            outImg.data[idx].g = (unsigned char)std::min(std::max(sum_g, 0.0f), 255.0f);
            outImg.data[idx].b = (unsigned char)std::min(std::max(sum_b, 0.0f), 255.0f);
        }
    }

    double end_time = omp_get_wtime(); // OpenMP 结束计时
    double total_time = end_time - start_time;

    std::cout << "OpenMP Execution Time: " << total_time << " seconds" << std::endl;

    // 4. 保存结果图片
    if (outImg.save(OUTPUT_FILENAME.c_str())) {
        std::cout << "Image saved to " << OUTPUT_FILENAME << std::endl;
    }

    // 5. 导出矩阵文件供验证 (作业要求1)
    std::ofstream matFile(MATRIX_FILENAME);
    if (matFile.is_open()) {
        matFile << "Convolution Kernel (5x5 Gaussian):\n";
        for(int i=0; i<5; i++) {
            for(int j=0; j<5; j++) matFile << KERNEL[i][j] << " ";
            matFile << "\n";
        }
        matFile << "\nOutput Pixel Matrix (R Channel Sample):\n";
        int print_h = std::min(20, outImg.height);
        int print_w = std::min(20, outImg.width);
        for(int y=0; y<print_h; y++) {
            for(int x=0; x<print_w; x++) {
                matFile << (int)outImg.data[y * outImg.width + x].r << " ";
            }
            matFile << "\n";
        }
        matFile.close();
        std::cout << "Verification matrix saved to " << MATRIX_FILENAME << std::endl;
    }

    return 0;
}