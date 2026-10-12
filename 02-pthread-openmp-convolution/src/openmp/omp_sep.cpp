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
const std::string OUTPUT_FILENAME = "OpenMP_Opt+" + STUDENT_ID + ".bmp";
const std::string MATRIX_FILENAME = "matrix_opt.txt";

// ================= BMP 处理模块 (与基础版完全一致) =================
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

// 用于中间结果的高精度像素结构 (防止第一次截断导致误差)
struct PixelFloat {
    float b, g, r;
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

// ================= 可分离卷积核 (1D Gaussian) =================
// 对应 Sigma=1.5, Radius=2 的 1D 分解
const int K_RADIUS = 2;
const float KERNEL_1D[5] = {0.120078f, 0.233880f, 0.292082f, 0.233880f, 0.120078f};

int main(int argc, char** argv) {
    // 1. 设置线程数
    int num_threads = 4;
    if (argc > 1) {
        num_threads = std::stoi(argv[1]);
    }
    omp_set_num_threads(num_threads);

    // 2. 读取图片
    const char* input_file = "timg.bmp";
    BMPImage img;
    if (!img.load(input_file)) {
        std::cerr << "Error: Cannot load " << input_file << std::endl;
        return 1;
    }

    BMPImage outImg = img;
    int height = img.height;
    int width = img.width;

    // 申请中间缓冲区 (Float精度)，大小与原图一致
    std::vector<PixelFloat> tempBuffer(width * height);

    std::cout << "Starting OpenMP OPTIMIZED (Separable) convolution with " << num_threads << " threads..." << std::endl;

    double start_time = omp_get_wtime();

    // =========================================================
    // 阶段一：水平卷积 (Row Convolution)
    // 利用 OpenMP 自动划分任务
    // =========================================================
    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;

            // 1D 核遍历
            for (int k = -K_RADIUS; k <= K_RADIUS; ++k) {
                int nx = x + k;
                
                // Zero Padding 边界处理
                if (nx >= 0 && nx < width) {
                    Pixel& p = img.data[y * width + nx];
                    float w = KERNEL_1D[k + K_RADIUS];
                    sum_r += p.r * w;
                    sum_g += p.g * w;
                    sum_b += p.b * w;
                }
            }
            // 存入中间 Buffer，保持 Float 精度，暂不截断
            tempBuffer[y * width + x] = {sum_b, sum_g, sum_r};
        }
    }
    // <--- 隐式同步点 (Implicit Barrier) --->
    // OpenMP 保证所有线程完成上面的 for 循环后，才会继续执行下面的代码
    // 因此不需要像 Pthread 那样手写 pthread_barrier_wait()

    // =========================================================
    // 阶段二：垂直卷积 (Column Convolution)
    // 读取 tempBuffer，写入 outImg
    // =========================================================
    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;

            for (int k = -K_RADIUS; k <= K_RADIUS; ++k) {
                int ny = y + k;

                // 边界处理
                if (ny >= 0 && ny < height) {
                    // 从中间 Buffer 读取
                    PixelFloat& p = tempBuffer[ny * width + x];
                    float w = KERNEL_1D[k + K_RADIUS];
                    sum_r += p.r * w;
                    sum_g += p.g * w;
                    sum_b += p.b * w;
                }
            }

            // 最终归一化与截断
            int idx = y * width + x;
            outImg.data[idx].r = (unsigned char)std::min(std::max(sum_r, 0.0f), 255.0f);
            outImg.data[idx].g = (unsigned char)std::min(std::max(sum_g, 0.0f), 255.0f);
            outImg.data[idx].b = (unsigned char)std::min(std::max(sum_b, 0.0f), 255.0f);
        }
    }

    double end_time = omp_get_wtime();
    std::cout << "OpenMP Optimized Time: " << (end_time - start_time) << " seconds" << std::endl;

    // 保存图片
    if (outImg.save(OUTPUT_FILENAME.c_str())) {
        std::cout << "Image saved to " << OUTPUT_FILENAME << std::endl;
    }

    // 输出验证矩阵 (依然输出 1D 核数据)
    std::ofstream matFile(MATRIX_FILENAME);
    if (matFile.is_open()) {
        matFile << "Separable Convolution Used (1D Kernel):\n";
        for(int i=0; i<5; i++) matFile << KERNEL_1D[i] << " ";
        matFile << "\n\nOutput Pixel Matrix Sample:\n";
        int print_h = std::min(20, outImg.height);
        int print_w = std::min(20, outImg.width);
        for(int y=0; y<print_h; y++) {
            for(int x=0; x<print_w; x++) {
                matFile << (int)outImg.data[y * outImg.width + x].r << " ";
            }
            matFile << "\n";
        }
        matFile.close();
    }

    return 0;
}
