#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <pthread.h>
#include <sys/time.h>
#include <iomanip>
#include <cstring>

// ------------------- BMP 处理模块 (手写以符合系统作业要求) -------------------
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
    std::vector<unsigned char> headerInfo; // 存储原始头信息以便写回

    bool load(const char* filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) return false;

        BMPFileHeader fileHeader;
        BMPInfoHeader infoHeader;

        file.read((char*)&fileHeader, sizeof(fileHeader));
        file.read((char*)&infoHeader, sizeof(infoHeader));

        if (fileHeader.bfType != 0x4D42) return false; // "BM"

        width = infoHeader.biWidth;
        height = std::abs(infoHeader.biHeight);
        
        // 保存头信息用于输出
        file.seekg(0, std::ios::beg);
        headerInfo.resize(fileHeader.bfOffBits);
        file.read((char*)headerInfo.data(), fileHeader.bfOffBits);

        // 计算Padding
        int padding = (4 - (width * 3) % 4) % 4;

        data.resize(width * height);
        file.seekg(fileHeader.bfOffBits, std::ios::beg);

        // 注意：BMP通常是倒着存的（从下到上），但我们按顺序读入内存处理即可
        // 只要写回时保持一致，就不影响卷积逻辑
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

// ------------------- 卷积核 (来自高斯核.txt) -------------------
const int K_SIZE = 5;
const int K_RADIUS = 2;
// 必须归一化，否则图片会变亮或变暗
const float KERNEL[5][5] = {
    {0.01441881f, 0.02808402f, 0.03507270f, 0.02808402f, 0.01441881f},
    {0.02808402f, 0.05470020f, 0.06831229f, 0.05470020f, 0.02808402f},
    {0.03507270f, 0.06831229f, 0.08531173f, 0.06831229f, 0.03507270f},
    {0.02808402f, 0.05470020f, 0.06831229f, 0.05470020f, 0.02808402f},
    {0.01441881f, 0.02808402f, 0.03507270f, 0.02808402f, 0.01441881f}
};

// ------------------- 并行相关数据结构 -------------------
struct ThreadData {
    int thread_id;
    int num_threads;
    BMPImage* srcImg;
    BMPImage* destImg;
    double cost_time; // 存储该线程的运行时间
};

// 获取当前时间戳（微秒级）
double get_wall_time() {
    struct timeval time;
    if (gettimeofday(&time, NULL)) return 0;
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

// ------------------- 线程函数 -------------------
void* convolution_routine(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    double start_time = get_wall_time();

    int height = data->srcImg->height;
    int width = data->srcImg->width;
    
    // 按行划分任务
  
    int chunk_size = height / data->num_threads;
    int start_row = data->thread_id * chunk_size;
    int end_row = (data->thread_id == data->num_threads - 1) ? height : start_row + chunk_size;

    // 卷积操作
    for (int y = start_row; y < end_row; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;

            // 应用 5x5 核
            for (int ky = -K_RADIUS; ky <= K_RADIUS; ++ky) {
                for (int kx = -K_RADIUS; kx <= K_RADIUS; ++kx) {
                    int ny = y + ky;
                    int nx = x + kx;

                    // 边缘节点填0
                    if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                        Pixel& p = data->srcImg->data[ny * width + nx];
                        float w = KERNEL[ky + K_RADIUS][kx + K_RADIUS];
                        sum_r += p.r * w;
                        sum_g += p.g * w;
                        sum_b += p.b * w;
                    }
                    // else: implicit +0
                }
            }

            // 写回结果
            int idx = y * width + x;
            data->destImg->data[idx].r = (unsigned char)std::min(std::max(sum_r, 0.0f), 255.0f);
            data->destImg->data[idx].g = (unsigned char)std::min(std::max(sum_g, 0.0f), 255.0f);
            data->destImg->data[idx].b = (unsigned char)std::min(std::max(sum_b, 0.0f), 255.0f);
        }
    }

    double end_time = get_wall_time();
    data->cost_time = end_time - start_time;
    
    // 打印线程内部时间（作业要求2：程序段首尾加入返回时间戳并显示）
    printf("Thread %d calculation time: %.6f seconds\n", data->thread_id, data->cost_time);

    pthread_exit(NULL);
}

// ------------------- 主函数 -------------------
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./pthread_conv <num_threads>" << std::endl;
        return 1;
    }

    int num_threads = std::stoi(argv[1]);
    const char* input_file = "timg.bmp";
    const char* output_file = "output.bmp";
    const char* matrix_file = "matrix.txt";

    BMPImage img;
    if (!img.load(input_file)) {
        std::cerr << "Error: Cannot load " << input_file << ". Please rename your BMP to timg.bmp" << std::endl;
        return 1;
    }

    BMPImage outImg = img; // 复制头信息和大小
    
    // 初始化线程
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];

    std::cout << "Starting convolution with " << num_threads << " threads..." << std::endl;

    double main_start = get_wall_time();

    for (int i = 0; i < num_threads; ++i) {
        thread_data[i].thread_id = i;
        thread_data[i].num_threads = num_threads;
        thread_data[i].srcImg = &img;
        thread_data[i].destImg = &outImg;
        int rc = pthread_create(&threads[i], NULL, convolution_routine, (void*)&thread_data[i]);
        if (rc) {
            std::cerr << "Error: unable to create thread," << rc << std::endl;
            return 1;
        }
    }

    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }

    double main_end = get_wall_time();
    double total_time = main_end - main_start;

    std::cout << "Total Parallel Execution Time: " << total_time << " seconds" << std::endl;

    // 保存图片
    if (outImg.save(output_file)) {
        std::cout << "Image saved to " << output_file << std::endl;
    }

    // ------------------- 作业要求1: 输出卷积核和像素矩阵 -------------------
    std::ofstream matFile(matrix_file);
    if (matFile.is_open()) {
        matFile << "Convolution Kernel (5x5 Gaussian):\n";
        for(int i=0; i<5; i++) {
            for(int j=0; j<5; j++) matFile << KERNEL[i][j] << " ";
            matFile << "\n";
        }
        
        matFile << "\nOutput Pixel Matrix (R Channel, First 20x20 for brevity):\n";
        // 为避免文件过大，只打印左上角20x20的R通道数据作为验证示例
        // 如果老师要求全部，请去掉 min 限制
        int print_h = std::min(20, outImg.height);
        int print_w = std::min(20, outImg.width);
        for(int y=0; y<print_h; y++) {
            for(int x=0; x<print_w; x++) {
                matFile << (int)outImg.data[y * outImg.width + x].r << " ";
            }
            matFile << "\n";
        }
        matFile.close();
        std::cout << "Kernel and matrix data saved to " << matrix_file << std::endl;
    }

    return 0;
}