#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <pthread.h>
#include <sys/time.h>
#include <iomanip>
#include <cstring>
#include <algorithm>

// ------------------- 1. BMP 基础结构定义 -------------------
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

// 用于中间结果的高精度像素结构
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

// ------------------- 2. 可分离卷积核 (1D Gaussian) -------------------
// 根据提供的2D核分解得到。Sum = 1.0
const int K_RADIUS = 2; // 半径
const int K_SIZE = 5;   // 长度
// 1D 权重向量
const float KERNEL_1D[5] = {0.120078f, 0.233880f, 0.292082f, 0.233880f, 0.120078f};

// ------------------- 3. 线程数据与同步 -------------------
struct ThreadData {
    int thread_id;
    int num_threads;
    BMPImage* srcImg;          // 源图
    std::vector<PixelFloat>* tempImg; // 中间结果(Float)
    BMPImage* destImg;         // 最终结果
    pthread_barrier_t* barrier; // 线程栅栏
    double cost_time;          // 耗时记录
};

double get_wall_time() {
    struct timeval time;
    if (gettimeofday(&time, NULL)) return 0;
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

// ------------------- 4. 并行工作函数 -------------------
void* separable_convolution_routine(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    double start_time = get_wall_time();

    int height = data->srcImg->height;
    int width = data->srcImg->width;

    // 任务划分：按行分块
    int chunk_size = height / data->num_threads;
    int start_row = data->thread_id * chunk_size;
    int end_row = (data->thread_id == data->num_threads - 1) ? height : start_row + chunk_size;

    // ================== 阶段 1: 水平卷积 (Row Convolution) ==================
    // 输入: srcImg (uchar) -> 输出: tempImg (float)
    // 对每一行，应用 1x5 卷积核
    for (int y = start_row; y < end_row; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
            
            for (int k = -K_RADIUS; k <= K_RADIUS; ++k) {
                int nx = x + k;
                float w = KERNEL_1D[k + K_RADIUS];

                // 边界 Zero Padding
                if (nx >= 0 && nx < width) {
                    Pixel& p = data->srcImg->data[y * width + nx];
                    sum_r += p.r * w;
                    sum_g += p.g * w;
                    sum_b += p.b * w;
                }
            }
            // 存入 float 类型的中间 buffer，防止精度丢失
            (*data->tempImg)[y * width + x] = {sum_b, sum_g, sum_r};
        }
    }

    // ================== 线程同步 (Barrier) ==================
    // 必须等待所有线程完成水平卷积，才能开始垂直卷积
    // 因为垂直卷积需要读取上下行的中间结果
    pthread_barrier_wait(data->barrier);

    // ================== 阶段 2: 垂直卷积 (Column Convolution) ==================
    // 输入: tempImg (float) -> 输出: destImg (uchar)
    // 对每一列，应用 5x1 卷积核
    for (int y = start_row; y < end_row; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;

            for (int k = -K_RADIUS; k <= K_RADIUS; ++k) {
                int ny = y + k;
                float w = KERNEL_1D[k + K_RADIUS];

                // 边界 Zero Padding
                if (ny >= 0 && ny < height) {
                    PixelFloat& p = (*data->tempImg)[ny * width + x];
                    sum_r += p.r * w;
                    sum_g += p.g * w;
                    sum_b += p.b * w;
                }
            }

            // 最终截断并存入结果
            int idx = y * width + x;
            data->destImg->data[idx].r = (unsigned char)std::min(std::max(sum_r, 0.0f), 255.0f);
            data->destImg->data[idx].g = (unsigned char)std::min(std::max(sum_g, 0.0f), 255.0f);
            data->destImg->data[idx].b = (unsigned char)std::min(std::max(sum_b, 0.0f), 255.0f);
        }
    }

    double end_time = get_wall_time();
    data->cost_time = end_time - start_time;
    printf("Thread %d calculation time: %.6f seconds\n", data->thread_id, data->cost_time);

    pthread_exit(NULL);
}

// ------------------- 5. 主函数 -------------------
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./pthread_separable <num_threads>" << std::endl;
        return 1;
    }

    int num_threads = std::stoi(argv[1]);
    const char* input_file = "timg.bmp";
    const char* output_file = "output.bmp";
    const char* matrix_file = "matrix.txt"; // 虽然是可分离，依然输出矩阵供验证

    BMPImage img;
    if (!img.load(input_file)) {
        std::cerr << "Error: Cannot load " << input_file << std::endl;
        return 1;
    }

    BMPImage outImg = img;
    // 申请中间缓冲区 (Float精度)
    std::vector<PixelFloat> tempBuffer(img.width * img.height);

    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];
    pthread_barrier_t barrier;

    // 初始化 Barrier
    pthread_barrier_init(&barrier, NULL, num_threads);

    std::cout << "Starting SEPARABLE convolution with " << num_threads << " threads..." << std::endl;
    double main_start = get_wall_time();

    for (int i = 0; i < num_threads; ++i) {
        thread_data[i].thread_id = i;
        thread_data[i].num_threads = num_threads;
        thread_data[i].srcImg = &img;
        thread_data[i].tempImg = &tempBuffer;
        thread_data[i].destImg = &outImg;
        thread_data[i].barrier = &barrier;
        
        int rc = pthread_create(&threads[i], NULL, separable_convolution_routine, (void*)&thread_data[i]);
        if (rc) {
            std::cerr << "Error: unable to create thread," << rc << std::endl;
            return 1;
        }
    }

    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }

    double main_end = get_wall_time();
    
    // 销毁 Barrier
    pthread_barrier_destroy(&barrier);

    std::cout << "Total Parallel Execution Time: " << (main_end - main_start) << " seconds" << std::endl;

    if (outImg.save(output_file)) {
        std::cout << "Image saved to " << output_file << std::endl;
    }

    // ------------------- 验证输出 (与作业要求保持一致) -------------------
    // 虽然我们用了 1D 核，但为了验证正确性，我们还是输出结果矩阵
    // 注意：这里不再输出 5x5 的 2D 核，因为代码里没用它。
    std::ofstream matFile(matrix_file);
    if (matFile.is_open()) {
        matFile << "Separable Convolution Used (1D Kernel):\n";
        for(int i=0; i<5; i++) matFile << KERNEL_1D[i] << " ";
        matFile << "\n\nOutput Pixel Matrix (R Channel sample):\n";
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