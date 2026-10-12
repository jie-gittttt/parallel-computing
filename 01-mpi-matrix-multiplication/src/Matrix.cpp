#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <mpi.h>

using namespace std;
// 矩阵维度
const int M = 1023;
const int N = 2047;
const int K = 1023;

// 使用 1D vector 来模拟 2D 矩阵 (更易于MPI传输)
using Matrix = vector<long long>;

// 辅助函数：从文件读取矩阵
void readMatrices(const string& filename, Matrix& A, Matrix& B, Matrix& C) {
    ifstream file(filename);
    string line;
    string header;
    long long value;

    // 读取 A
    getline(file, header); // 跳过 #A
    A.resize(M * N);
    for (int i = 0; i < M; ++i) {
        getline(file, line);
        stringstream ss(line);
        for (int j = 0; j < N; ++j) {
            ss >> A[i * N + j];
        }
    }

    // 读取 B
    getline(file, header); // 跳过 #B
    B.resize(N * K);
    for (int i = 0; i < N; ++i) {
        getline(file, line);
        stringstream ss(line);
        for (int j = 0; j < K; ++j) {
            ss >> B[i * K + j];
        }
    }

    // 读取 C
    getline(file, header); // 跳过 #C
    C.resize(M * K);
    for (int i = 0; i < M; ++i) {
        getline(file, line);
        stringstream ss(line);
        for (int j = 0; j < K; ++j) {
            ss >> C[i * K + j];
        }
    }
    file.close();
}

// 辅助函数：将结果矩阵 D 写入文件
void writeResult(const string& filename, const Matrix& D) {
    ofstream file(filename);
    file << "#D\n";
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            file << D[i * K + j] << (j == K - 1 ? "" : " ");
        }
        file << "\n";
    }
    file.close();
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);// 启动MPI环境

    int world_rank, world_size;
    // 获取当前进程编号（0-3）
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // 获取总进程数（4）
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    Matrix A, B, C, D;
    double start_time, end_time;

    if (world_rank == 0) {
        start_time = MPI_Wtime();
        cout << "Running with " << world_size << " processes." << endl;
        cout << "Reading matrices from RandMatrix.txt..." << endl;
        // 进程 0 读取所有数据
        readMatrices("RandMatrix.txt", A, B, C);
        D.resize(M * K);
        cout << "Matrices read. Starting timer and scattering data." << endl;
        // 启动计时器 (包含I/O和通信)
        // start_time = MPI_Wtime();
    } else {
        // 其他进程分配 B 的空间
        B.resize(N * K);
    }

    // --- 1. 计算 Scatterv 所需的参数 ---
    // 按行分割 M (1023)
    vector<int> sendcounts_A(world_size); // A 和 C 的行数 (A是M*N, C是M*K)
    vector<int> displs_A(world_size);
    vector<int> sendcounts_C(world_size);
    vector<int> displs_C(world_size);

    int base_rows = M / world_size;
    int remainder = M % world_size;
    int current_row = 0;

    for (int i = 0; i < world_size; ++i) {
        int rows_for_this_proc = base_rows + (i < remainder ? 1 : 0);
        
        sendcounts_A[i] = rows_for_this_proc * N; // A 的元素个数
        displs_A[i] = current_row * N;           // A 的偏移量
        
        sendcounts_C[i] = rows_for_this_proc * K; // C 和 D 的元素个数
        displs_C[i] = current_row * K;           // C 和 D 的偏移量

        current_row += rows_for_this_proc;
    }
    
    // 本地进程接收的行数
    int local_rows = base_rows + (world_rank < remainder ? 1 : 0);
    Matrix A_local(local_rows * N);
    Matrix C_local(local_rows * K);
    Matrix D_local(local_rows * K); // 存储本地计算结果

    // --- 2. 分发数据 ---
    // 使用 Scatterv 分发 A
    MPI_Scatterv(
        world_rank == 0 ? A.data() : nullptr, // 仅 rank 0 发送
        sendcounts_A.data(),
        displs_A.data(),
        MPI_LONG_LONG,                         // 数据类型
        A_local.data(),                        // 接收缓冲区
        sendcounts_A[world_rank],              // 接收的元素个数
        MPI_LONG_LONG,
        0,                                     // root 进程
        MPI_COMM_WORLD
    );

    // 使用 Scatterv 分发 C
    MPI_Scatterv(
        world_rank == 0 ? C.data() : nullptr,
        sendcounts_C.data(),
        displs_C.data(),
        MPI_LONG_LONG,
        C_local.data(),
        sendcounts_C[world_rank],
        MPI_LONG_LONG,
        0,
        MPI_COMM_WORLD
    );

    // 广播完整的矩阵 B
    MPI_Bcast(
        B.data(),       // 发送/接收缓冲区
        N * K,          // 元素个数
        MPI_LONG_LONG,  // 数据类型
        0,              // root 进程
        MPI_COMM_WORLD
    );

    // --- 3. 本地计算 D_local = A_local * B + C_local ---
    if (world_rank == 0) {
        cout << "Data scattered. Starting local computations..." << endl;
    }
    
    for (int i = 0; i < local_rows; ++i) {
        for (int j = 0; j < K; ++j) {
            long long sum = 0;
            for (int k = 0; k < N; ++k) {
                // A_local[i][k] * B[k][j]
                sum += A_local[i * N + k] * B[k * K + j];
            }
            // D_local[i][j] = sum + C_local[i][j]
            D_local[i * K + j] = sum + C_local[i * K + j];
        }
    }

    if (world_rank == 0) {
        cout << "Local computations complete. Gathering results..." << endl;
    }

    // --- 4. 收集结果 ---
    // 使用 Gatherv 将所有 D_local 收集到进程 0 的 D 中
    // Gatherv 的 sendcounts 和 displs 与 Scatterv (C) 相同
    MPI_Gatherv(
        D_local.data(),             // 发送缓冲区
        sendcounts_C[world_rank],   // 发送的元素个数
        MPI_LONG_LONG,
        world_rank == 0 ? D.data() : nullptr, // 仅 rank 0 接收
        sendcounts_C.data(),        // 接收计数 (仅 root 需要)
        displs_C.data(),            // 偏移量 (仅 root 需要)
        MPI_LONG_LONG,
        0,                          // root 进程
        MPI_COMM_WORLD
    );

    // --- 5. 结束和输出 ---
    if (world_rank == 0) {
        //end_time = MPI_Wtime();
        
        
        cout << "Results gathered. Writing to result.txt..." << endl;
        writeResult("result.txt", D);

        end_time = MPI_Wtime();
        double total_time = end_time - start_time;
        
        cout << "------------------------------------------" << endl;
        cout << "Job Complete." << endl;
        printf("Total execution time with %d processes: %.6f seconds\n", world_size, total_time);
        cout << "------------------------------------------" << endl;
    }

    MPI_Finalize();
    return 0;
}