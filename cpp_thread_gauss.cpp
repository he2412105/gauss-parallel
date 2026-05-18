#define _GNU_SOURCE
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <arm_neon.h>

using namespace std;

typedef struct {
    float *A;
    float *b;
    int n;
    int start_row;
    int end_row;
    int k;
} ThreadData;

// SIMD 行更新函数
void row_update_simd(float *row_i, float *row_k, float factor, int start_col, int n) {
    float32x4_t v_factor = vdupq_n_f32(factor);
    int j = start_col;
    for (; j + 4 <= n; j += 4) {
        float32x4_t v_ai = vld1q_f32(&row_i[j]);
        float32x4_t v_ak = vld1q_f32(&row_k[j]);
        v_ai = vsubq_f32(v_ai, vmulq_f32(v_factor, v_ak));
        vst1q_f32(&row_i[j], v_ai);
    }
    for (; j < n; j++) {
        row_i[j] -= factor * row_k[j];
    }
}

// 生成对角占优矩阵
void generate_matrix(float *A, float *b, int n) {
    for (int i = 0; i < n; i++) {
        float sum = 0;
        for (int j = 0; j < n; j++) {
            if (i != j) {
                A[i * n + j] = (rand() % 100) / 100.0f;
                sum += fabs(A[i * n + j]);
            }
        }
        A[i * n + i] = sum + 1.0f;
        b[i] = rand() % 100;
    }
}

// 回代求解
void back_substitution(float *A, float *b, float *x, int n) {
    for (int i = n - 1; i >= 0; i--) {
        x[i] = b[i];
        float32x4_t v_sum = vdupq_n_f32(0.0f);
        int j = i + 1;
        for (; j + 4 <= n; j += 4) {
            float32x4_t v_a = vld1q_f32(&A[i * n + j]);
            float32x4_t v_x = vld1q_f32(&x[j]);
            v_sum = vmlaq_f32(v_sum, v_a, v_x);
        }
        float s[4];
        vst1q_f32(s, v_sum);
        float total = s[0] + s[1] + s[2] + s[3];
        for (; j < n; j++) {
            total += A[i * n + j] * x[j];
        }
        x[i] -= total;
    }
}

// 高斯消去主函数（多线程版本）
void gauss_cpp_thread(float *A, float *b, float *x, int n, int num_threads) {
    // 消去过程
    for (int k = 0; k < n; k++) {
        // 主元归一化（串行）
        float pivot = A[k * n + k];
        float32x4_t v_pivot = vdupq_n_f32(pivot);
        int j = k + 1;
        for (; j + 4 <= n; j += 4) {
            float32x4_t v_row = vld1q_f32(&A[k * n + j]);
            v_row = vdivq_f32(v_row, v_pivot);
            vst1q_f32(&A[k * n + j], v_row);
        }
        for (; j < n; j++) {
            A[k * n + j] /= pivot;
        }
        A[k * n + k] = 1.0f;
        b[k] /= pivot;

        // 计算每个线程处理的行范围
        int rows_to_process = n - k - 1;
        if (rows_to_process <= 0) continue;
        
        int rows_per_thread = rows_to_process / num_threads;
        int remainder = rows_to_process % num_threads;
        
        vector<thread> threads;
        
        // 创建线程
        int start_i = k + 1;
        for (int t = 0; t < num_threads; t++) {
            int end_i = start_i + rows_per_thread;
            if (t < remainder) end_i++;
            
            if (start_i < end_i) {
                threads.emplace_back([&, start_i, end_i, k]() {
                    for (int i = start_i; i < end_i; i++) {
                        float factor = A[i * n + k];
                        if (fabs(factor) < 1e-12) continue;
                        row_update_simd(&A[i * n], &A[k * n], factor, k + 1, n);
                        A[i * n + k] = 0.0f;
                        b[i] -= factor * b[k];
                    }
                });
            }
            start_i = end_i;
        }
        
        // 等待所有线程完成
        for (auto& t : threads) {
            t.join();
        }
    }
    
    back_substitution(A, b, x, n);
}

int main(int argc, char* argv[]) {
    int N = 1024;
    int num_threads = 4;
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) num_threads = atoi(argv[2]);
    
    // 16字节对齐分配内存
    float *A = (float*)aligned_alloc(16, N * N * sizeof(float));
    float *b = (float*)aligned_alloc(16, N * sizeof(float));
    float *x = (float*)aligned_alloc(16, N * sizeof(float));
    
    generate_matrix(A, b, N);
    
    auto st = chrono::high_resolution_clock::now();
    gauss_cpp_thread(A, b, x, N, num_threads);
    auto ed = chrono::high_resolution_clock::now();
    
    chrono::duration<double, micro> elapsed = ed - st;
    cout << "C++ Thread + SIMD, N=" << N << ", threads=" << num_threads 
         << ", time=" << elapsed.count() << " us" << endl;
    
    free(A);
    free(b);
    free(x);
    
    return 0;
}