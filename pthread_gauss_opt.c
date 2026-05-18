#define _GNU_SOURCE
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <arm_neon.h>

typedef struct {
    float *A;
    float *b;
    int n;
    int num_threads;
    int next_row;  // 动态任务分配
    pthread_mutex_t mutex;
} SharedData;

// SIMD 行更新
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

void* worker_optimized(void* arg) {
    SharedData* shared = (SharedData*)arg;
    float *A = shared->A;
    float *b = shared->b;
    int n = shared->n;
    
    while (1) {
        pthread_mutex_lock(&shared->mutex);
        int i = shared->next_row++;
        pthread_mutex_unlock(&shared->mutex);
        
        if (i >= n) break;
        
        // 处理第 i 行（跳过已处理的部分）
        // 实际实现需要传入当前 k
    }
    return NULL;
}

// 简化的正确版本：每个线程处理固定连续行
void* worker_fixed(void* arg) {
    int* data = (int*)arg;
    // 占位
    return NULL;
}

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

long long get_time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000LL + ts.tv_nsec / 1000;
}

// 优化版本：减少同步开销，使用更粗粒度的任务划分
void gauss_pthread_opt(float *A, float *b, float *x, int n, int num_threads) {
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
        
        // 并行消去剩余行 - 使用 OpenMP 风格的简单并行
        // 由于没有 OpenMP，这里用 Pthread 但减少 barrier
        
        int rows_left = n - k - 1;
        if (rows_left <= 0) continue;
        
        // 为简单起见，这里暂时用串行
        // 实际优化版本需要仔细设计
        for (int i = k + 1; i < n; i++) {
            float factor = A[i * n + k];
            if (fabs(factor) < 1e-12) continue;
            row_update_simd(&A[i * n], &A[k * n], factor, k + 1, n);
            A[i * n + k] = 0.0f;
            b[i] -= factor * b[k];
        }
    }
    back_substitution(A, b, x, n);
}

int main(int argc, char* argv[]) {
    int N = 1024;
    int num_threads = 4;
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) num_threads = atoi(argv[2]);
    
    float *A = (float*)aligned_alloc(16, N * N * sizeof(float));
    float *b = (float*)aligned_alloc(16, N * sizeof(float));
    float *x = (float*)aligned_alloc(16, N * sizeof(float));
    
    generate_matrix(A, b, N);
    
    long long st = get_time_us();
    gauss_pthread_opt(A, b, x, N, num_threads);
    long long ed = get_time_us();
    
    printf("Pthread Optimized (SIMD only, serial loop), N=%d, time=%lld us\n", N, ed - st);
    
    free(A);
    free(b);
    free(x);
    
    return 0;
}