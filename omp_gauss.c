#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <arm_neon.h>
#include <omp.h>

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

// 获取时间（微秒）
long long get_time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000LL + ts.tv_nsec / 1000;
}

// OpenMP + SIMD 高斯消去
void gauss_omp_simd(float *A, float *b, float *x, int n) {
    for (int k = 0; k < n; k++) {
        // 主元归一化（串行，避免竞争）
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

        // 行消去（OpenMP 并行，dynamic 调度）
        #pragma omp parallel for schedule(dynamic, 16)
        for (int i = k + 1; i < n; i++) {
            float factor = A[i * n + k];
            if (fabs(factor) < 1e-12) continue;
            
            // SIMD 向量化
            float32x4_t v_factor = vdupq_n_f32(factor);
            int j = k + 1;
            for (; j + 4 <= n; j += 4) {
                float32x4_t v_ai = vld1q_f32(&A[i * n + j]);
                float32x4_t v_ak = vld1q_f32(&A[k * n + j]);
                v_ai = vsubq_f32(v_ai, vmulq_f32(v_factor, v_ak));
                vst1q_f32(&A[i * n + j], v_ai);
            }
            for (; j < n; j++) {
                A[i * n + j] -= factor * A[k * n + j];
            }
            
            A[i * n + k] = 0.0f;
            b[i] -= factor * b[k];
        }
    }

    // 回代
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

int main(int argc, char* argv[]) {
    int N = 1024;
    int num_threads = 4;
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) num_threads = atoi(argv[2]);
    
    omp_set_num_threads(num_threads);
    
    float *A = (float*)aligned_alloc(16, N * N * sizeof(float));
    float *b = (float*)aligned_alloc(16, N * sizeof(float));
    float *x = (float*)aligned_alloc(16, N * sizeof(float));
    
    generate_matrix(A, b, N);
    
    long long st = get_time_us();
    gauss_omp_simd(A, b, x, N);
    long long ed = get_time_us();
    
    printf("OpenMP + SIMD, N=%d, threads=%d, time=%lld us\n", N, num_threads, ed - st);
    
    free(A);
    free(b);
    free(x);
    
    return 0;
}