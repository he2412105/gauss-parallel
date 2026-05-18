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
    int start_row;
    int end_row;
    int k;  // 当前消去步
} ThreadData;

pthread_barrier_t barrier;

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

void* worker(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int n = data->n;
    float *A = data->A;
    float *b = data->b;
    int start = data->start_row;
    int end = data->end_row;
    
    for (int k = 0; k < n; k++) {
        // 只有负责第 k 行的线程做归一化
        if (k >= start && k < end) {
            float pivot = A[k * n + k];
            // 归一化
            for (int j = k + 1; j < n; j++) {
                A[k * n + j] /= pivot;
            }
            A[k * n + k] = 1.0;
            b[k] /= pivot;
        }
        
        pthread_barrier_wait(&barrier);
        
        // 各线程消去自己负责的行
        for (int i = (start > k+1 ? start : k+1); i < end; i++) {
            float factor = A[i * n + k];
            if (factor == 0) continue;
            row_update_simd(&A[i * n], &A[k * n], factor, k+1, n);
            A[i * n + k] = 0;
            b[i] -= factor * b[k];
        }
        
        pthread_barrier_wait(&barrier);
    }
    return NULL;
}

void back_substitution(float *A, float *b, float *x, int n) {
    for (int i = n-1; i >= 0; i--) {
        x[i] = b[i];
        for (int j = i+1; j < n; j++) {
            x[i] -= A[i * n + j] * x[j];
        }
    }
}

void generate_matrix(float *A, float *b, int n) {
    for (int i = 0; i < n; i++) {
        float sum = 0;
        for (int j = 0; j < n; j++) {
            if (i != j) {
                A[i * n + j] = (rand() % 100) / 100.0;
                sum += fabs(A[i * n + j]);
            }
        }
        A[i * n + i] = sum + 1.0;
        b[i] = rand() % 100;
    }
}

long long get_time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000LL + ts.tv_nsec / 1000;
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
    
    pthread_t threads[num_threads];
    ThreadData thread_data[num_threads];
    
    pthread_barrier_init(&barrier, NULL, num_threads);
    
    int rows_per_thread = N / num_threads;
    int remainder = N % num_threads;
    int start = 0;
    
    long long st = get_time_us();
    
    for (int t = 0; t < num_threads; t++) {
        thread_data[t].A = A;
        thread_data[t].b = b;
        thread_data[t].n = N;
        thread_data[t].start_row = start;
        int extra = (t < remainder) ? 1 : 0;
        thread_data[t].end_row = start + rows_per_thread + extra;
        start = thread_data[t].end_row;
        pthread_create(&threads[t], NULL, worker, &thread_data[t]);
    }
    
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }
    
    pthread_barrier_destroy(&barrier);
    
    back_substitution(A, b, x, N);
    
    long long ed = get_time_us();
    printf("Pthread + SIMD, N=%d, threads=%d, time=%lld us\n", N, num_threads, ed - st);
    
    free(A);
    free(b);
    free(x);
    return 0;
}