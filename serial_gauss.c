#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

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

void gauss_serial(float *A, float *b, float *x, int n) {
    for (int k = 0; k < n; k++) {
        float pivot = A[k * n + k];
        for (int j = k + 1; j < n; j++)
            A[k * n + j] /= pivot;
        A[k * n + k] = 1.0f;
        b[k] /= pivot;

        for (int i = k + 1; i < n; i++) {
            float factor = A[i * n + k];
            for (int j = k + 1; j < n; j++)
                A[i * n + j] -= factor * A[k * n + j];
            A[i * n + k] = 0.0f;
            b[i] -= factor * b[k];
        }
    }

    for (int i = n - 1; i >= 0; i--) {
        x[i] = b[i];
        for (int j = i + 1; j < n; j++)
            x[i] -= A[i * n + j] * x[j];
    }
}

long long get_time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000LL + ts.tv_nsec / 1000;
}

int main(int argc, char* argv[]) {
    int N = 1024;
    if (argc > 1) N = atoi(argv[1]);
    
    float *A = (float*)aligned_alloc(16, N * N * sizeof(float));
    float *b = (float*)aligned_alloc(16, N * sizeof(float));
    float *x = (float*)aligned_alloc(16, N * sizeof(float));
    
    generate_matrix(A, b, N);
    
    long long st = get_time_us();
    gauss_serial(A, b, x, N);
    long long ed = get_time_us();
    
    printf("Serial, N=%d, time=%lld us\n", N, ed - st);
    
    free(A);
    free(b);
    free(x);
    
    return 0;
}