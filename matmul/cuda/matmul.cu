#include <cuda_runtime.h>

__global__
void matmul_naive(const float *A, const float *B, float *C, int M, int K, int N) {

    int x = (blockDim.x * blockIdx.x) + threadIdx.x;
    int y = (blockDim.y * blockIdx.y) + threadIdx.y;

    if (x < N && y < M) {
        float dot = 0;

        for (int i = 0; i < K; i ++) {
            dot += A[(y * K) + i] * B[x + (i * N)];
        }
        C[(y * N) + x] = dot;
    }
}

extern "C"
void launch_matmul_naive(const float *A, const float *B, float *C, int M, int K, int N) {
    int threads = 32;
    int blocks_x = (N + threads - 1) / threads;
    int blocks_y = (M + threads - 1) / threads;

    dim3 gridDim(blocks_x, blocks_y, 1);
    dim3 blockDim(threads, threads, 1);
    matmul_naive<<<gridDim, blockDim>>>(A, B, C, M, K, N);
}