#include <cuda_runtime.h>

template <int block_size>
__global__
void matmul_sharedmem(const float *A, const float *B, float *C, int M, int K, int N) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int cx = blockIdx.x*block_size + tx;
    int cy = blockIdx.y*block_size + ty;
    int num_blocks = K / block_size;
    float thread_sum = 0;

    __shared__ float smem_a[block_size * block_size];
    __shared__ float smem_b[block_size * block_size];

    for (int i = 0; i < num_blocks; i ++) {
        smem_a[ty*block_size + tx] = A[cy*K + block_size*i + tx];
        smem_b[ty*block_size + tx] = B[block_size*i*N + ty*N + cx];
        __syncthreads();

        for (int j = 0; j < block_size; j ++) {
            thread_sum += smem_a[ty*block_size + j] * smem_b[j*block_size + tx];
        }
        __syncthreads();
    }
    C[cy*N + cx] = thread_sum;
}

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
void launch_matmul_sharedmem(const float *A, const float *B, float *C, int M, int K, int N) {
    const int block_size = 16;
    int blocks_x = (N + block_size - 1) / block_size;
    int blocks_y = (M + block_size - 1) / block_size;

    dim3 gridDim(blocks_x, blocks_y, 1);
    dim3 blockDim(block_size, block_size, 1);
    matmul_sharedmem<block_size><<<gridDim, blockDim>>>(A, B, C, M, K, N);
}

extern "C"
void launch_matmul_naive(const float *A, const float *B, float *C, int M, int K, int N) {
    int threads = 16;
    int blocks_x = (N + threads - 1) / threads;
    int blocks_y = (M + threads - 1) / threads;

    dim3 gridDim(blocks_x, blocks_y, 1);
    dim3 blockDim(threads, threads, 1);
    matmul_naive<<<gridDim, blockDim>>>(A, B, C, M, K, N);
}
