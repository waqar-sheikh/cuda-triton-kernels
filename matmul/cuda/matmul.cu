#include <cuda_runtime.h>


template <int block_size, int tile_size>
__global__
void matmul_threadtiling(const float *A, const float *B, float *C, int M, int K, int N) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int num_blocks = K / block_size;
    int num_tiles = block_size / tile_size;
    int num_threads = (block_size*block_size) / (tile_size*tile_size);

    __shared__ float smem_a[block_size * block_size];
    __shared__ float smem_b[block_size * block_size];

    float reg_a[tile_size * tile_size];
    float reg_b[tile_size * tile_size];
    float result[tile_size * tile_size] = {0.0};

    for (int k_index = 0; k_index < num_blocks; k_index ++) {

        int block_start_a = by*block_size*K + k_index*block_size;
        int block_start_b = k_index*block_size*N + bx*block_size;
        int col_start = ty*(block_size/tile_size) + tx;

        for (int col = col_start; col < block_size; col += num_threads) {
            for (int row = 0; row <  block_size; row ++) {

                smem_a[col + row*block_size] = A[block_start_a + col + row*K];
                smem_b[col + row*block_size] = B[block_start_b + col + row*N];
            }
        }
        __syncthreads();

        for (int tile_index = 0; tile_index < num_tiles; tile_index ++) {

            for (int m = 0; m < tile_size; m ++) {
                for (int n = 0; n < tile_size; n ++) {

                    reg_a[m*tile_size + n] =
                        smem_a[ty*tile_size*block_size + m*block_size + tile_index*tile_size + n];

                    reg_b[m*tile_size + n] =
                        smem_b[tile_index*tile_size*block_size + m*block_size + tx*tile_size + n];
                }
            }

            for (int m = 0; m < tile_size; m ++) {
                for (int n = 0; n < tile_size; n ++) {
                    for (int o = 0; o < tile_size; o ++) {
                        result[m*tile_size + n] += reg_a[m*tile_size + o] * reg_b[n + o*tile_size];
                    }
                }
            }
        }
        __syncthreads();
    }

    for (int m = 0; m < tile_size; m ++) {
        for (int n = 0; n < tile_size; n ++) {
            C[by*block_size*N + ty*tile_size*N + m*N + bx*block_size + tx*tile_size + n] =
                result[m*tile_size + n];
        }
    }
}

template <int block_size>
__global__
void matmul_blocktiling(const float *A, const float *B, float *C, int M, int K, int N) {
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
void launch_matmul_threadtiling(const float *A, const float *B, float *C, int M, int K, int N) {
    const int block_size = 32;
    const int tile_size = 4;
    int blocks_x = (N + block_size - 1) / block_size;
    int blocks_y = (M + block_size - 1) / block_size;

    dim3 gridDim(blocks_x, blocks_y, 1);
    dim3 blockDim(block_size/tile_size, block_size/tile_size, 1);
    matmul_threadtiling<block_size, tile_size><<<gridDim, blockDim>>>(A, B, C, M, K, N);
}

extern "C"
void launch_matmul_blocktiling(const float *A, const float *B, float *C, int M, int K, int N) {
    const int block_size = 16;
    int blocks_x = (N + block_size - 1) / block_size;
    int blocks_y = (M + block_size - 1) / block_size;

    dim3 gridDim(blocks_x, blocks_y, 1);
    dim3 blockDim(block_size, block_size, 1);
    matmul_blocktiling<block_size><<<gridDim, blockDim>>>(A, B, C, M, K, N);
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
