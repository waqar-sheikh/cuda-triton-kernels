#include <cuda_runtime.h>

__global__
void softmax_sharedmem(const float *input, float *output, const int nrows, const int ncols) {
    __shared__ float smem[128];
    __shared__ float max[1];
    __shared__ float sum[1];

    int tid = threadIdx.x;
    int row_start = blockIdx.x * ncols;
    float thread_max = -INFINITY;

    for (int i = row_start + tid; i < row_start + ncols; i += blockDim.x) {
        float elem = input[i];
        if (elem > thread_max)
            thread_max = elem;
    }
    smem[tid] = thread_max;
    __syncthreads();

    if (tid == 0) {
        for (int i = 0; i < blockDim.x; i ++) {
            if (smem[i] > max[0])
                max[0] = smem[i];
        }
    }
    __syncthreads();

    smem[tid] = 0;
    for (int i = row_start + tid; i < row_start + ncols; i += blockDim.x) {
        smem[tid] += expf(input[i] - max[0]);
    }
    __syncthreads();

    if (tid == 0) {
        sum[0] = 0;
        for (int i = 0; i < blockDim.x; i ++) {
            sum[0] += smem[i];
        }
    }
    __syncthreads();

    for (int i = row_start + tid; i < row_start + ncols; i += blockDim.x) {
        output[i] = exp(input[i] - max[0]) / sum[0];
    }
}

__global__
void softmax_online(const float *input, float *output, const int nrows, const int ncols) {

    int row_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (row_idx < nrows) {
        int start_idx = row_idx * ncols;
        float row_sum = 0;
        float row_max = -INFINITY;

        for (int i = start_idx; i < start_idx + ncols; i ++) {
            if (input[i] > row_max){
                row_sum *= expf(row_max - input[i]);
                row_max = input[i];
            }
            row_sum += expf(input[i] - row_max);
        }

        for (int i = start_idx; i < start_idx + ncols; i ++) {
            output[i] = expf(input[i] - row_max) / row_sum;
        }
    }
}

__global__
void softmax_naive(const float *input, float *output, const int nrows, const int ncols) {

    int row_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (row_idx < nrows) {
        int start_idx = row_idx * ncols;
        float row_sum = 0;
        float row_max = -INFINITY;

        for (int i = start_idx; i < start_idx + ncols; i ++) {
            if (input[i] > row_max)
                row_max = input[i];
        }
        
        for (int i = start_idx; i < start_idx + ncols; i ++) {
            row_sum += expf(input[i] - row_max);
        }
        
        for (int i = start_idx; i < start_idx + ncols; i ++) {
            output[i] = expf(input[i] - row_max) / row_sum;
        }
    }
}

extern "C"
void launch_softmax_sharedmem(const float *input, float *output, const int nrows, const int ncols) {

    int threads = 128;
    int blocks = nrows;
    softmax_sharedmem<<<blocks, threads>>>(input, output, nrows, ncols);
}

extern "C"
void launch_softmax_online(const float *input, float *output, const int nrows, const int ncols) {

    int threads = 128;
    int blocks = (nrows + threads - 1) / threads;
    softmax_online<<<blocks, threads>>>(input, output, nrows, ncols);
}

extern "C"
void launch_softmax_naive(const float *input, float *output, const int nrows, const int ncols) {

    int threads = 128;
    int blocks = (nrows + threads - 1) / threads;
    softmax_naive<<<blocks, threads>>>(input, output, nrows, ncols);
}