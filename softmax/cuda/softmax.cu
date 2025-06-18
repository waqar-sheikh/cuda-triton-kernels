#include <cuda_runtime.h>

__global__
void softmax_sharedmem(const float *input, float *output, const int nrows, const int ncols) {
    __shared__ float smem[128];

    int tid = threadIdx.x;
    int row_start = blockIdx.x * ncols;
    float thread_max = -INFINITY;
    float thread_sum = 0;

    for (int i = row_start + tid; i < row_start + ncols; i += blockDim.x) {
        float elem = input[i];
        if (elem > thread_max) {
            thread_sum *= expf(thread_max - elem);
            thread_max = elem;
        }
        thread_sum += expf(elem - thread_max);
    }

    smem[tid] = thread_max;
    __syncthreads();

    for (int n_reduce = blockDim.x; n_reduce > 1; n_reduce /= 2) {
        if (tid < n_reduce / 2) {
            if (smem[tid + n_reduce/2] > smem[tid])
                smem[tid] = smem[tid + n_reduce/2];
        }
        __syncthreads();
    }

    float row_max = smem[0];
    __syncthreads();

    smem[tid] = thread_sum * expf(thread_max - row_max);
    __syncthreads();

    for (int n_reduce = blockDim.x; n_reduce > 1; n_reduce /= 2) {
        if (tid < n_reduce / 2) {
            smem[tid] += smem[tid + n_reduce/2];
        }
        __syncthreads();
    }

    float row_sum = smem[0];
    __syncthreads();

    for (int i = row_start + tid; i < row_start + ncols; i += blockDim.x) {
        output[i] = exp(input[i] - row_max) / row_sum;
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