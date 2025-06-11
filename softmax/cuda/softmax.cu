#include <cuda_runtime.h>

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
void launch_softmax_online(const float *input, float *output, const int nrows, const int ncols) {

    int threads = 256;
    int blocks = (nrows + threads - 1) / threads;
    softmax_online<<<blocks, threads>>>(input, output, nrows, ncols);
}

extern "C"
void launch_softmax_naive(const float *input, float *output, const int nrows, const int ncols) {

    int threads = 256;
    int blocks = (nrows + threads - 1) / threads;
    softmax_naive<<<blocks, threads>>>(input, output, nrows, ncols);
}