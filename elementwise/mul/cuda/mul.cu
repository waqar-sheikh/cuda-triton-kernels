#include <cuda_runtime.h>

__global__
void mul_forward(const float* a, const float* b, float* out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__
void mul_backward(const float* grad_out, const float* a, const float* b, float* grad_a, float* grad_b, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_a[idx] = grad_out[idx] * b[idx];
        grad_b[idx] = grad_out[idx] * a[idx];
    }
}

extern "C"
void launch_mul_forward(const float* a, const float* b, float* out, int size)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    mul_forward<<<blocks, threads>>>(a, b, out, size);
}

extern "C"
void launch_mul_backward(const float* grad_out, const float* a, const float* b, float* grad_a, float* grad_b, int size)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    mul_backward<<<blocks, threads>>>(grad_out, a, b, grad_a, grad_b, size);
}