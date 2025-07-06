#include <string>
#include <iostream>
#include <torch/extension.h>

extern "C" void launch_matmul_naive(const float *A, const float *B, float *C, int M, int K, int N);
extern "C" void launch_matmul_blocktiling(const float *A, const float *B, float *C, int M, int K, int N);
extern "C" void launch_matmul_threadtiling(const float *A, const float *B, float *C, int M, int K, int N);


torch::Tensor cuda_matmul_forward(torch::Tensor A, torch::Tensor B, std::string impl) {
    TORCH_CHECK(A.device().is_cuda() and B.device().is_cuda(), "Input tensors must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat32 and B.dtype() == torch::kFloat32, "Input tensors must be float32");

    auto output = torch::zeros({A.size(0), B.size(1)}, torch::TensorOptions().device(torch::kCUDA));

    if (impl == "threadtiling") {
        launch_matmul_threadtiling(A.data_ptr<float>(), B.data_ptr<float>(), output.data_ptr<float>(), A.size(0), A.size(1),
            B.size(1));
    } else if (impl == "blocktiling") {
        launch_matmul_blocktiling(A.data_ptr<float>(), B.data_ptr<float>(), output.data_ptr<float>(), A.size(0), A.size(1),
            B.size(1));
    } else if (impl == "naive") {
        launch_matmul_naive(A.data_ptr<float>(), B.data_ptr<float>(), output.data_ptr<float>(), A.size(0), A.size(1),
            B.size(1));
    } else {
        TORCH_CHECK(false, "Unknown matmul implementation: ", impl);
    }
    return output;
}