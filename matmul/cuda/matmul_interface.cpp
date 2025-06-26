#include <string>
#include <iostream>
#include <torch/extension.h>

extern "C" void launch_matmul_naive(const float *A, const float *B, float *C, int M, int K, int N);

torch::Tensor cuda_matmul_forward(torch::Tensor A, torch::Tensor B, std::string impl) {
    TORCH_CHECK(A.device().is_cuda() and B.device().is_cuda(), "Input tensors must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat32 and B.dtype() == torch::kFloat32, "Input tensors must be float32");

    auto output = torch::empty({A.size(0), B.size(1)}, torch::TensorOptions().device(torch::kCUDA));

    if (impl == "naive") {
        launch_matmul_naive(A.data_ptr<float>(), B.data_ptr<float>(), output.data_ptr<float>(),
            A.size(0), A.size(1), B.size(1));
    }
    return output;
}