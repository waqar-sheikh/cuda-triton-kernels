#include <torch/extension.h>

extern "C" void launch_mul_forward(const float* a, const float* b, float* out, int size);
extern "C" void launch_mul_backward(const float* grad_out, const float* a, const float* b, float* grad_a, float* grad_b,
                                    int size);


torch::Tensor cuda_mul_forward(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.device().is_cuda(), "Tensor a must be on CUDA");
    TORCH_CHECK(b.device().is_cuda(), "Tensor b must be on CUDA");
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensor sizes must match");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "Tensors must be float32");

    auto out = torch::empty_like(a);
    launch_mul_forward(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), a.numel());
    return out;
}

std::vector<torch::Tensor> cuda_mul_backward(torch::Tensor grad_out, torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(grad_out.device().is_cuda(), "grad_out must be on CUDA");
    TORCH_CHECK(a.device().is_cuda(), "Tensor a must be on CUDA");
    TORCH_CHECK(b.device().is_cuda(), "Tensor b must be on CUDA");
    TORCH_CHECK(grad_out.sizes() == a.sizes(), "grad_out and a sizes must match");
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensor sizes must match");
    TORCH_CHECK(grad_out.dtype() == torch::kFloat32, "Tensors must be float32");

    grad_out = grad_out.contiguous();
    auto grad_a = torch::empty_like(a);
    auto grad_b = torch::empty_like(b);

    launch_mul_backward(grad_out.data_ptr<float>(), a.data_ptr<float>(), b.data_ptr<float>(), grad_a.data_ptr<float>(),
                        grad_b.data_ptr<float>(), a.numel());
    return {grad_a, grad_b};
}