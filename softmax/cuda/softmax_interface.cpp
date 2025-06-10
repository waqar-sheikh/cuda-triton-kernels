#include <torch/extension.h>

extern "C" void launch_softmax_forward(const float *input, float *output, const int nrows, const int ncols);

torch::Tensor cuda_softmax_forward(torch::Tensor input) {
    TORCH_CHECK(input.device().is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be float32");

    auto output = torch::empty_like(input);
    launch_softmax_forward(input.data_ptr<float>(), output.data_ptr<float>(), input.size(0), input.size(1));
    return output;
}