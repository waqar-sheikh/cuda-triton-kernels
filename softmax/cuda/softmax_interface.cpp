#include <string>
#include <iostream>
#include <torch/extension.h>

extern "C" void launch_softmax_naive(const float *input, float *output, const int nrows, const int ncols);
extern "C" void launch_softmax_online(const float *input, float *output, const int nrows, const int ncols);
extern "C" void launch_softmax_sharedmem(const float *input, float *output, const int nrows, const int ncols);

torch::Tensor cuda_softmax_forward(torch::Tensor input, std::string impl) {
    TORCH_CHECK(input.device().is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be float32");

    auto output = torch::empty_like(input);

    if (impl == "sharedmem") {
        launch_softmax_sharedmem(input.data_ptr<float>(), output.data_ptr<float>(), input.size(0), input.size(1));
    } else if (impl == "online") {
        launch_softmax_online(input.data_ptr<float>(), output.data_ptr<float>(), input.size(0), input.size(1));
    } else if (impl == "naive") {
        launch_softmax_naive(input.data_ptr<float>(), output.data_ptr<float>(), input.size(0), input.size(1));
    } else {
        TORCH_CHECK(false, "Unknown softmax implementation: ", impl);
    }
    return output;
}