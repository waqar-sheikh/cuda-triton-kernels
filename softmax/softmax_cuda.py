import torch
from torch import autograd
from torch.utils.cpp_extension import load_inline

with open("cuda/softmax.cu", "r") as f:
    cuda_source = f.read()

with open("cuda/softmax_interface.cpp", "r") as f:
    cpp_source = f.read()

ext = load_inline(
    name="cuda_softmax",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["cuda_softmax_forward"],
    extra_cflags=["-std=c++17"],
    extra_cuda_cflags=["-std=c++17"],
    verbose=True
)

class Softmax(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return ext.cuda_softmax_forward(input)