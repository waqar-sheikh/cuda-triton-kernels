import torch
from torch import autograd
from torch.utils.cpp_extension import load_inline

with open("cuda/add.cu", "r") as f:
    cuda_source = f.read()

with open("cuda/add_interface.cpp", "r") as f:
    cpp_source = f.read()

extension = load_inline(
        name="cuda_add",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["cuda_add_forward", "cuda_add_backward"],
        extra_cflags=["-std=c++17"],
        extra_cuda_cflags=["-std=c++17"],
        verbose=True
    )

class Add(autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        return extension.cuda_add_forward(a, b)

    @staticmethod
    def backward(ctx, grad_out):
        grad_a, grad_b = extension.cuda_add_backward(grad_out)
        return grad_a, grad_b