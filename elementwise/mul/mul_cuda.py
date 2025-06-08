import torch
from torch import autograd
from torch.utils.cpp_extension import load_inline

with open("cuda/mul.cu", "r") as f:
    cuda_source = f.read()

with open("cuda/mul_interface.cpp", "r") as f:
    cpp_source = f.read()

ext = load_inline(
    name="cuda_mul",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["cuda_mul_forward", "cuda_mul_backward"],
    extra_cflags=["-std=c++17"],
    extra_cuda_cflags=["-std=c++17"],
    verbose=True
)

class Mul(autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return ext.cuda_mul_forward(a, b)

    @staticmethod
    def backward(ctx, grad_out):
        a, b = ctx.saved_tensors
        grad_a, grad_b = ext.cuda_mul_backward(grad_out, a, b)
        return grad_a, grad_b