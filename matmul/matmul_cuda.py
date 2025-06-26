import torch
from torch import autograd
from torch.utils.cpp_extension import load_inline

with open("cuda/matmul.cu", "r") as f:
    cuda_source = f.read()

with open("cuda/matmul_interface.cpp", "r") as f:
    cpp_source = f.read()

ext = load_inline(
    name="cuda_matmul",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["cuda_matmul_forward"],
    extra_cflags=["-std=c++17"],
    extra_cuda_cflags=["-std=c++17"],
    verbose=True
)

class Matmul(autograd.Function):
    _impl = 'naive'

    @staticmethod
    def forward(ctx, A, B):
        return ext.cuda_matmul_forward(A, B, Matmul._impl)

    @staticmethod
    def use(impl):
        Matmul._impl = impl