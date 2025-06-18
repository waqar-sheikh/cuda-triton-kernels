import sys
import torch
from softmax_cuda import Softmax as SoftmaxCUDA
from softmax_triton import Softmax as SoftmaxTriton

sys.path.append('../utils')
from benchmark import *

def softmax_baseline(x, dim):
    maximums = torch.max(x, dim=dim, keepdim=True).values
    scores = torch.exp(x - maximums)
    sums = torch.sum(scores, dim=dim, keepdim=True)
    scores = scores / sums
    return scores

def main():
    # Create input tensor
    input = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)

    # Benchmark baseline softmax()
    mean = benchmark(softmax_baseline, input, -1, num_steps=10, nvtx_string="baseline_softmax")
    print("baseline ", mean / 1e-6)

    # Benchmark torch.softmax()
    mean = benchmark(torch.softmax, input, -1, num_steps=10, nvtx_string="torch_softmax")
    print("torch    ", mean / 1e-6)

    # Benchmark naive CUDA softmax kernel
    SoftmaxCUDA.use("naive")
    mean = benchmark(SoftmaxCUDA.apply, input, num_steps=10, nvtx_string="naive_softmax")
    print("naive    ", mean / 1e-6)

    # Benchmark online CUDA softmax kernel
    SoftmaxCUDA.use("online")
    mean = benchmark(SoftmaxCUDA.apply, input, num_steps=10, nvtx_string="online_softmax")
    print("online   ", mean / 1e-6)

    # Benchmark shared mem CUDA softmax kernel
    SoftmaxCUDA.use("sharedmem")
    mean = benchmark(SoftmaxCUDA.apply, input, num_steps=10, nvtx_string="sharedmem_softmax")
    print("sharedmem", mean / 1e-6)

    # Benchmark triton softmax kernel
    mean = benchmark(SoftmaxTriton.apply, input, num_steps=10, nvtx_string="triton_softmax")
    print("triton   ", mean / 1e-6)

if __name__ == "__main__":
    main()