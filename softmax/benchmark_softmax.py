import sys
import torch
from softmax_cuda import Softmax

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
    print("baseline ", mean / 1e-9)

    # Benchmark torch.softmax()
    mean = benchmark(torch.softmax, input, -1, num_steps=10, nvtx_string="torch_softmax")
    print("torch    ", mean / 1e-9)

    # Benchmark naive CUDA softmax kernel
    Softmax.use("naive")
    mean = benchmark(Softmax.apply, input, num_steps=10, nvtx_string="naive_softmax")
    print("naive    ", mean / 1e-9)

    # Benchmark online CUDA softmax kernel
    Softmax.use("online")
    mean = benchmark(Softmax.apply, input, num_steps=10, nvtx_string="online_softmax")
    print("online   ", mean / 1e-9)

    # Benchmark shared mem CUDA softmax kernel
    Softmax.use("sharedmem")
    mean = benchmark(Softmax.apply, input, num_steps=10, nvtx_string="sharedmem_softmax")
    print("sharedmem", mean / 1e-9)

if __name__ == "__main__":
    main()