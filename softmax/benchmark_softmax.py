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
    input = torch.randn(1024, 256, device="cuda", dtype=torch.float32)

    # Benchmark baseline softmax()
    mean, _ = benchmark(softmax_baseline, input, -1, num_steps=10)
    print(mean / 1e-9)

    # Benchmark torch.softmax()
    mean, _ = benchmark(torch.softmax, input, -1, num_steps=10)
    print(mean / 1e-9)

    # Benchmark naive CUDA softmax kernel
    Softmax.use("naive")
    mean, _ = benchmark(Softmax.apply, input, num_steps=10)
    print(mean / 1e-9)

    # Benchmark online CUDA softmax kernel
    Softmax.use("online")
    mean, _ = benchmark(Softmax.apply, input, num_steps=10)
    print(mean / 1e-9)

if __name__ == "__main__":
    main()