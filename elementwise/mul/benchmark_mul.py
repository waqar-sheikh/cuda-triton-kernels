import sys
import torch
from mul_cuda import Mul as MulCUDA
from mul_triton import Mul as MulTriton

sys.path.append('../../utils')
from benchmark import *

def main():
    # Create input tensors
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
    y = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)

    # Benchmark torch mul
    mean = benchmark(lambda a, b: a*b, x, y, num_steps=20)
    print("torch", mean / 1e-9)

    # Benchmark cuda mul kernel
    mean = benchmark(MulCUDA.apply, x, y, num_steps=20)
    print("cuda ", mean / 1e-9)

    # Benchmark triton mul kernel
    mean = benchmark(MulTriton.apply, x, y, num_steps=20)
    print("triton ", mean / 1e-9)


if __name__ == "__main__":
    main()