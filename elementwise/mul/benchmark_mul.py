import sys
import torch
from mul_cuda import Mul as MulCUDA
from mul_triton import Mul as MulTriton

sys.path.append('../../utils')
from benchmark import *

def main():
    # Create input tensors
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)
    y = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)

    # Benchmark torch mul
    mean = benchmark(lambda a, b: a*b, x, y, num_steps=10, nvtx_string="torch_mul")
    print("torch", mean / 1e-6)

    # Benchmark cuda mul kernel
    mean = benchmark(MulCUDA.apply, x, y, num_steps=10, nvtx_string="cuda_mul")
    print("cuda ", mean / 1e-6)

    # Benchmark triton mul kernel
    mean = benchmark(MulTriton.apply, x, y, num_steps=10, nvtx_string="triton_mul")
    print("triton ", mean / 1e-6)


if __name__ == "__main__":
    main()