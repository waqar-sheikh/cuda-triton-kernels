import sys
import torch
from matmul_cuda import Matmul as MatmulCUDA

sys.path.append('../utils')
from benchmark import *


def main():
    # Create input tensors
    A = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
    B = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)

    # Benchmark torch.matmul()
    mean = benchmark(torch.matmul, A, B, num_steps=10, nvtx_string="torch_matmul")
    print("torch    ", mean / 1e-6)

    # Benchmark naive CUDA matmul kernel
    MatmulCUDA.use("naive")
    mean = benchmark(MatmulCUDA.apply, A, B, num_steps=10, nvtx_string="naive_matmul")
    print("naive    ", mean / 1e-6)

    # Benchmark sharedmem CUDA matmul kernel
    MatmulCUDA.use("sharedmem")
    mean = benchmark(MatmulCUDA.apply, A, B, num_steps=10, nvtx_string="sharedmem_matmul")
    print("sharedmem", mean / 1e-6)

if __name__ == "__main__":
    main()