import sys
import torch
from add_cuda import Add as AddCUDA
from add_triton import Add as AddTriton

sys.path.append('../../utils')
from benchmark import *

def main():
    # Create input tensors
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)
    y = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)

    # Benchmark torch add
    mean = benchmark(lambda a, b: a + b, x, y, num_steps=10, nvtx_string="torch_add")
    print("torch", mean / 1e-6)

    # Benchmark cuda add kernel
    mean = benchmark(AddCUDA.apply, x, y, num_steps=10, nvtx_string="cuda_add")
    print("cuda ", mean / 1e-6)

    # Benchmark triton add kernel
    mean = benchmark(AddTriton.apply, x, y, num_steps=10, nvtx_string="triton_add")
    print("triton ", mean / 1e-6)


if __name__ == "__main__":
    main()