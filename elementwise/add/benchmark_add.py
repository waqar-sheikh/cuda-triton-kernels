import sys
import torch
from add_cuda import Add as AddCUDA
from add_triton import Add as AddTriton

sys.path.append('../../utils')
from benchmark import *

def main():
    # Create input tensors
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
    y = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)

    # Benchmark torch add
    mean = benchmark(lambda a, b: a + b, x, y, num_steps=10)
    print("torch", mean / 1e-9)

    # Benchmark cuda add kernel
    mean = benchmark(AddCUDA.apply, x, y, num_steps=10)
    print("cuda ", mean / 1e-9)

    # Benchmark triton add kernel
    mean = benchmark(AddTriton.apply, x, y, num_steps=10)
    print("triton ", mean / 1e-9)


if __name__ == "__main__":
    main()