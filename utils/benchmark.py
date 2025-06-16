import torch
import time
import torch.cuda.nvtx as nvtx

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def benchmark(func, *args, warmup_steps=20, num_steps=10, nvtx_string=""):
    if warmup_steps > 0:
        for _ in range(warmup_steps):
            func(*args)
            cuda_sync()

    times = []
    for _ in range(num_steps):
        start = time.time()

        with nvtx.range(nvtx_string):
            func(*args)
            cuda_sync()

        times.append(time.time() - start)

    return sum(times) / len(times)