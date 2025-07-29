import torch
import time
import torch.cuda.nvtx as nvtx

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def benchmark(func, *args, warmup_steps=20, num_steps=10, nvtx_string="", profile_mem=False):
    if warmup_steps > 0:
        for _ in range(warmup_steps):
            func(*args)
            cuda_sync()

    times = []
    if profile_mem:
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    for _ in range(num_steps):
        start = time.time()

        with nvtx.range(nvtx_string):
            func(*args)
            cuda_sync()

        times.append(time.time() - start)

    if profile_mem:
        torch.cuda.memory._dump_snapshot(f"{nvtx_string}.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)

    return sum(times) / len(times)