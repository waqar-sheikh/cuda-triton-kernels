import torch
import time

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark(func, *args, warmup_steps=10, num_steps=10):
    if warmup_steps > 0:
        for i in range(warmup_steps):
            func(*args)
            cuda_sync()

    times = []
    for i in range(num_steps):
        start = time.time()
        func(*args)
        cuda_sync()
        times.append(time.time() - start)

    return sum(times) / len(times)
