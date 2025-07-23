import sys
import torch
from attention_triton import FlashAttention
from attention_triton2 import FlashAttention2
from test_attention import baseline_attention

sys.path.append('../utils')
from benchmark import *

def main():

    q = torch.randn(256, 256, device="cuda", dtype=torch.float32)
    k = torch.randn(256, 256, device="cuda", dtype=torch.float32)
    v = torch.randn(256, 256, device="cuda", dtype=torch.float32)

    mean = benchmark(baseline_attention, q, k, v, None, num_steps=10, nvtx_string="baseline_attention")
    print("baseline", mean / 1e-6)

    mean = benchmark(FlashAttention.apply, q, k, v, num_steps=10, nvtx_string="triton_attention")
    print("triton  ", mean / 1e-6)

if __name__ == "__main__":
    main()
