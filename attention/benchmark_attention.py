import sys
import torch
from test_attention import baseline_attention
from attention_triton import FlashAttention

sys.path.append('../utils')
from benchmark import *

def main():
    seq_len = 8*1024
    d_model = 1024
    d_head = 64
    num_heads = d_model//d_head

    q = torch.randn(seq_len, d_model, device="cuda", dtype=torch.float32, requires_grad=False)
    k = torch.randn(seq_len, d_model, device="cuda", dtype=torch.float32, requires_grad=False)
    v = torch.randn(seq_len, d_model, device="cuda", dtype=torch.float32, requires_grad=False)

    q = q.view(seq_len, num_heads, d_head).transpose(0, 1)
    k = k.view(seq_len, num_heads, d_head).transpose(0, 1)
    v = v.view(seq_len, num_heads, d_head).transpose(0, 1)

    mean = benchmark(baseline_attention, q, k, v, None, num_steps=10, nvtx_string="baseline_attention")
    print("baseline", mean * 1e6)

    mean = benchmark(torch.nn.functional.scaled_dot_product_attention, q, k, v, None, num_steps=10, nvtx_string="torch_attention")
    print("torch   ", mean * 1e6)

    mean = benchmark(FlashAttention.apply, q, k, v, num_steps=10, nvtx_string="triton_attention")
    print("triton  ", mean * 1e6)

if __name__ == "__main__":
    main()
