import math
import pytest
import torch
import torch.nn.functional as F
from attention_triton import FlashAttention


def baseline_attention(q, k, v, mask):
    scores = q @ k.transpose(-2, -1)
    scores /= math.sqrt(k.shape[-1])
    #scores = torch.where(mask, scores, float('-inf'))
    scores = F.softmax(scores, dim=-1)
    return scores @ v

def attention_forward_test(AttentionFunction):
    seq_len = 2048
    d_model = 1024
    d_head = 64
    num_heads = d_model//d_head

    q = torch.randn(seq_len, d_model, device="cuda", dtype=torch.float32)
    k = torch.randn(seq_len, d_model, device="cuda", dtype=torch.float32)
    v = torch.randn(seq_len, d_model, device="cuda", dtype=torch.float32)

    q = q.view(seq_len, num_heads, d_head).transpose(0, 1)
    k = k.view(seq_len, num_heads, d_head).transpose(0, 1)
    v = v.view(seq_len, num_heads, d_head).transpose(0, 1)

    output = AttentionFunction.apply(q, k, v)
    expected = baseline_attention(q, k, v, None)

    assert torch.allclose(output, expected, rtol=1e-3, atol=1e-3), "Forward pass results do not match!"


def test_flash_attention():
    """Test the triton flash attention kernel."""
    attention_forward_test(FlashAttention)