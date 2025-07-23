import math
import pytest
import torch
import torch.nn.functional as F
from attention_triton import FlashAttention
from attention_triton2 import FlashAttention2


def baseline_attention(q, k, v, mask):
    scores = q @ k.transpose(-2, -1)
    scores /= math.sqrt(k.shape[-1])
    #scores = torch.where(mask, scores, float('-inf'))
    scores = F.softmax(scores, dim=-1)
    return scores @ v

#def baseline_attention(q, k, v, mask):
#    scores = q @ k.transpose(-2, -1)
#    out = scores @ v
#    return out


def attention_forward_test(AttentionFunction):
    q = torch.randn(32, 32, device="cuda", dtype=torch.float32)
    k = torch.randn(32, 32, device="cuda", dtype=torch.float32)
    v = torch.randn(32, 32, device="cuda", dtype=torch.float32)
    
    output = AttentionFunction.apply(q, k, v)
    expected = baseline_attention(q, k, v, None)

    assert torch.allclose(output, expected, rtol=1e-1, atol=1e-1), "Forward pass results do not match!"


def test_flash_attention():
    """Test the triton flash attention kernel."""
    attention_forward_test(FlashAttention)