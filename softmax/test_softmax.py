import pytest
import torch
import torch.nn.functional as F
from softmax_cuda import Softmax as SoftmaxCUDA


def softmax_forward_test(SoftmaxFunction):
    input = torch.randn(1024, 128, device="cuda", dtype=torch.float32, requires_grad=True)
    output = SoftmaxFunction.apply(input)
    expected = F.softmax(input, dim=1)
    assert torch.allclose(output, expected, rtol=1e-4, atol=1e-6), "Forward pass results do not match!"


def test_cuda_softmax_forward():
    """Test the forward pass of the CUDA softmax kernel."""
    softmax_forward_test(SoftmaxCUDA)