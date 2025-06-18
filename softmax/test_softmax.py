import pytest
import torch
import torch.nn.functional as F
from softmax_cuda import Softmax as SoftmaxCUDA
from softmax_triton import Softmax as SoftmaxTriton


def softmax_forward_test(SoftmaxFunction):
    input = torch.randn(1024, 1024, device="cuda", dtype=torch.float32, requires_grad=True)
    output = SoftmaxFunction.apply(input)
    expected = F.softmax(input, dim=1)
    assert torch.allclose(output, expected, rtol=1e-4, atol=1e-6), "Forward pass results do not match!"


def test_triton_softmax():
    """Test the triton softmax kernel."""
    softmax_forward_test(SoftmaxTriton)


def test_cuda_softmax_sharedmem():
    """Test the shared memory CUDA softmax kernel."""
    SoftmaxCUDA.use("sharedmem")
    softmax_forward_test(SoftmaxCUDA)


def test_cuda_softmax_online():
    """Test the online CUDA softmax kernel."""
    SoftmaxCUDA.use("online")
    softmax_forward_test(SoftmaxCUDA)


def test_cuda_softmax_naive():
    """Test the naive CUDA softmax kernel."""
    SoftmaxCUDA.use("naive")
    softmax_forward_test(SoftmaxCUDA)
