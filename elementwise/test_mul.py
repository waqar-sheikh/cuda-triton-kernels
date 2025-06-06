import pytest
import torch
from mul_cuda import Mul as MulCUDA


def test_cuda_mul_forward():
    """Test the forward pass of the CUDA multiplication."""
    a = torch.randn(1024, device="cuda", dtype=torch.float32, requires_grad=True)
    b = torch.randn(1024, device="cuda", dtype=torch.float32, requires_grad=True)
    c = MulCUDA.apply(a, b)
    expected = a * b
    assert torch.allclose(c, expected, rtol=1e-4, atol=1e-6), "Forward pass results do not match!"


def test_cuda_mul_backward():
    """Test the backward pass of the CUDA multiplication."""
    a = torch.randn(1024, device="cuda", dtype=torch.float32, requires_grad=True)
    b = torch.randn(1024, device="cuda", dtype=torch.float32, requires_grad=True)
    c = MulCUDA.apply(a, b)
    c.sum().backward()
    grad_a = a.grad.clone()
    grad_b = b.grad.clone()
    a.grad.zero_()
    b.grad.zero_()

    c = a * b
    c.sum().backward()
    expected_grad_a = a.grad.clone()
    expected_grad_b = b.grad.clone()
    assert torch.allclose(grad_a, expected_grad_a, rtol=1e-4, atol=1e-6), "Gradient for a does not match!"
    assert torch.allclose(grad_b, expected_grad_b, rtol=1e-4, atol=1e-6), "Gradient for b does not match!"