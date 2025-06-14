import pytest
import torch
from add_cuda import Add as AddCUDA
from add_triton import Add as AddTriton


def add_forward_test(AddFunction):
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.float32, requires_grad=True)
    b = torch.randn(1024, 1024, device="cuda", dtype=torch.float32, requires_grad=True)
    c = AddFunction.apply(a, b)
    expected = a + b
    assert torch.allclose(c, expected, rtol=1e-4, atol=1e-6), "Forward pass results do not match!"


def add_backward_test(AddFunction):
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.float32, requires_grad=True)
    b = torch.randn(1024, 1024, device="cuda", dtype=torch.float32, requires_grad=True)
    c = AddFunction.apply(a, b)
    c.mean().backward()
    grad_a = a.grad.clone()
    grad_b = b.grad.clone()
    a.grad.zero_()
    b.grad.zero_()

    c = a + b
    c.mean().backward()
    expected_grad_a = a.grad.clone()
    expected_grad_b = b.grad.clone()
    assert torch.allclose(grad_a, expected_grad_a, rtol=1e-4, atol=1e-6), "Gradient for a does not match!"
    assert torch.allclose(grad_b, expected_grad_b, rtol=1e-4, atol=1e-6), "Gradient for b does not match!"


def test_cuda_add_forward():
    """Test the forward pass of the CUDA addition."""
    add_forward_test(AddCUDA)


def test_cuda_add_backward():
    """Test the backward pass of the CUDA addition kernel."""
    add_backward_test(AddCUDA)


def test_triton_add_forward():
    """Test the forward pass of the Triton addition kernel."""
    add_forward_test(AddTriton)


def test_triton_add_backward():
    """Test the backward pass of the Triton addition kernel."""
    add_backward_test(AddTriton)