import pytest
import torch
import torch.nn.functional as F
from matmul_cuda import Matmul as MatmulCUDA


def matmul_forward_test(MatmulFunction):
    A = torch.randn(1024, 1024, device="cuda", dtype=torch.float32, requires_grad=True)
    B = torch.randn(1024, 1024, device="cuda", dtype=torch.float32, requires_grad=True)
    output = MatmulFunction.apply(A, B)
    expected = torch.matmul(A, B)
    assert torch.allclose(output, expected, rtol=1e-4, atol=1e-4), "Forward pass results do not match!"


def test_cuda_matmul_naive():
    """Test the naive CUDA matmul kernel."""
    MatmulCUDA.use("naive")
    matmul_forward_test(MatmulCUDA)
