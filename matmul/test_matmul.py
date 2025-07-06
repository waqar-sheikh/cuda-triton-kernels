import pytest
import torch
import torch.nn.functional as F
from matmul_cuda import Matmul as MatmulCUDA


def matmul_forward_test(MatmulFunction):
    A = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
    B = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)

    output = MatmulFunction.apply(A, B)
    expected = torch.matmul(A, B)
    assert torch.allclose(output, expected, rtol=1e-4, atol=1e-4), "Forward pass results do not match!"


def test_cuda_matmul_naive():
    """Test the naive CUDA matmul kernel."""
    MatmulCUDA.use("naive")
    matmul_forward_test(MatmulCUDA)


def test_cuda_matmul_blocktiling():
    """Test the blocktiling based CUDA matmul kernel."""
    MatmulCUDA.use("blocktiling")
    matmul_forward_test(MatmulCUDA)


def test_cuda_matmul_threadtiling():
    """Test the threadtiling based CUDA matmul kernel."""
    MatmulCUDA.use("threadtiling")
    matmul_forward_test(MatmulCUDA)
