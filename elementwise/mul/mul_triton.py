import torch
import triton
from torch import autograd
import triton.language as tl

DEVICE = torch.device("cuda:0")


@triton.jit
def mul_forward(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * y
    tl.store(out_ptr + offsets, output, mask=mask)


@triton.jit
def mul_backward(grad_out_ptr, x_ptr, y_ptr, grad_x_ptr, grad_y_ptr, n_elements, BLOCK_SIZE:tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    grad_x = grad_out * y
    grad_y = grad_out * x
    tl.store(grad_x_ptr + offsets, grad_x, mask=mask)
    tl.store(grad_y_ptr + offsets, grad_y, mask=mask)


def triton_mul_forward(x: torch.Tensor, y: torch.Tensor):
    out = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    mul_forward[grid](x, y, out, n_elements, BLOCK_SIZE=128)
    return out


def triton_mul_backward(grad_out: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
    grad_out = grad_out.contiguous()
    grad_x = torch.empty_like(x)
    grad_y = torch.empty_like(y)
    assert grad_out.device == DEVICE and x.device == DEVICE and y.device == DEVICE
    n_elements = grad_out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    mul_backward[grid](grad_out, x, y, grad_x, grad_y, n_elements, BLOCK_SIZE=128)
    return grad_x, grad_y


class Mul(autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return triton_mul_forward(x, y)
    
    @staticmethod
    def backward(ctx, grad_out):
        x, y = ctx.saved_tensors
        grad_x, grad_y = triton_mul_backward(grad_out, x, y)
        return grad_x, grad_y