import torch
import triton
from torch import autograd
import triton.language as tl

DEVICE = torch.device("cuda:0")

@triton.jit
def add_forward(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE:tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)


@triton.jit
def add_backward(grad_out_ptr, grad_x_ptr, grad_y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask)
    tl.store(grad_x_ptr + offsets, grad_out, mask=mask)
    tl.store(grad_y_ptr + offsets, grad_out, mask=mask)


def triton_add_forward(x: torch.Tensor, y: torch.Tensor):
    assert x.device == DEVICE and y.device == DEVICE
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_forward[grid](x, y, output, n_elements, BLOCK_SIZE=128)
    return output


def triton_add_backward(grad_out: torch.Tensor):
    assert grad_out.device == DEVICE
    grad_x = torch.empty_like(grad_out)
    grad_y = torch.empty_like(grad_out)
    n_elements = grad_out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_backward[grid](grad_out, grad_x, grad_y, n_elements, BLOCK_SIZE=128)
    return grad_x, grad_y


class Add(autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        return triton_add_forward(x, y)
    
    @staticmethod
    def backward(ctx, grad_out):
        grad_x, grad_y = triton_add_backward(grad_out)
        return grad_x, grad_y