import sys
import torch
from torch import autograd
import triton
import triton.language as tl
from triton.runtime import driver

sys.path.append('../utils')
from triton_utils import *

device = torch.device("cuda:0")

@triton.jit
def triton_softmax(input_ptr, output_ptr, input_shape, input_stride, output_stride, block_size:tl.constexpr):
    num_rows, num_cols = input_shape
    start_idx = tl.program_id(0)
    step = tl.num_programs(0)

    for row_index in tl.range(start_idx, num_rows, step):
        input_start = row_index * input_stride[0]
        input_offsets = input_start + tl.arange(0, block_size) * input_stride[1]
        mask = tl.arange(0, block_size) < num_cols

        row_data = tl.load(input_ptr + input_offsets, mask=mask, other=-float('inf'))
        row_exp = tl.exp(row_data - tl.max(row_data))
        row_exp_sum = tl.sum(row_exp)
        output = row_exp / row_exp_sum

        output_start = row_index * output_stride[0]
        output_offsets = output_start + tl.arange(0, block_size) * output_stride[1]
        tl.store(output_ptr + output_offsets, output, mask=mask)


def launch_triton_softmax(input: torch.Tensor):
    assert(input.device) == device
    output = torch.empty_like(input)
    num_rows, num_cols = input.shape
    block_size = triton.next_power_of_2(num_cols)
    num_warps = 8

    cache_key = (triton_softmax, num_warps, input.shape)
    kernel, num_regs, size_smem = get_precompiled_kernel(cache_key, triton_softmax, num_warps,
         input, output, input.shape, input.stride(), output.stride(), block_size)

    num_programs = get_num_programs(device, num_regs, size_smem, num_warps)
    num_programs = min(num_programs, num_rows)
    kernel[(num_programs, 1, 1)](input, output, input.shape, input.stride(), output.stride(), block_size)
    return output


class Softmax(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return launch_triton_softmax(input)