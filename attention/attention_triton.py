import sys
import torch
import triton
import triton.language as tl

sys.path.append('../utils')
from triton_utils import *

device = torch.device("cuda:0")

@triton.jit
def triton_flash_attention(
            q_ptr, k_ptr, v_ptr, out_ptr,
            q_stride_head, q_stride_seq, q_stride_dim,
            k_stride_head, k_stride_seq, k_stride_dim,
            v_stride_head, v_stride_seq, v_stride_dim,
            out_stride_head, out_stride_seq, out_stride_dim,
            seq_len: tl.constexpr,
            d_head: tl.constexpr,
            num_heads: tl.constexpr,
            BS_SEQ: tl.constexpr):

    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(0)

    for head_idx in range(0, num_heads):

        for seq_idx in tl.range(pid, seq_len//BS_SEQ, num_programs):

            out = tl.zeros((BS_SEQ, d_head), dtype=tl.float32)
            row_sum = tl.zeros((BS_SEQ, 1), dtype=tl.float32)
            row_max = tl.full((BS_SEQ, 1), float('-inf'), dtype=tl.float32)

            q_offset = head_idx * q_stride_head + seq_idx * BS_SEQ * q_stride_seq
            out_offset = head_idx * out_stride_head + seq_idx * BS_SEQ * out_stride_seq
            kv_offset = head_idx * k_stride_head

            q_block_ptr = tl.make_block_ptr(
                base=q_ptr + q_offset,
                shape=(seq_len, d_head),
                strides=(q_stride_seq, q_stride_dim),
                offsets=(0, 0),
                block_shape=(BS_SEQ, d_head),
                order=(1, 0)
            )

            k_block_ptr = tl.make_block_ptr(
                base=k_ptr + kv_offset,
                shape=(seq_len, d_head),
                strides=(k_stride_seq, k_stride_dim),
                offsets=(0, 0),
                block_shape=(BS_SEQ, d_head),
                order=(1, 0)
            )

            v_block_ptr = tl.make_block_ptr(
                base=v_ptr + kv_offset,
                shape=(seq_len, d_head),
                strides=(v_stride_seq, v_stride_dim),
                offsets=(0, 0),
                block_shape=(BS_SEQ, d_head),
                order=(1, 0)
            )
    
            out_block_ptr = tl.make_block_ptr(
                base=out_ptr + out_offset,
                shape=(seq_len, d_head),
                strides=(out_stride_seq, out_stride_dim),
                offsets=(0, 0),
                block_shape=(BS_SEQ, d_head),
                order=(1, 0)
            )

            q_block = tl.load(q_block_ptr, boundary_check=(0, 1))

            for kv_idx in range(0, seq_len//BS_SEQ):

                k_block = tl.load(k_block_ptr, boundary_check=(0, 1))
                scores = tl.dot(q_block, k_block.T) / (d_head ** 0.5)
        
                max = tl.maximum(row_max, tl.max(scores, axis=-1, keep_dims=True))
                scores = tl.exp(scores - max)

                row_sum = row_sum * tl.exp(row_max - max) + tl.sum(scores, axis=-1, keep_dims=True)
                v_block = tl.load(v_block_ptr, boundary_check=(0, 1))
                out = out * tl.exp(row_max - max) + tl.dot(scores, v_block)
                row_max = max

                k_block_ptr = k_block_ptr.advance((BS_SEQ, 0))
                v_block_ptr = v_block_ptr.advance((BS_SEQ, 0))

            out = out / row_sum
            tl.store(out_block_ptr, out)


def launch_triton_flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor):
    num_heads = q.shape[0]
    seq_len = q.shape[1]
    d_head = q.shape[2]
    BS_SEQ = 32
    num_warps = 4

    cache_key = (triton_flash_attention, num_warps, BS_SEQ, q.shape)
    kernel, num_regs, size_smem = get_precompiled_kernel(cache_key, triton_flash_attention, num_warps,
        q, k, v, out,
        q.stride()[0], q.stride()[1], q.stride()[2],
        k.stride()[0], k.stride()[1], k.stride()[2],
        v.stride()[0], v.stride()[1], v.stride()[2],
        out.stride()[0], out.stride()[1], out.stride()[2],
        seq_len, d_head, num_heads, BS_SEQ)

    num_programs = get_num_programs(device, num_regs, size_smem, num_warps)
    num_programs = min(num_programs, seq_len//BS_SEQ)
    
    kernel[(num_programs, 1, 1)](
        q, k, v, out,
        q.stride()[0], q.stride()[1], q.stride()[2],
        k.stride()[0], k.stride()[1], k.stride()[2],
        v.stride()[0], v.stride()[1], v.stride()[2],
        out.stride()[0], out.stride()[1], out.stride()[2],
        seq_len, d_head, num_heads, BS_SEQ)


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        out = torch.zeros_like(q)
        launch_triton_flash_attention(q, k, v, out)
        return out