import math
import torch
import triton
import triton.language as tl


@triton.jit
def triton_flash_attention(
            q_ptr, k_ptr, v_ptr, out_ptr,
            q_stride_0, q_stride_1,
            k_stride_0, k_stride_1,
            v_stride_0, v_stride_1,
            out_stride_0, out_stride_1,
            seq_len: tl.constexpr,
            d_model: tl.constexpr,
            block_size: tl.constexpr):
    
    
    pid = tl.program_id(axis=0)
    q_offset = pid*block_size*d_model
    out_offset = q_offset
    out = tl.zeros((block_size, d_model), dtype=tl.float32)
    row_sum = tl.zeros((block_size, 1), dtype=tl.float32)
    row_max = tl.full((block_size, 1), float('-inf'), dtype=tl.float32)

    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + q_offset,
        shape=(seq_len, d_model),
        strides=(q_stride_0, q_stride_1),
        offsets=(0, 0),
        block_shape=(block_size, d_model),
        order=(1, 0)
    )

    out_block_ptr = tl.make_block_ptr(
            base=out_ptr + out_offset,
            shape=(seq_len, d_model),
            strides=(out_stride_0, out_stride_1),
            offsets=(0, 0),
            block_shape=(block_size, d_model),
            order=(1, 0)
    )
    
    for k in range(0, seq_len//block_size):
        k_offset = k * block_size * d_model
        v_offset = k_offset

        k_block_ptr = tl.make_block_ptr(
            base=k_ptr + k_offset,
            shape=(seq_len, d_model),
            strides=(k_stride_0, k_stride_1),
            offsets=(0, 0),
            block_shape=(block_size, d_model),
            order=(1, 0)
        )

        v_block_ptr = tl.make_block_ptr(
            base=v_ptr + v_offset,
            shape=(seq_len, d_model),
            strides=(v_stride_0, v_stride_1),
            offsets=(0, 0),
            block_shape=(block_size, d_model),
            order=(1, 0)
        )

        q_block = tl.load(q_block_ptr, boundary_check=(0, 1))
        k_block = tl.load(k_block_ptr, boundary_check=(0, 1))
        v_block = tl.load(v_block_ptr, boundary_check=(0, 1))
        
        scores = tl.dot(q_block, k_block.T) / (d_model ** 0.5)
        max = tl.maximum(row_max, tl.max(scores, axis=-1, keep_dims=True))
        scores = tl.exp(scores - max)

        row_sum = row_sum * tl.exp(row_max - max) + tl.sum(scores, axis=-1, keep_dims=True)
        out = out * tl.exp(row_max - max) + tl.dot(scores, v_block)
        row_max = max

    out = out / row_sum
    tl.store(out_block_ptr, out)


def launch_triton_flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor):
    seq_len = q.shape[0]
    d_model = q.shape[1]
    block_size = 16
    
    triton_flash_attention[(seq_len//block_size, 1, 1)](
        q, k, v, out,
        q.stride()[0], q.stride()[1],
        k.stride()[0], k.stride()[1],
        v.stride()[0], v.stride()[1],
        out.stride()[0], out.stride()[1],
        seq_len, d_model, block_size)


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        out = torch.zeros_like(q)
        launch_triton_flash_attention(q, k, v, out)
        return out