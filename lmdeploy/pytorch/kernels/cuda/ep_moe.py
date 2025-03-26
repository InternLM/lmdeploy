# Copyright (c) OpenMMLab. All rights reserved.
# modify from sglang
from typing import List, Optional

import torch
import triton
import triton.language as tl


@triton.jit
def silu_and_mul_triton_kernel(
    gateup_output,
    down_input,
    hidden_size,
    reorder_topk_ids,
    scales,
    start_expert_id,
    end_expert_id,
    BLOCK_SIZE: tl.constexpr,
):
    InDtype = gateup_output.dtype.element_ty
    OutDtype = down_input.dtype.element_ty

    half_hidden_size = hidden_size // 2

    pid = tl.program_id(0)
    expert_id = tl.load(reorder_topk_ids + pid)
    if expert_id >= start_expert_id and expert_id <= end_expert_id:
        gateup_output_ptr = gateup_output + pid * hidden_size
        gate_output_ptr = gateup_output_ptr
        up_output_ptr = gateup_output_ptr + half_hidden_size
        down_input_ptr = down_input + pid * half_hidden_size

        if scales is not None:
            scale = tl.load(scales + expert_id - start_expert_id)
            scale = (1 / scale).to(InDtype)
        else:
            scale = 1

        for start_offset in tl.range(0, half_hidden_size, BLOCK_SIZE):
            offset = start_offset + tl.arange(0, BLOCK_SIZE)
            mask = offset < half_hidden_size

            gate_output = tl.load(gate_output_ptr + offset, mask=mask).to(tl.float32)
            up_output = tl.load(up_output_ptr + offset, mask=mask)

            # silu & mul & quantize
            gate_output = gate_output * tl.sigmoid(gate_output)
            gate_output = gate_output.to(InDtype)

            silu_mul_output = gate_output * up_output * scale
            silu_mul_output = silu_mul_output.to(OutDtype)
            tl.store(down_input_ptr + offset, silu_mul_output, mask=mask)


@triton.jit
def compute_m_range(
    pid,
    batch_size,
    seg_indptr,
    weight_indices,
    m_num_tiles_indptr,
    BLOCK_SIZE_M: tl.constexpr,
):
    idx = 0
    for bs in range(batch_size):
        tiles = tl.load(m_num_tiles_indptr + bs)
        if pid >= tiles:
            idx = bs

    idx_start = tl.load(m_num_tiles_indptr + idx)

    m_range_start = tl.load(seg_indptr + idx) + (pid - idx_start) * BLOCK_SIZE_M
    m_range_end = min(tl.load(seg_indptr + idx + 1), m_range_start + BLOCK_SIZE_M)
    expert_id = tl.load(weight_indices + idx)
    return m_range_start, m_range_end, expert_id


@triton.jit
def grouped_gemm_triton_kernel(
    a,
    b,
    c,
    batch_size,
    N,
    K,
    seg_indptr,
    weight_indices,
    m_num_tiles_indptr,
    scale_a,
    scale_b,
    use_fp8_w8a8: tl.constexpr,
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    a_stride_0: tl.constexpr,
    b_stride_0: tl.constexpr,
    b_stride_1: tl.constexpr,
    as_stride_0: tl.constexpr,
    as_stride_1: tl.constexpr,
    bs_stride_0: tl.constexpr,
    bs_stride_2: tl.constexpr,
    bs_stride_1: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    c_dtype = c.dtype.element_ty

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    total_m_block = tl.load(m_num_tiles_indptr + batch_size)
    if pid_m >= total_m_block:
        return

    m_range_start, m_range_end, expert_id = compute_m_range(pid_m, batch_size, seg_indptr, weight_indices,
                                                            m_num_tiles_indptr, BLOCK_SIZE_M)
    if m_range_end - m_range_start == 0:
        return

    n_range_start = pid_n * BLOCK_SIZE_N
    n_range_end = min(n_range_start + BLOCK_SIZE_N, N)

    offs_am = tl.arange(0, BLOCK_SIZE_M)
    offs_bn = tl.arange(0, BLOCK_SIZE_N)

    offs_am = tl.where(offs_am < m_range_end - m_range_start, offs_am, 0)
    offs_bn = tl.where(offs_bn < n_range_end - n_range_start, offs_bn, 0)
    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptr = a + (m_range_start + offs_am[:, None]) * a_stride_0 + offs_k[None, :]
    b_ptr = b + ((expert_id * b_stride_0) + (n_range_start + offs_bn[:, None]) * b_stride_1 + offs_k[None, :])

    if group_k > 0 and group_n > 0:
        a_scale_ptrs = scale_a + (m_range_start + offs_am[:, None]) * as_stride_0
        offs_bsn = (n_range_start + offs_bn) // group_n
        b_scale_ptrs = scale_b + (expert_id * bs_stride_0) + offs_bsn * bs_stride_1

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_tile = tl.load(a_ptr, mask=offs_k[None, :] < (K - k * BLOCK_SIZE_K), other=0.0)
        b_tile = tl.load(b_ptr, mask=offs_k[None, :] < (K - k * BLOCK_SIZE_K), other=0.0)

        if group_k > 0 and group_n > 0:
            k_start = k * BLOCK_SIZE_K
            offs_ks = k_start // group_k
            a_scale = tl.load(a_scale_ptrs + offs_ks * as_stride_1)
            b_scale = tl.load(b_scale_ptrs + offs_ks * bs_stride_2)
            accumulator += tl.dot(a_tile, b_tile.T) * a_scale * b_scale[None, :]
        else:
            accumulator = tl.dot(a_tile, b_tile.T, accumulator)
        a_ptr += BLOCK_SIZE_K
        b_ptr += BLOCK_SIZE_K

    if use_fp8_w8a8 and not (group_k > 0 and group_n > 0):
        scale_a_value = tl.load(scale_a + expert_id)
        scale_b_value = tl.load(scale_b + expert_id)
        accumulator *= scale_a_value * scale_b_value

    c_tile = accumulator.to(c_dtype)

    offs_cm = m_range_start + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_range_start + tl.arange(0, BLOCK_SIZE_N)
    c_ptr = c + offs_cm[:, None] * N + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < m_range_end) & (offs_cn[None, :] < n_range_end)
    tl.store(c_ptr, c_tile, mask=c_mask)


@triton.jit
def compute_m_num_tiles_indptr(m_num_tiles_indptr, seg_indptr, batch_size: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):
    for bs in range(batch_size):
        m = tl.load(seg_indptr + bs + 1) - tl.load(seg_indptr + bs)
        cur_num_tiles = tl.cdiv(m, BLOCK_SIZE_M)
        pre_num_tiles = tl.load(m_num_tiles_indptr + bs)
        tl.store(m_num_tiles_indptr + bs + 1, pre_num_tiles + cur_num_tiles)


def grouped_gemm_triton(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    batch_size: int,
    weight_column_major: bool,
    seg_indptr: Optional[torch.Tensor] = None,
    weight_indices: Optional[torch.Tensor] = None,
    use_fp8_w8a8: bool = False,
    scale_a: torch.Tensor = None,
    scale_b: torch.Tensor = None,
    block_shape: Optional[List[int]] = None,
):
    assert weight_column_major
    if use_fp8_w8a8 and block_shape is None:
        assert scale_a is not None and scale_b is not None

    if block_shape is not None:
        assert len(block_shape) == 2
        block_n, block_k = block_shape[0], block_shape[1]
        assert triton.cdiv(a.shape[-1], block_k) == scale_a.shape[-1]
        assert triton.cdiv(b.shape[-2], block_n) == scale_b.shape[-2]
        assert triton.cdiv(b.shape[-1], block_k) == scale_b.shape[-1]

    # TODO: adjust config or tune kernel
    # Reduce block size to prevent L40 shared memory overflow.
    config = {
        'BLOCK_SIZE_M': 64,
        'BLOCK_SIZE_N': 32,
        'BLOCK_SIZE_K': 128,
    }

    m_num_tiles_indptr = torch.zeros(batch_size + 1, device=a.device, dtype=torch.int64)
    compute_m_num_tiles_indptr[(1, )](m_num_tiles_indptr, seg_indptr, batch_size, config['BLOCK_SIZE_M'])

    def grid(META):
        return (
            triton.cdiv(a.size(0), META['BLOCK_SIZE_M']) + batch_size,
            triton.cdiv(b.size(1), META['BLOCK_SIZE_N']),
        )

    grouped_gemm_triton_kernel[grid](
        a,
        b,
        c,
        batch_size,
        b.size(1),
        b.size(2),
        seg_indptr,
        weight_indices,
        m_num_tiles_indptr,
        scale_a,
        scale_b,
        use_fp8_w8a8,
        0 if block_shape is None else block_shape[0],
        0 if block_shape is None else block_shape[1],
        a.stride(0),
        b.stride(0),
        b.stride(1),
        scale_a.stride(0) if scale_a is not None and scale_a.ndim == 2 else 0,
        scale_a.stride(1) if scale_a is not None and scale_a.ndim == 2 else 0,
        scale_b.stride(0) if scale_b is not None and scale_b.ndim >= 2 else 0,
        scale_b.stride(2) if scale_b is not None and scale_b.ndim == 3 else 0,
        scale_b.stride(1) if scale_b is not None and scale_b.ndim >= 2 else 0,
        **config,
    )
    return c
