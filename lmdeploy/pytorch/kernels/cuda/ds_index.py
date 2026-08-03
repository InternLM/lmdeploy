# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl

from .utils import get_device_props


@triton.jit(do_not_specialize=['max_q_seqlen', 'num_split'])
def _fp8_index_kernel(
    q_ptr,
    q_s_ptr,
    k_cache_ptr,
    k_s_cache_ptr,
    cu_seqlen_q_ptr,
    k_seqlen_ptr,
    block_offset_ptr,
    raw_k_seqlen_ptr,
    row_k_seqlen_out_ptr,
    out_ptr,
    stride_qm: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_qsm: tl.constexpr,
    stride_qsh: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_ksb: tl.constexpr,
    stride_ksn: tl.constexpr,
    stride_boff0,
    stride_boff1: tl.constexpr,
    stride_om,
    stride_on: tl.constexpr,
    max_q_seqlen,
    causal: tl.constexpr,
    use_raw_causal_seqlen: tl.constexpr,
    return_row_k_seqlen: tl.constexpr,
    trim_causal_tail: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    num_split,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fp8 index kernel."""
    m_id = tl.program_id(0).to(tl.int64)
    split_id = tl.program_id(1).to(tl.int64)

    assert stride_qd == 1
    assert stride_kd == 1

    batch_id = m_id // max_q_seqlen
    q_id = m_id % max_q_seqlen
    q_start = tl.load(cu_seqlen_q_ptr + batch_id)
    q_seqlen = tl.load(cu_seqlen_q_ptr + batch_id + 1) - q_start
    if q_id >= q_seqlen:
        return

    q_pos = q_start + q_id
    k_seqlen = tl.load(k_seqlen_ptr + batch_id)
    if k_seqlen <= 0:
        if return_row_k_seqlen and split_id == 0:
            tl.store(row_k_seqlen_out_ptr + q_pos, tl.full((), 0, tl.int32))
        return

    causal_k_seqlen = k_seqlen
    if causal:
        if use_raw_causal_seqlen:
            raw_k_seqlen = tl.load(raw_k_seqlen_ptr + batch_id).to(tl.int64)
            raw_q_start = raw_k_seqlen - q_seqlen
            causal_k_seqlen = (raw_q_start + q_id + 1) // COMPRESS_RATIO
        else:
            causal_k_seqlen = k_seqlen - q_seqlen + q_id + 1
        causal_k_seqlen = tl.minimum(tl.maximum(causal_k_seqlen, 0), k_seqlen)

    if return_row_k_seqlen and split_id == 0:
        tl.store(row_k_seqlen_out_ptr + q_pos, causal_k_seqlen.to(tl.int32))

    if trim_causal_tail:
        loop_k_seqlen = causal_k_seqlen
    else:
        loop_k_seqlen = k_seqlen
    if loop_k_seqlen <= 0:
        return

    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)

    q_ptrs = q_ptr + q_pos * stride_qm + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q_s_ptrs = q_s_ptr + q_pos * stride_qsm + offs_h * stride_qsh
    q = tl.load(q_ptrs)
    q_s = tl.load(q_s_ptrs)

    k_ptrs = k_cache_ptr + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
    k_s_ptrs = k_s_cache_ptr + offs_n * stride_ksn
    o_ptrs = out_ptr + q_pos * stride_om + offs_n * stride_on + split_id * BLOCK_N * stride_on
    boff_ptr = block_offset_ptr + batch_id * stride_boff0 + split_id * stride_boff1

    num_blocks = tl.cdiv(loop_k_seqlen, BLOCK_N)
    split_count = num_split.to(tl.int64)
    boff_id = split_id
    while boff_id < num_blocks:
        boff = tl.load(boff_ptr).to(tl.int64)

        k = tl.load(k_ptrs + boff * stride_kb)
        k_s = tl.load(k_s_ptrs + boff * stride_ksb)

        logits = tl.zeros((BLOCK_H, BLOCK_N), dtype=tl.float32)
        logits = tl.dot(q, k, acc=logits)
        logits = tl.maximum(logits, 0) * q_s[:, None]
        logits_sum = tl.sum(logits, axis=0) * k_s

        if causal:
            mask_off = boff_id * BLOCK_N + offs_n
            mask = mask_off < causal_k_seqlen
            logits_sum = tl.where(mask, logits_sum, float('-inf'))

        tl.store(o_ptrs, logits_sum, mask=offs_n + boff_id * BLOCK_N < loop_k_seqlen)
        boff_id += split_count
        boff_ptr += split_count * stride_boff1
        o_ptrs += split_count * BLOCK_N * stride_on


def fp8_index(q: torch.Tensor,
              q_s: torch.Tensor,
              k_cache: torch.Tensor,
              k_s_cache: torch.Tensor,
              cu_seqlen_q: torch.Tensor,
              k_seqlens: torch.Tensor,
              block_offset: torch.Tensor,
              max_q_seqlen: int = None,
              max_k_seqlen: int = None,
              causal: bool = False,
              raw_k_seqlens: torch.Tensor | None = None,
              compress_ratio: int = 1,
              return_row_k_seqlens: bool = False,
              trim_causal_tail: bool = False):
    """Fp8 index.

    q: (cum_seqlen, num_heads, head_dim)
    q_s: (cum_seqlen, num_heads)
    k_cache: (num_blocks, block_size, head_dim)
    k_s_cache: (num_blocks, block_size)
    cu_seqlen_q: (batch_size,)
    cu_seqlen_k: (batch_size,)
    block_offset: (batch_size, num_blocks)
    raw_k_seqlens: optional uncompressed KV lengths for compressed-cache
        causal masking.
    """
    assert q.dim() == 3
    assert k_cache.dim() == 3
    assert q_s.dim() == 2
    assert k_s_cache.dim() == 2
    assert compress_ratio >= 1
    cum_seqlen, num_heads, head_dim = q.shape
    block_size = k_cache.size(1)
    batch_size = k_seqlens.numel()
    use_raw_causal_seqlen = raw_k_seqlens is not None
    if use_raw_causal_seqlen:
        assert raw_k_seqlens.dim() == 1
        assert raw_k_seqlens.numel() == batch_size
    is_decoding = batch_size == cum_seqlen
    if max_k_seqlen is None:
        max_num_blocks = k_cache.size(0)
        max_k_seqlen = max_num_blocks * block_size

    # max q seqlen
    if is_decoding:
        if max_q_seqlen is None:
            max_q_seqlen = 1
        assert max_q_seqlen == 1
    elif max_q_seqlen is None:
        max_q_seqlen = cum_seqlen

    assert q.stride(-1) == 1 and k_cache.stride(-1) == 1

    out = q.new_empty((cum_seqlen, max_k_seqlen), dtype=torch.float32)
    row_k_seqlens = None
    if return_row_k_seqlens:
        row_k_seqlens = torch.empty((cum_seqlen, ), dtype=torch.int32, device=q.device)
    raw_k_seqlens_arg = raw_k_seqlens if use_raw_causal_seqlen else k_seqlens
    row_k_seqlens_arg = row_k_seqlens if row_k_seqlens is not None else k_seqlens

    num_warps = 4
    device_idx = q.device.index
    props = get_device_props(device_idx)
    num_sm = props['multi_processor_count']
    # estimated occupancy 12.5%
    warps_per_sm = props['warps_per_sm'] // 8
    assert warps_per_sm >= num_warps
    cta_per_sm = warps_per_sm // num_warps
    cta_per_device = num_sm * cta_per_sm
    # we better have a tensor to indicate batch id of each q
    M = max_q_seqlen * batch_size
    num_split = max(1, triton.cdiv(cta_per_device, M))
    grid = (M, num_split)

    _fp8_index_kernel[grid](q,
                            q_s,
                            k_cache,
                            k_s_cache,
                            cu_seqlen_q,
                            k_seqlens,
                            block_offset,
                            raw_k_seqlens_arg,
                            row_k_seqlens_arg,
                            out,
                            *q.stride(),
                            *q_s.stride(),
                            *k_cache.stride(),
                            *k_s_cache.stride(),
                            *block_offset.stride(),
                            *out.stride(),
                            max_q_seqlen=max_q_seqlen,
                            causal=causal,
                            use_raw_causal_seqlen=use_raw_causal_seqlen,
                            return_row_k_seqlen=return_row_k_seqlens,
                            trim_causal_tail=trim_causal_tail,
                            COMPRESS_RATIO=compress_ratio,
                            num_split=num_split,
                            BLOCK_H=num_heads,
                            BLOCK_N=block_size,
                            BLOCK_D=head_dim,
                            num_warps=num_warps)
    if return_row_k_seqlens:
        return out, row_k_seqlens
    return out
