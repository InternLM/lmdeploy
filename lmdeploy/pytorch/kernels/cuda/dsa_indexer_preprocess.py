# Copyright (c) OpenMMLab. All rights reserved.
"""Fused Q/K preprocessing and cache writes for the DSA FP8 indexer."""

import torch
import triton
import triton.language as tl
from torch import Tensor

from .blocked_gemm_fp8 import fast_round_scale

# The unfused indexer applies the same normalized Hadamard transform to Q and
# K. Their dot product is unchanged because H @ H.T is identity, so this path
# omits both transforms and prepares the tensors directly for FP8 indexing.
# FP8 rounding makes the two paths numerically close rather than bit-identical.


@triton.jit
def _apply_rope_first(
    x,
    x_pair,
    cos,
    sin,
    feat_off,
    rope_dim: tl.constexpr,
    rope_interleaved: tl.constexpr,
):
    """Apply RoPE to the leading dimensions for both supported pair layouts."""
    prod_cos = x * cos + 0
    prod_sin = x_pair * sin + 0
    if rope_interleaved:
        is_lower = (feat_off & 1) == 0
    else:
        is_lower = feat_off < rope_dim // 2
    rotated = tl.where(is_lower, prod_cos - prod_sin, prod_cos + prod_sin)
    return tl.where(feat_off < rope_dim, rotated, x)


@triton.jit
def _prepare_dsa_indexer_q_kernel(
    Q,
    Weights,
    Cos,
    Sin,
    QOut,
    QScaleOut,
    num_heads: tl.constexpr,
    score_scale: tl.constexpr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    stride_qt: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_wt: tl.constexpr,
    stride_wh: tl.constexpr,
    stride_cs: tl.constexpr,
    stride_cd: tl.constexpr,
    stride_ot: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_od: tl.constexpr,
    stride_st: tl.constexpr,
    stride_sh: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    rope_interleaved: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row_id = tl.program_id(0)
    token_id = row_id // num_heads
    head_id = row_id % num_heads

    feat_off = tl.arange(0, BLOCK_D)
    feat_mask = feat_off < head_dim
    q_ptr = Q + token_id * stride_qt + head_id * stride_qh
    x = tl.load(q_ptr + feat_off * stride_qd, mask=feat_mask, other=0.0)

    # GLM pairs adjacent features (0, 1), (2, 3), ... and shares one frequency
    # per pair. DeepSeek uses NeoX pairs (0, D/2), (1, D/2 + 1), ... and its
    # cos/sin table already contains the corresponding low/high frequencies.
    if rope_interleaved:
        pair_off = feat_off ^ 1
        freq_off = feat_off // 2
    else:
        half_rope_dim = rope_dim // 2
        pair_off = tl.where(feat_off < half_rope_dim, feat_off + half_rope_dim, feat_off - half_rope_dim)
        freq_off = feat_off
    x_pair = tl.load(q_ptr + pair_off * stride_qd, mask=feat_mask, other=0.0)
    freq_mask = feat_off < rope_dim
    cos = tl.load(Cos + token_id * stride_cs + freq_off * stride_cd, mask=freq_mask, other=1.0)
    sin = tl.load(Sin + token_id * stride_cs + freq_off * stride_cd, mask=freq_mask, other=0.0)
    x = _apply_rope_first(x, x_pair, cos, sin, feat_off, rope_dim, rope_interleaved)

    # Match the unfused path: RoPE materializes BF16 before dynamic FP8 quantization.
    x = x.to(tl.bfloat16)
    abs_max = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-6).to(tl.float32)
    scale = fast_round_scale(abs_max, 1.0 / fp8_max)
    out = tl.clamp(x.to(tl.float32) * (1.0 / scale), fp8_min, fp8_max).to(QOut.dtype.element_ty)

    out_ptr = QOut + token_id * stride_ot + head_id * stride_oh
    tl.store(out_ptr + feat_off * stride_od, out, mask=feat_mask)
    weight = tl.load(Weights + token_id * stride_wt + head_id * stride_wh).to(tl.float32)
    # Fold the head gate and attention scale into Q's FP8 scale.
    tl.store(QScaleOut + token_id * stride_st + head_id * stride_sh, scale * weight * score_scale)


@triton.jit
def _prepare_dsa_indexer_k_kernel(
    K,
    NormWeight,
    NormBias,
    Cos,
    Sin,
    KOut,
    eps: tl.constexpr,
    stride_kt: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_cs: tl.constexpr,
    stride_cd: tl.constexpr,
    stride_ot: tl.constexpr,
    stride_od: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    rope_interleaved: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_id = tl.program_id(0)
    feat_off = tl.arange(0, BLOCK_D)
    feat_mask = feat_off < head_dim
    k_ptr = K + token_id * stride_kt

    x = tl.load(k_ptr + feat_off * stride_kd, mask=feat_mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / head_dim
    centered = x - mean
    inv_std = tl.rsqrt(tl.sum(centered * centered, axis=0) / head_dim + eps)
    weight = tl.load(NormWeight + feat_off, mask=feat_mask, other=0.0).to(tl.float32)
    bias = tl.load(NormBias + feat_off, mask=feat_mask, other=0.0).to(tl.float32)
    x = (centered * inv_std * weight + bias).to(tl.bfloat16)

    # Recreate the normalized RoPE partner from raw K with the same row
    # statistics instead of materializing the complete LayerNorm output.
    if rope_interleaved:
        pair_off = feat_off ^ 1
        freq_off = feat_off // 2
    else:
        half_rope_dim = rope_dim // 2
        pair_off = tl.where(feat_off < half_rope_dim, feat_off + half_rope_dim, feat_off - half_rope_dim)
        freq_off = feat_off
    x_pair_raw = tl.load(k_ptr + pair_off * stride_kd, mask=feat_mask, other=0.0).to(tl.float32)
    pair_weight = tl.load(NormWeight + pair_off, mask=feat_mask, other=0.0).to(tl.float32)
    pair_bias = tl.load(NormBias + pair_off, mask=feat_mask, other=0.0).to(tl.float32)
    x_pair = ((x_pair_raw - mean) * inv_std * pair_weight + pair_bias).to(tl.bfloat16)

    freq_mask = feat_off < rope_dim
    cos = tl.load(Cos + token_id * stride_cs + freq_off * stride_cd, mask=freq_mask, other=1.0)
    sin = tl.load(Sin + token_id * stride_cs + freq_off * stride_cd, mask=freq_mask, other=0.0)
    x = _apply_rope_first(x, x_pair, cos, sin, feat_off, rope_dim, rope_interleaved)
    tl.store(KOut + token_id * stride_ot + feat_off * stride_od, x.to(tl.bfloat16), mask=feat_mask)


@triton.jit
def _prepare_dsa_indexer_k_cache_kernel(
    K,
    NormWeight,
    NormBias,
    Cos,
    Sin,
    KCache,
    KSCache,
    CuSeqLenQ,
    KVSeqLens,
    BlockOffsets,
    eps: tl.constexpr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    stride_kt: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_cs: tl.constexpr,
    stride_cd: tl.constexpr,
    stride_kcb: tl.constexpr,
    stride_kcs: tl.constexpr,
    stride_kcd: tl.constexpr,
    stride_ksb: tl.constexpr,
    stride_kss: tl.constexpr,
    stride_boff: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    rope_interleaved: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    q_id = tl.program_id(0)
    batch_id = tl.program_id(1)
    q_start = tl.load(CuSeqLenQ + batch_id)
    q_seqlen = tl.load(CuSeqLenQ + batch_id + 1) - q_start
    if q_id >= q_seqlen:
        return

    kv_seqlen = tl.load(KVSeqLens + batch_id)
    history_seqlen = kv_seqlen - q_seqlen
    # The input contains only this step's tokens; append them after the cached
    # history and translate the logical position through the page table.
    kv_pos = history_seqlen + q_id
    logical_block = kv_pos // block_size
    page_off = kv_pos % block_size
    physical_block = tl.load(BlockOffsets + batch_id * stride_boff + logical_block).to(tl.int64)
    token_id = q_start + q_id

    feat_off = tl.arange(0, BLOCK_D)
    feat_mask = feat_off < head_dim
    k_ptr = K + token_id * stride_kt
    x = tl.load(k_ptr + feat_off * stride_kd, mask=feat_mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / head_dim
    centered = x - mean
    inv_std = tl.rsqrt(tl.sum(centered * centered, axis=0) / head_dim + eps)
    weight = tl.load(NormWeight + feat_off, mask=feat_mask, other=0.0).to(tl.float32)
    bias = tl.load(NormBias + feat_off, mask=feat_mask, other=0.0).to(tl.float32)
    x = (centered * inv_std * weight + bias).to(tl.bfloat16)

    # Recreate the normalized partner needed by RoPE without a temporary K
    # tensor, then quantize directly into its paged-cache destination.
    if rope_interleaved:
        pair_off = feat_off ^ 1
        freq_off = feat_off // 2
    else:
        half_rope_dim = rope_dim // 2
        pair_off = tl.where(feat_off < half_rope_dim, feat_off + half_rope_dim, feat_off - half_rope_dim)
        freq_off = feat_off
    x_pair_raw = tl.load(k_ptr + pair_off * stride_kd, mask=feat_mask, other=0.0).to(tl.float32)
    pair_weight = tl.load(NormWeight + pair_off, mask=feat_mask, other=0.0).to(tl.float32)
    pair_bias = tl.load(NormBias + pair_off, mask=feat_mask, other=0.0).to(tl.float32)
    x_pair = ((x_pair_raw - mean) * inv_std * pair_weight + pair_bias).to(tl.bfloat16)

    freq_mask = feat_off < rope_dim
    cos = tl.load(Cos + token_id * stride_cs + freq_off * stride_cd, mask=freq_mask, other=1.0)
    sin = tl.load(Sin + token_id * stride_cs + freq_off * stride_cd, mask=freq_mask, other=0.0)
    x = _apply_rope_first(x, x_pair, cos, sin, feat_off, rope_dim, rope_interleaved)
    x = x.to(tl.bfloat16)

    abs_max = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-6).to(tl.float32)
    # Keep the ue8m0 scale rounding used by the FP8 index kernel.
    scale = fast_round_scale(abs_max, 1.0 / fp8_max)
    out = tl.clamp(x.to(tl.float32) * (1.0 / scale), fp8_min, fp8_max).to(KCache.dtype.element_ty)
    cache_ptr = KCache + physical_block * stride_kcb + page_off * stride_kcs
    tl.store(cache_ptr + feat_off * stride_kcd, out, mask=feat_mask)
    tl.store(KSCache + physical_block * stride_ksb + page_off * stride_kss, scale)


def prepare_dsa_indexer_q(
    q: Tensor,
    weights: Tensor,
    cos: Tensor,
    sin: Tensor,
    score_scale: float,
    out_dtype: torch.dtype,
    rope_interleaved: bool,
) -> tuple[Tensor, Tensor]:
    """Fuse RoPE, FP8 quantization, and head-gate scaling."""
    assert q.dtype == torch.bfloat16 and q.dim() == 3
    assert q.size(-1) == 128 and cos.size(-1) == 64 and sin.shape == cos.shape
    assert q.shape[:-1] == weights.shape and q.size(0) == cos.size(0)
    assert q.stride(-1) == 1 and weights.stride(-1) == 1

    q_out = torch.empty_like(q, dtype=out_dtype)
    q_scale = q.new_empty(q.shape[:-1], dtype=torch.float32)
    finfo = torch.finfo(out_dtype)
    grid = (q.size(0) * q.size(1), )
    _prepare_dsa_indexer_q_kernel[grid](q,
                                        weights,
                                        cos,
                                        sin,
                                        q_out,
                                        q_scale,
                                        num_heads=q.size(1),
                                        score_scale=score_scale,
                                        fp8_min=finfo.min,
                                        fp8_max=finfo.max,
                                        stride_qt=q.stride(0),
                                        stride_qh=q.stride(1),
                                        stride_qd=q.stride(2),
                                        stride_wt=weights.stride(0),
                                        stride_wh=weights.stride(1),
                                        stride_cs=cos.stride(0),
                                        stride_cd=cos.stride(1),
                                        stride_ot=q_out.stride(0),
                                        stride_oh=q_out.stride(1),
                                        stride_od=q_out.stride(2),
                                        stride_st=q_scale.stride(0),
                                        stride_sh=q_scale.stride(1),
                                        head_dim=128,
                                        rope_dim=64,
                                        rope_interleaved=rope_interleaved,
                                        BLOCK_D=128,
                                        num_warps=4,
                                        num_stages=1)
    return q_out, q_scale


def prepare_dsa_indexer_k(
    k: Tensor,
    norm_weight: Tensor,
    norm_bias: Tensor,
    cos: Tensor,
    sin: Tensor,
    eps: float,
    rope_interleaved: bool,
) -> Tensor:
    """Reference fused LayerNorm and RoPE K preparation."""
    assert k.dtype == torch.bfloat16 and k.dim() == 2 and k.size(-1) == 128
    assert norm_weight.shape == norm_bias.shape == (128, )
    assert cos.shape == sin.shape == (k.size(0), 64)
    k_out = torch.empty_like(k)
    _prepare_dsa_indexer_k_kernel[(k.size(0), )](k,
                                                 norm_weight,
                                                 norm_bias,
                                                 cos,
                                                 sin,
                                                 k_out,
                                                 eps=eps,
                                                 stride_kt=k.stride(0),
                                                 stride_kd=k.stride(1),
                                                 stride_cs=cos.stride(0),
                                                 stride_cd=cos.stride(1),
                                                 stride_ot=k_out.stride(0),
                                                 stride_od=k_out.stride(1),
                                                 head_dim=128,
                                                 rope_dim=64,
                                                 rope_interleaved=rope_interleaved,
                                                 BLOCK_D=128,
                                                 num_warps=4,
                                                 num_stages=1)
    return k_out


def prepare_dsa_indexer_k_cache(
    k: Tensor,
    norm_weight: Tensor,
    norm_bias: Tensor,
    cos: Tensor,
    sin: Tensor,
    k_cache: Tensor,
    k_s_cache: Tensor,
    cu_seqlen_q: Tensor,
    kv_seqlens: Tensor,
    block_offsets: Tensor,
    max_q_seqlen: int,
    eps: float,
    rope_interleaved: bool,
) -> None:
    """Fuse K LayerNorm, RoPE, FP8 quantization, and cache fill."""
    assert k.dtype == torch.bfloat16 and k.dim() == 2 and k.size(-1) == 128
    assert k_cache.dim() == 3 and k_cache.size(-1) == 128
    assert k_s_cache.dim() == 2 and k_s_cache.shape == k_cache.shape[:2]
    assert norm_weight.shape == norm_bias.shape == (128, )
    assert cos.shape == sin.shape == (k.size(0), 64)
    assert cu_seqlen_q.numel() == kv_seqlens.numel() + 1

    block_offsets = block_offsets.contiguous()
    finfo = torch.finfo(k_cache.dtype)
    grid = (max_q_seqlen, kv_seqlens.numel())
    _prepare_dsa_indexer_k_cache_kernel[grid](k,
                                              norm_weight,
                                              norm_bias,
                                              cos,
                                              sin,
                                              k_cache,
                                              k_s_cache,
                                              cu_seqlen_q,
                                              kv_seqlens,
                                              block_offsets,
                                              eps=eps,
                                              fp8_min=finfo.min,
                                              fp8_max=finfo.max,
                                              stride_kt=k.stride(0),
                                              stride_kd=k.stride(1),
                                              stride_cs=cos.stride(0),
                                              stride_cd=cos.stride(1),
                                              stride_kcb=k_cache.stride(0),
                                              stride_kcs=k_cache.stride(1),
                                              stride_kcd=k_cache.stride(2),
                                              stride_ksb=k_s_cache.stride(0),
                                              stride_kss=k_s_cache.stride(1),
                                              stride_boff=block_offsets.stride(0),
                                              block_size=k_cache.size(1),
                                              head_dim=128,
                                              rope_dim=64,
                                              rope_interleaved=rope_interleaved,
                                              BLOCK_D=128,
                                              num_warps=4,
                                              num_stages=1)
