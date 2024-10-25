# Copyright (c) OpenMMLab. All rights reserved.
from typing import Literal

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _flatten_kv_cache(
    kc_ptr,
    vc_ptr,
    ko_ptr,
    vo_ptr,
    start_loc_ptr,
    seqlens_ptr,
    block_offsets_ptr,
    stride_kcb: tl.constexpr,
    stride_kcs: tl.constexpr,
    stride_kch: tl.constexpr,
    stride_kcd: tl.constexpr,
    stride_vcb: tl.constexpr,
    stride_vcs: tl.constexpr,
    stride_vch: tl.constexpr,
    stride_vcd: tl.constexpr,
    stride_koh,
    stride_kos: tl.constexpr,
    stride_kod: tl.constexpr,
    stride_voh,
    stride_vos: tl.constexpr,
    stride_vod: tl.constexpr,
    stride_boff,
    HEAD_DIM_K: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    BLOCK_BS: tl.constexpr,
    BLOCK_DK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    """flatten kv cache."""
    page_id = tl.program_id(0)
    batch_id = tl.program_id(1)
    head_id = tl.program_id(2)

    seqlen = tl.load(seqlens_ptr + batch_id)
    if page_id * BLOCK_BS >= seqlen:
        return

    start_loc = tl.load(start_loc_ptr + batch_id)
    b_off = tl.load(block_offsets_ptr + batch_id * stride_boff + page_id)

    offs_bs = tl.arange(0, BLOCK_BS)
    offs_dk = tl.arange(0, BLOCK_DK) % HEAD_DIM_K
    offs_dv = tl.arange(0, BLOCK_DV) % HEAD_DIM_V
    offs_obs = page_id * BLOCK_BS + tl.arange(0, BLOCK_BS)
    mask_bs = offs_obs < seqlen
    mask_dk = tl.arange(0, BLOCK_DK) < HEAD_DIM_K
    mask_dv = tl.arange(0, BLOCK_DV) < HEAD_DIM_V

    kc_ptrs = (kc_ptr + b_off * stride_kcb + offs_bs[:, None] * stride_kcs +
               head_id * stride_kch + offs_dk[None, :] * stride_kcd)
    vc_ptrs = (vc_ptr + b_off * stride_vcb + offs_bs[:, None] * stride_vcs +
               head_id * stride_vch + offs_dv[None, :] * stride_vcd)
    ko_ptrs = (ko_ptr + head_id * stride_koh +
               (start_loc + offs_obs[:, None]) * stride_kos +
               offs_dk[None, :] * stride_kod)
    vo_ptrs = (vo_ptr + head_id * stride_voh +
               (start_loc + offs_obs[:, None]) * stride_vos +
               offs_dv[None, :] * stride_vod)

    kc = tl.load(kc_ptrs)
    tl.store(ko_ptrs, kc, mask=mask_bs[:, None] and mask_dk[None, :])
    vc = tl.load(vc_ptrs)
    tl.store(vo_ptrs, vc, mask=mask_bs[:, None] and mask_dv[None, :])


def flatten_kv_cache(k_caches: Tensor,
                     v_caches: Tensor,
                     seqlens: Tensor,
                     block_offsets: Tensor,
                     start_loc: Tensor = None,
                     out_size: int = None,
                     out_dtype: torch.dtype = None,
                     k_scales_zeros: Tensor = None,
                     v_scales_zeros: Tensor = None,
                     quant_policy: Literal[0, 4, 8] = 0,
                     kv_layout: str = 'bshd'):
    """recovery paged kv cache to normal kv cache."""
    if kv_layout == 'bshd':
        b_dim, s_dim, h_dim, d_dim = (0, 1, 2, 3)
    elif kv_layout == 'bhsd':
        b_dim, s_dim, h_dim, d_dim = (0, 2, 1, 3)
    else:
        raise RuntimeError('Unsupported layout.')

    if out_dtype is None:
        out_dtype = k_caches.dtype

    if out_size is None:
        out_size = k_caches.size(b_dim) * k_caches.size(s_dim)

    if start_loc is None:
        start_loc = seqlens.cumsum(0) - seqlens

    batch_size, num_blocks = block_offsets.size()
    num_heads = k_caches.size(h_dim)
    k_head_dim = k_caches.size(d_dim)
    v_head_dim = v_caches.size(d_dim)
    BLOCK_DK = triton.next_power_of_2(k_head_dim)
    BLOCK_DV = triton.next_power_of_2(v_head_dim)
    BLOCK_BS = k_caches.size(s_dim)

    k_states = k_caches.new_empty(num_heads,
                                  out_size,
                                  k_head_dim,
                                  dtype=out_dtype)
    v_states = v_caches.new_empty(num_heads,
                                  out_size,
                                  v_head_dim,
                                  dtype=out_dtype)

    grid = (num_blocks, batch_size, num_heads)
    _flatten_kv_cache[grid](
        k_caches,
        v_caches,
        k_states,
        v_states,
        start_loc,
        seqlens,
        block_offsets,
        stride_kcb=k_caches.stride(b_dim),
        stride_kcs=k_caches.stride(s_dim),
        stride_kch=k_caches.stride(h_dim),
        stride_kcd=k_caches.stride(d_dim),
        stride_vcb=v_caches.stride(b_dim),
        stride_vcs=v_caches.stride(s_dim),
        stride_vch=v_caches.stride(h_dim),
        stride_vcd=v_caches.stride(d_dim),
        stride_koh=k_states.stride(0),
        stride_kos=k_states.stride(1),
        stride_kod=k_states.stride(2),
        stride_voh=v_states.stride(0),
        stride_vos=v_states.stride(1),
        stride_vod=v_states.stride(2),
        stride_boff=block_offsets.stride(0),
        HEAD_DIM_K=k_head_dim,
        HEAD_DIM_V=v_head_dim,
        BLOCK_BS=BLOCK_BS,
        BLOCK_DK=BLOCK_DK,
        BLOCK_DV=BLOCK_DV,
    )

    return k_states, v_states
