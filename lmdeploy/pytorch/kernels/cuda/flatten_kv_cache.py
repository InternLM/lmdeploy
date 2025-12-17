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
    OUT_SIZE,
    HEAD_DIM_K: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    BLOCK_BS: tl.constexpr,
    BLOCK_DK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    """Flatten kv cache."""
    page_id = tl.program_id(0)
    batch_id = tl.program_id(1)
    head_id = tl.program_id(2)

    num_batches = tl.num_programs(1)

    seqlen = tl.load(seqlens_ptr + batch_id)
    start_loc = tl.load(start_loc_ptr + batch_id)
    # fill last block to prevent attention nan
    if batch_id == num_batches - 1:
        seqlen = (OUT_SIZE - start_loc).to(seqlen.dtype)
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

    kc_ptrs = (kc_ptr + b_off * stride_kcb + offs_bs[:, None] * stride_kcs + head_id * stride_kch +
               offs_dk[None, :] * stride_kcd)
    vc_ptrs = (vc_ptr + b_off * stride_vcb + offs_bs[:, None] * stride_vcs + head_id * stride_vch +
               offs_dv[None, :] * stride_vcd)
    ko_ptrs = (ko_ptr + head_id * stride_koh + (start_loc + offs_obs[:, None]) * stride_kos +
               offs_dk[None, :] * stride_kod)
    vo_ptrs = (vo_ptr + head_id * stride_voh + (start_loc + offs_obs[:, None]) * stride_vos +
               offs_dv[None, :] * stride_vod)

    kc = tl.load(kc_ptrs)
    tl.store(ko_ptrs, kc, mask=mask_bs[:, None] & mask_dk[None, :])
    if HEAD_DIM_V > 0:
        vc = tl.load(vc_ptrs)
        tl.store(vo_ptrs, vc, mask=mask_bs[:, None] & mask_dv[None, :])


@triton.jit
def _dequant_int4(val, HEAD_DIM: tl.constexpr, BLOCK: tl.constexpr):
    """Dequant int4."""
    offs = tl.arange(0, BLOCK) // (HEAD_DIM // 2)
    shift = (offs % 2) * 4
    return (val >> shift) & 0xf


@triton.jit
def _flatten_kv_cache_quant(
    kc_ptr,
    vc_ptr,
    ko_ptr,
    vo_ptr,
    ksz_ptr,
    vsz_ptr,
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
    stride_kszb: tl.constexpr,
    stride_kszs: tl.constexpr,
    stride_kszh: tl.constexpr,
    stride_kszd: tl.constexpr,
    stride_vszb: tl.constexpr,
    stride_vszs: tl.constexpr,
    stride_vszh: tl.constexpr,
    stride_vszd: tl.constexpr,
    stride_koh,
    stride_kos: tl.constexpr,
    stride_kod: tl.constexpr,
    stride_voh,
    stride_vos: tl.constexpr,
    stride_vod: tl.constexpr,
    stride_boff,
    quant_policy: tl.constexpr,
    OUT_SIZE,
    HEAD_DIM_K: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    BLOCK_BS: tl.constexpr,
    BLOCK_DK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    """Flatten kv cache."""
    page_id = tl.program_id(0)
    batch_id = tl.program_id(1)
    head_id = tl.program_id(2)

    num_batches = tl.num_programs(1)

    seqlen = tl.load(seqlens_ptr + batch_id)
    start_loc = tl.load(start_loc_ptr + batch_id)
    if batch_id == num_batches - 1:
        seqlen = OUT_SIZE - start_loc
    if page_id * BLOCK_BS >= seqlen:
        return

    b_off = tl.load(block_offsets_ptr + batch_id * stride_boff + page_id)

    offs_bs = tl.arange(0, BLOCK_BS)
    if quant_policy == 4:
        HALF_HDK: tl.constexpr = HEAD_DIM_K // 2
        HALF_HDV: tl.constexpr = HEAD_DIM_V // 2
        offs_dk = tl.arange(0, BLOCK_DK) % HALF_HDK
        offs_dv = tl.arange(0, BLOCK_DV) % HALF_HDV
    else:
        offs_dk = tl.arange(0, BLOCK_DK) % HEAD_DIM_K
        offs_dv = tl.arange(0, BLOCK_DV) % HEAD_DIM_V
    offs_obs = page_id * BLOCK_BS + tl.arange(0, BLOCK_BS)
    mask_bs = offs_obs < seqlen

    offs_dok = tl.arange(0, BLOCK_DK)
    offs_dov = tl.arange(0, BLOCK_DV)
    mask_dok = offs_dok < HEAD_DIM_K
    mask_dov = offs_dov < HEAD_DIM_V

    kc_ptrs = (kc_ptr + b_off * stride_kcb + offs_bs[:, None] * stride_kcs + head_id * stride_kch +
               offs_dk[None, :] * stride_kcd)
    vc_ptrs = (vc_ptr + b_off * stride_vcb + offs_bs[:, None] * stride_vcs + head_id * stride_vch +
               offs_dv[None, :] * stride_vcd)
    ksz_ptrs = (ksz_ptr + b_off * stride_kszb + offs_bs * stride_kszs + head_id * stride_kszh)
    vsz_ptrs = (vsz_ptr + b_off * stride_vszb + offs_bs * stride_vszs + head_id * stride_vszh)
    ko_ptrs = (ko_ptr + head_id * stride_koh + (start_loc + offs_obs[:, None]) * stride_kos +
               offs_dok[None, :] * stride_kod)
    vo_ptrs = (vo_ptr + head_id * stride_voh + (start_loc + offs_obs[:, None]) * stride_vos +
               offs_dov[None, :] * stride_vod)

    kc = tl.load(kc_ptrs)
    if quant_policy == 4:
        kc = _dequant_int4(kc, HEAD_DIM_K, BLOCK_DK)
    ks = tl.load(ksz_ptrs)
    kz = tl.load(ksz_ptrs + stride_kszd)
    ksz = ks * kz
    kq = (kc * ks[:, None] - ksz[:, None]).to(ko_ptr.dtype.element_ty)
    tl.store(ko_ptrs, kq, mask=mask_bs[:, None] & mask_dok[None, :])
    vc = tl.load(vc_ptrs)
    if quant_policy == 4:
        vc = _dequant_int4(vc, HEAD_DIM_V, BLOCK_DV)
    vs = tl.load(vsz_ptrs)
    vz = tl.load(vsz_ptrs + stride_vszd)
    vsz = vs * vz
    vq = (vc * vs[:, None] - vsz[:, None]).to(vo_ptr.dtype.element_ty)
    tl.store(vo_ptrs, vq, mask=mask_bs[:, None] & mask_dov[None, :])


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
                     kv_layout: str = 'bshd',
                     flatten_kv_layout: str = 'hsd'):
    """Recovery paged kv cache to normal kv cache."""
    if kv_layout == 'bshd':
        b_dim, s_dim, h_dim, d_dim = (0, 1, 2, 3)
    elif kv_layout == 'bhsd':
        b_dim, s_dim, h_dim, d_dim = (0, 2, 1, 3)
    else:
        raise RuntimeError('Unsupported layout.')

    if out_dtype is None:
        out_dtype = k_caches.dtype

    if out_size is None or out_size <= 0:
        out_size = k_caches.size(b_dim) * k_caches.size(s_dim)

    if start_loc is None:
        start_loc = seqlens.cumsum(0) - seqlens

    batch_size, num_blocks = block_offsets.size()
    num_heads = k_caches.size(h_dim)
    k_head_dim = k_caches.size(d_dim)
    v_head_dim = v_caches.size(d_dim)
    if quant_policy == 4:
        k_head_dim *= 2
        v_head_dim *= 2
    BLOCK_DK = triton.next_power_of_2(k_head_dim)
    BLOCK_DV = triton.next_power_of_2(v_head_dim)
    BLOCK_BS = k_caches.size(s_dim)
    shared_kv = k_caches.data_ptr() == v_caches.data_ptr() and v_head_dim < k_head_dim
    if flatten_kv_layout == 'hsd':
        k_states = k_caches.new_empty(num_heads, out_size, k_head_dim, dtype=out_dtype)
        if quant_policy == 0 and shared_kv:
            v_states = k_states[..., :v_head_dim]
            v_head_dim = 0
        else:
            v_states = v_caches.new_empty(num_heads, out_size, v_head_dim, dtype=out_dtype)
        stride_koh = k_states.stride(0)
        stride_kos = k_states.stride(1)
        stride_voh = v_states.stride(0)
        stride_vos = v_states.stride(1)
    elif flatten_kv_layout == 'shd':
        k_states = k_caches.new_empty(out_size, num_heads, k_head_dim, dtype=out_dtype)
        if quant_policy == 0 and shared_kv:
            v_states = k_states[..., :v_head_dim]
            v_head_dim = 0
        else:
            v_states = v_caches.new_empty(out_size, num_heads, v_head_dim, dtype=out_dtype)
        stride_koh = k_states.stride(1)
        stride_kos = k_states.stride(0)
        stride_voh = v_states.stride(1)
        stride_vos = v_states.stride(0)
    else:
        raise RuntimeError('Unsupported layout.')

    grid = (num_blocks, batch_size, num_heads)
    if quant_policy == 0:
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
            stride_koh=stride_koh,
            stride_kos=stride_kos,
            stride_kod=k_states.stride(2),
            stride_voh=stride_voh,
            stride_vos=stride_vos,
            stride_vod=v_states.stride(2),
            stride_boff=block_offsets.stride(0),
            OUT_SIZE=out_size,
            HEAD_DIM_K=k_head_dim,
            HEAD_DIM_V=v_head_dim,
            BLOCK_BS=BLOCK_BS,
            BLOCK_DK=BLOCK_DK,
            BLOCK_DV=BLOCK_DV,
        )
    else:
        _flatten_kv_cache_quant[grid](
            k_caches,
            v_caches,
            k_states,
            v_states,
            k_scales_zeros,
            v_scales_zeros,
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
            stride_kszb=k_scales_zeros.stride(b_dim),
            stride_kszs=k_scales_zeros.stride(s_dim),
            stride_kszh=k_scales_zeros.stride(h_dim),
            stride_kszd=k_scales_zeros.stride(d_dim),
            stride_vszb=v_scales_zeros.stride(b_dim),
            stride_vszs=v_scales_zeros.stride(s_dim),
            stride_vszh=v_scales_zeros.stride(h_dim),
            stride_vszd=v_scales_zeros.stride(d_dim),
            stride_koh=stride_koh,
            stride_kos=stride_kos,
            stride_kod=k_states.stride(2),
            stride_voh=stride_voh,
            stride_vos=stride_vos,
            stride_vod=v_states.stride(2),
            stride_boff=block_offsets.stride(0),
            quant_policy=quant_policy,
            OUT_SIZE=out_size,
            HEAD_DIM_K=k_head_dim,
            HEAD_DIM_V=v_head_dim,
            BLOCK_BS=BLOCK_BS,
            BLOCK_DK=BLOCK_DK,
            BLOCK_DV=BLOCK_DV,
        )

    return k_states, v_states


@triton.jit
def dequant_fp8(x, scale, GROUP_SIZE: tl.constexpr):
    """Dequant fp8."""
    M: tl.constexpr = x.shape[0]
    N: tl.constexpr = x.shape[1]
    x = x.to(scale.dtype)
    x = x.reshape(M, N // GROUP_SIZE, GROUP_SIZE)
    scale = scale.reshape(M, N // GROUP_SIZE, 1)
    x = x * scale
    x = x.reshape(M, N)
    return x


@triton.jit
def flatten_kv_cache_mla_fp8_kernel(
    kc_nope_ptr,
    kc_scale_ptr,
    kc_pe_ptr,
    ko_ptr,
    start_loc_ptr,
    seqlens_ptr,
    block_offsets_ptr,
    stride_kcb: tl.constexpr,
    stride_kcs: tl.constexpr,
    stride_kch: tl.constexpr,
    stride_kcd: tl.constexpr,
    stride_kcsb: tl.constexpr,
    stride_kcss: tl.constexpr,
    stride_kcsh: tl.constexpr,
    stride_kcsd: tl.constexpr,
    stride_kcpb: tl.constexpr,
    stride_kcps: tl.constexpr,
    stride_kcph: tl.constexpr,
    stride_kcpd: tl.constexpr,
    stride_koh,
    stride_kos: tl.constexpr,
    stride_kod: tl.constexpr,
    stride_boff,
    OUT_SIZE,
    BLOCK_BS: tl.constexpr,
    BLOCK_NOPE: tl.constexpr,
    BLOCK_PE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """Mla fp8 flatten kv cache kernel."""
    page_id = tl.program_id(0)
    batch_id = tl.program_id(1)
    head_id = tl.program_id(2)
    num_batches = tl.num_programs(1)

    seqlen = tl.load(seqlens_ptr + batch_id)
    start_loc = tl.load(start_loc_ptr + batch_id)
    # fill last block to prevent attention nan
    if batch_id == num_batches - 1:
        seqlen = OUT_SIZE - start_loc
    if page_id * BLOCK_BS >= seqlen:
        return

    b_off = tl.load(block_offsets_ptr + batch_id * stride_boff + page_id)

    BLOCK_SCALE: tl.constexpr = BLOCK_NOPE // GROUP_SIZE
    offs_bs = tl.arange(0, BLOCK_BS)
    offs_dnope = tl.arange(0, BLOCK_NOPE)
    offs_scale = tl.arange(0, BLOCK_SCALE)
    offs_dpe = tl.arange(0, BLOCK_PE)
    offs_obs = page_id * BLOCK_BS + tl.arange(0, BLOCK_BS)
    mask_bs = offs_obs < seqlen

    offs_kc = b_off * stride_kcb + offs_bs[:, None] * stride_kcs + head_id * stride_kch
    kc_nope_ptrs = (kc_nope_ptr + offs_kc + offs_dnope[None, :] * stride_kcd)

    offs_kc_scale = b_off * stride_kcsb + offs_bs[:, None] * stride_kcss + head_id * stride_kcsh
    kc_scale_ptrs = (kc_scale_ptr + offs_kc_scale + offs_scale[None, :] * stride_kcsd)

    offs_kc_pe = b_off * stride_kcpb + offs_bs[:, None] * stride_kcps + head_id * stride_kcph
    kc_pe_ptrs = (kc_pe_ptr + offs_kc_pe + offs_dpe[None, :] * stride_kcpd)

    offs_ko = head_id * stride_koh + (start_loc + offs_obs[:, None]) * stride_kos
    ko_nope_ptrs = (ko_ptr + offs_ko + offs_dnope[None, :] * stride_kod)
    ko_pe_ptrs = (ko_ptr + offs_ko + (BLOCK_NOPE + offs_dpe[None, :]) * stride_kod)

    # nope
    kc_nope = tl.load(kc_nope_ptrs)
    kc_scale = tl.load(kc_scale_ptrs)
    ko_nope = dequant_fp8(kc_nope, kc_scale, GROUP_SIZE)
    ko_nope = ko_nope.to(ko_ptr.dtype.element_ty)
    tl.store(ko_nope_ptrs, ko_nope, mask=mask_bs[:, None])

    # pe
    kc_pe = tl.load(kc_pe_ptrs)
    tl.store(ko_pe_ptrs, kc_pe, mask=mask_bs[:, None])


def flatten_kv_cache_mla_fp8(k_caches: Tensor,
                             seqlens: Tensor,
                             block_offsets: Tensor,
                             start_loc: Tensor = None,
                             out_size: int = None,
                             out_dtype: torch.dtype = None,
                             flatten_kv_layout: str = 'hsd'):
    """This kernel is designed to support mla fp8."""
    assert k_caches.dim() == 4

    b_dim, s_dim, h_dim, d_dim = (0, 1, 2, 3)

    if out_dtype is None:
        out_dtype = torch.bfloat16

    if out_size is None or out_size <= 0:
        out_size = k_caches.size(b_dim) * k_caches.size(s_dim)

    # TODO: DIRTY magic number
    k_caches_nope = k_caches[..., :512]
    k_caches_scale = k_caches[..., 512:512 + 16].view(torch.float32)
    k_caches_pe = k_caches[..., 512 + 16:].view(out_dtype)

    if start_loc is None:
        start_loc = seqlens.cumsum(0) - seqlens

    batch_size, num_blocks = block_offsets.size()
    num_heads = k_caches.size(h_dim)
    k_head_dim = 576
    BLOCK_NOPE = 512
    BLOCK_PE = 64
    BLOCK_BS = k_caches.size(s_dim)
    if flatten_kv_layout == 'hsd':
        k_states = k_caches.new_empty(num_heads, out_size, k_head_dim, dtype=out_dtype)
        stride_koh = k_states.stride(0)
        stride_kos = k_states.stride(1)
    elif flatten_kv_layout == 'shd':
        k_states = k_caches.new_empty(out_size, num_heads, k_head_dim, dtype=out_dtype)
        stride_koh = k_states.stride(1)
        stride_kos = k_states.stride(0)
    else:
        raise RuntimeError(f'Unsupported layout: {flatten_kv_layout}.')

    grid = (num_blocks, batch_size, num_heads)
    flatten_kv_cache_mla_fp8_kernel[grid](
        k_caches_nope,
        k_caches_scale,
        k_caches_pe,
        k_states,
        start_loc,
        seqlens,
        block_offsets,
        stride_kcb=k_caches_nope.stride(b_dim),
        stride_kcs=k_caches_nope.stride(s_dim),
        stride_kch=k_caches_nope.stride(h_dim),
        stride_kcd=k_caches_nope.stride(d_dim),
        stride_kcsb=k_caches_scale.stride(b_dim),
        stride_kcss=k_caches_scale.stride(s_dim),
        stride_kcsh=k_caches_scale.stride(h_dim),
        stride_kcsd=k_caches_scale.stride(d_dim),
        stride_kcpb=k_caches_pe.stride(b_dim),
        stride_kcps=k_caches_pe.stride(s_dim),
        stride_kcph=k_caches_pe.stride(h_dim),
        stride_kcpd=k_caches_pe.stride(d_dim),
        stride_koh=stride_koh,
        stride_kos=stride_kos,
        stride_kod=k_states.stride(2),
        stride_boff=block_offsets.stride(0),
        OUT_SIZE=out_size,
        BLOCK_BS=BLOCK_BS,
        BLOCK_NOPE=BLOCK_NOPE,
        BLOCK_PE=BLOCK_PE,
        GROUP_SIZE=128,
    )

    return k_states
