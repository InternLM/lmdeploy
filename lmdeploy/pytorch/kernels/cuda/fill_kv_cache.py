# Copyright (c) OpenMMLab. All rights reserved.
from typing import Literal

import torch
import triton
import triton.language as tl
from torch import Tensor

from .triton_utils import get_kernel_meta, wrap_jit_func


@triton.jit
def _div_up(val, other):
    return (val + other - 1) // other


@triton.jit
def _quant_int8(val):
    val_min = tl.min(val, 1)
    val_max = tl.max(val, 1)
    scales = (val_max - val_min) / 255
    zeros = -val_min / scales
    q_val = (val / scales[:, None] + zeros[:, None] + 0.5).to(tl.uint8)
    return q_val, scales, zeros


@triton.jit
def _quant_int4(val1, val2):
    val1 = val1.to(tl.float32)
    val2 = val2.to(tl.float32)
    val_min = tl.min(tl.minimum(val1, val2), 1)
    val_max = tl.max(tl.maximum(val1, val2), 1)
    scales = (val_max - val_min) / 15
    zeros = -val_min / scales
    q_val1 = (val1 / scales[:, None] + zeros[:, None] + 0.5).to(tl.uint8)
    q_val2 = (val2 / scales[:, None] + zeros[:, None] + 0.5).to(tl.uint8)
    q_val = q_val1 + q_val2 * 16
    return q_val, scales, zeros


@wrap_jit_func(type_hint=dict(
    KStates=Tensor,
    VStates=Tensor,
    KCaches=Tensor,
    VCaches=Tensor,
    QStartLoc=Tensor,
    QSeqLens=Tensor,
    KVSeqLens=Tensor,
    BlockOffsets=Tensor,
    num_heads=torch.int32,
    head_dim=torch.int32,
    stride_kss=int,
    stride_ksh=int,
    stride_ksd=int,
    stride_vss=int,
    stride_vsh=int,
    stride_vsd=int,
    stride_kcn=int,
    stride_kcb=int,
    stride_kch=int,
    stride_kcd=int,
    stride_vcn=int,
    stride_vcb=int,
    stride_vch=int,
    stride_vcd=int,
    stride_boff=int,
    BLOCK=torch.int32,
    BLOCK_D=torch.int32,
    BLOCK_DV=torch.int32,
    BLOCK_H=torch.int32,
))
@triton.jit
def _fill_kv_cache_kernel(
    KStates,
    VStates,
    KCaches,
    VCaches,
    QStartLoc,
    QSeqLens,
    KVSeqLens,
    BlockOffsets,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    head_dim_v: tl.constexpr,
    stride_kss,
    stride_ksh,
    stride_ksd,
    stride_vss,
    stride_vsh,
    stride_vsd,
    stride_kcn: tl.constexpr,
    stride_kcb: tl.constexpr,
    stride_kch: tl.constexpr,
    stride_kcd: tl.constexpr,
    stride_vcn: tl.constexpr,
    stride_vcb: tl.constexpr,
    stride_vch: tl.constexpr,
    stride_vcd: tl.constexpr,
    stride_boff,
    BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """fill kv cache kernel."""
    batch_id = tl.program_id(0)
    block_id = tl.program_id(1)

    # initialize
    h_off = tl.arange(0, BLOCK_H)
    d_off = tl.arange(0, BLOCK_D)

    q_startloc = tl.load(QStartLoc + batch_id)
    q_seqlen = tl.load(QSeqLens + batch_id)
    kv_seqlen = tl.load(KVSeqLens + batch_id)
    history_seqlen = kv_seqlen - q_seqlen

    block0_first_tokenloc = history_seqlen % BLOCK

    state_token_offset = tl.maximum(block_id * BLOCK - block0_first_tokenloc,
                                    0)
    kv_block_id = _div_up(history_seqlen + 1, BLOCK) - 1 + block_id
    kv_block_id = min(kv_block_id, stride_boff - 1)
    block_off = tl.load(BlockOffsets + batch_id * stride_boff + kv_block_id)

    cur_startloc = q_startloc + state_token_offset
    ks_ptr = KStates + cur_startloc * stride_kss
    vs_ptr = VStates + cur_startloc * stride_vss

    kc_ptr = KCaches + block_off * stride_kcn
    vc_ptr = VCaches + block_off * stride_vcn

    c_first_tokenloc = block0_first_tokenloc
    if block_id != 0:
        c_first_tokenloc *= 0
    c_last_tokenloc = tl.minimum(
        BLOCK, q_seqlen + block0_first_tokenloc - block_id * BLOCK)

    for bidx in range(c_first_tokenloc, c_last_tokenloc):
        sidx = bidx - c_first_tokenloc
        mask = (h_off[:, None] < num_heads) & (d_off[None, :] < head_dim)
        k = tl.load(ks_ptr + sidx * stride_kss + h_off[:, None] * stride_ksh +
                    d_off[None, :] * stride_ksd,
                    mask=mask)
        tl.store(kc_ptr + bidx * stride_kcb + h_off[:, None] * stride_kch +
                 d_off[None, :] * stride_kcd,
                 k,
                 mask=mask)

        if BLOCK_DV > 0:
            dv_off = tl.arange(0, BLOCK_DV)
            maskv = (h_off[:, None] < num_heads) & (dv_off[None, :] <
                                                    head_dim_v)
            v = tl.load(vs_ptr + sidx * stride_vss +
                        h_off[:, None] * stride_vsh +
                        dv_off[None, :] * stride_vsd,
                        mask=maskv)
            tl.store(vc_ptr + bidx * stride_vcb + h_off[:, None] * stride_vch +
                     dv_off[None, :] * stride_vcd,
                     v,
                     mask=maskv)


@wrap_jit_func(type_hint=dict(
    KStates=Tensor,
    VStates=Tensor,
    KCaches=Tensor,
    VCaches=Tensor,
    KScalesZeros=Tensor,
    VScalesZeros=Tensor,
    QStartLoc=Tensor,
    QSeqLens=Tensor,
    KVSeqLens=Tensor,
    BlockOffsets=Tensor,
    num_heads=torch.int32,
    head_dim=torch.int32,
    stride_kss=int,
    stride_ksh=int,
    stride_ksd=int,
    stride_vss=int,
    stride_vsh=int,
    stride_vsd=int,
    stride_kcn=int,
    stride_kcb=int,
    stride_kch=int,
    stride_kcd=int,
    stride_vcn=int,
    stride_vcb=int,
    stride_vch=int,
    stride_vcd=int,
    stride_kszn=int,
    stride_kszb=int,
    stride_kszh=int,
    stride_kszd=int,
    stride_vszn=int,
    stride_vszb=int,
    stride_vszh=int,
    stride_vszd=int,
    stride_boff=int,
    BLOCK=torch.int32,
    BLOCK_D=torch.int32,
    BLOCK_DV=torch.int32,
    BLOCK_H=torch.int32,
))
@triton.jit
def _fill_kv_cache_quant_kernel(
    KStates,
    VStates,
    KCaches,
    VCaches,
    KScalesZeros,
    VScalesZeros,
    QStartLoc,
    QSeqLens,
    KVSeqLens,
    BlockOffsets,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    head_dim_v: tl.constexpr,
    stride_kss,
    stride_ksh,
    stride_ksd,
    stride_vss,
    stride_vsh,
    stride_vsd,
    stride_kcn: tl.constexpr,
    stride_kcb: tl.constexpr,
    stride_kch: tl.constexpr,
    stride_kcd: tl.constexpr,
    stride_vcn: tl.constexpr,
    stride_vcb: tl.constexpr,
    stride_vch: tl.constexpr,
    stride_vcd: tl.constexpr,
    stride_kszn: tl.constexpr,
    stride_kszb: tl.constexpr,
    stride_kszh: tl.constexpr,
    stride_kszd: tl.constexpr,
    stride_vszn: tl.constexpr,
    stride_vszb: tl.constexpr,
    stride_vszh: tl.constexpr,
    stride_vszd: tl.constexpr,
    quant_policy: tl.constexpr,
    stride_boff,
    BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """fill kv cache kernel with int4 and int8 quant fuzed.

    Args:
        stride_xss: stride of sequence length dim of key or value states
        stride_xsh: stride of head_num dim of key or value states
        stride_xsh: stride of head_size dim of key or value states
        stride_xn: stride of page num dim
        stride_xb: stride of block size dim
        stride_xh: stride of head_num dim
        stride_xd: stride of head_size dim
    """
    batch_id = tl.program_id(0)
    block_id = tl.program_id(1)
    d_off = tl.arange(0, BLOCK_D)

    # initialize
    h_off = tl.arange(0, BLOCK_H)
    szd_off = tl.arange(0, 2)

    q_startloc = tl.load(QStartLoc + batch_id)
    q_seqlen = tl.load(QSeqLens + batch_id)
    kv_seqlen = tl.load(KVSeqLens + batch_id)
    history_seqlen = kv_seqlen - q_seqlen

    block0_first_tokenloc = history_seqlen % BLOCK

    state_token_offset = tl.maximum(block_id * BLOCK - block0_first_tokenloc,
                                    0)
    kv_block_id = _div_up(history_seqlen + 1, BLOCK) - 1 + block_id
    kv_block_id = min(kv_block_id, stride_boff - 1)
    block_off = tl.load(BlockOffsets + batch_id * stride_boff + kv_block_id)

    cur_startloc = q_startloc + state_token_offset
    ks_ptr = KStates + cur_startloc * stride_kss
    vs_ptr = VStates + cur_startloc * stride_vss

    kc_ptr = KCaches + block_off * stride_kcn
    vc_ptr = VCaches + block_off * stride_vcn

    ksz_ptr = KScalesZeros + block_off * stride_kszn
    vsz_ptr = VScalesZeros + block_off * stride_vszn

    c_first_tokenloc = block0_first_tokenloc
    if block_id != 0:
        c_first_tokenloc *= 0
    c_last_tokenloc = tl.minimum(
        BLOCK, q_seqlen + block0_first_tokenloc - block_id * BLOCK)

    for bidx in range(c_first_tokenloc, c_last_tokenloc):
        sidx = bidx - c_first_tokenloc
        mask = (h_off[:, None] < num_heads) & (d_off[None, :] < head_dim)
        if quant_policy == 4:
            k1 = tl.load(ks_ptr + sidx * stride_kss +
                         h_off[:, None] * stride_ksh +
                         d_off[None, :] * stride_ksd,
                         mask=mask)
            k2 = tl.load(ks_ptr + sidx * stride_kss +
                         h_off[:, None] * stride_ksh +
                         d_off[None, :] * stride_ksd + head_dim * stride_ksd,
                         mask=mask)
            q_k, k_scales, k_zeros = _quant_int4(k1, k2)
        else:
            k = tl.load(ks_ptr + sidx * stride_kss +
                        h_off[:, None] * stride_ksh +
                        d_off[None, :] * stride_ksd,
                        mask=mask)
            q_k, k_scales, k_zeros = _quant_int8(k)
        tl.store(kc_ptr + bidx * stride_kcb + h_off[:, None] * stride_kch +
                 d_off[None, :] * stride_kcd,
                 q_k,
                 mask=mask)
        tl.store(ksz_ptr + bidx * stride_kszb + h_off[:, None] * stride_kszh +
                 szd_off[None, :] * stride_kszd,
                 k_scales[:, None],
                 mask=(h_off[:, None] < num_heads) & (szd_off[None, :] < 1))
        tl.store(ksz_ptr + bidx * stride_kszb + h_off[:, None] * stride_kszh +
                 szd_off[None, :] * stride_kszd,
                 k_zeros[:, None],
                 mask=(h_off[:, None] < num_heads) & (szd_off[None, :] == 1))

        if BLOCK_DV > 0:
            if quant_policy == 4:
                dv_off = tl.arange(0, BLOCK_DV //
                                   2)  # int4 pack, half the head_dim
                maskv = (h_off[:, None] < num_heads) & (dv_off[None, :] <
                                                        head_dim_v // 2)
                v1 = tl.load(vs_ptr + sidx * stride_vss +
                             h_off[:, None] * stride_vsh +
                             dv_off[None, :] * stride_vsd,
                             mask=maskv)
                v2 = tl.load(vs_ptr + sidx * stride_vss +
                             h_off[:, None] * stride_vsh +
                             dv_off[None, :] * stride_vsd +
                             head_dim_v // 2 * stride_vsd,
                             mask=maskv)
                q_v, v_scales, v_zeros = _quant_int4(v1, v2)
            else:
                dv_off = tl.arange(0, BLOCK_DV)
                maskv = (h_off[:, None] < num_heads) & (dv_off[None, :] <
                                                        head_dim_v)
                v = tl.load(vs_ptr + sidx * stride_vss +
                            h_off[:, None] * stride_vsh +
                            dv_off[None, :] * stride_vsd,
                            mask=maskv)
                q_v, v_scales, v_zeros = _quant_int8(v)
            tl.store(vc_ptr + bidx * stride_vcb + h_off[:, None] * stride_vch +
                     dv_off[None, :] * stride_vcd,
                     q_v,
                     mask=maskv)
            tl.store(
                vsz_ptr + bidx * stride_vszb + h_off[:, None] * stride_vszh +
                szd_off[None, :] * stride_vszd,
                v_scales[:, None],
                mask=(h_off[:, None] < num_heads) & (szd_off[None, :] < 1))
            tl.store(
                vsz_ptr + bidx * stride_vszb + h_off[:, None] * stride_vszh +
                szd_off[None, :] * stride_vszd,
                v_zeros[:, None],
                mask=(h_off[:, None] < num_heads) & (szd_off[None, :] == 1))


def fill_kv_cache(k_states: Tensor,
                  v_states: Tensor,
                  k_caches: Tensor,
                  v_caches: Tensor,
                  q_start_loc: Tensor,
                  q_seq_length: Tensor,
                  kv_seq_length: Tensor,
                  max_q_seq_length: int,
                  block_offsets: Tensor,
                  k_scales_zeros: Tensor = None,
                  v_scales_zeros: Tensor = None,
                  quant_policy: Literal[0, 4, 8] = 0,
                  kv_layout: str = 'bshd'):
    """fill key/value state to cache for paged attention."""
    if kv_layout == 'bshd':
        b_dim, s_dim, h_dim, d_dim = (0, 1, 2, 3)
    elif kv_layout == 'bhsd':
        b_dim, s_dim, h_dim, d_dim = (0, 2, 1, 3)
    else:
        raise RuntimeError('Unsupported layout.')

    block_offsets = block_offsets.contiguous()
    batch_size = block_offsets.size(0)
    block_size = k_caches.size(s_dim)
    num_heads = k_caches.size(h_dim)
    head_dim = k_caches.size(d_dim)
    head_dim_v = v_states.size(-1)
    max_num_blocks = triton.cdiv(max_q_seq_length, block_size) + 1

    BLOCK = block_size
    BLOCK_H = triton.next_power_of_2(num_heads)
    BLOCK_D = triton.next_power_of_2(head_dim)
    BLOCK_DV = triton.next_power_of_2(head_dim_v)
    grid = [batch_size, max_num_blocks]
    kernel_meta = get_kernel_meta(k_states)
    if quant_policy == 0:
        _fill_kv_cache_kernel[grid](
            k_states,
            v_states,
            k_caches,
            v_caches,
            q_start_loc,
            q_seq_length,
            kv_seq_length,
            block_offsets,
            num_heads=num_heads,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            stride_kss=k_states.stride(-3),
            stride_ksh=k_states.stride(-2),
            stride_ksd=k_states.stride(-1),
            stride_vss=v_states.stride(-3),
            stride_vsh=v_states.stride(-2),
            stride_vsd=v_states.stride(-1),
            stride_kcn=k_caches.stride(b_dim),
            stride_kcb=k_caches.stride(s_dim),
            stride_kch=k_caches.stride(h_dim),
            stride_kcd=k_caches.stride(d_dim),
            stride_vcn=v_caches.stride(b_dim),
            stride_vcb=v_caches.stride(s_dim),
            stride_vch=v_caches.stride(h_dim),
            stride_vcd=v_caches.stride(d_dim),
            stride_boff=block_offsets.stride(0),
            BLOCK=BLOCK,
            BLOCK_D=BLOCK_D,
            BLOCK_DV=BLOCK_DV,
            BLOCK_H=BLOCK_H,
            num_warps=4,
            num_stages=3,
            **kernel_meta,
        )
    else:
        _fill_kv_cache_quant_kernel[grid](
            k_states,
            v_states,
            k_caches,
            v_caches,
            k_scales_zeros,
            v_scales_zeros,
            q_start_loc,
            q_seq_length,
            kv_seq_length,
            block_offsets,
            num_heads=num_heads,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            stride_kss=k_states.stride(-3),
            stride_ksh=k_states.stride(-2),
            stride_ksd=k_states.stride(-1),
            stride_vss=v_states.stride(-3),
            stride_vsh=v_states.stride(-2),
            stride_vsd=v_states.stride(-1),
            stride_kcn=k_caches.stride(b_dim),
            stride_kcb=k_caches.stride(s_dim),
            stride_kch=k_caches.stride(h_dim),
            stride_kcd=k_caches.stride(d_dim),
            stride_vcn=v_caches.stride(b_dim),
            stride_vcb=v_caches.stride(s_dim),
            stride_vch=v_caches.stride(h_dim),
            stride_vcd=v_caches.stride(d_dim),
            stride_kszn=k_scales_zeros.stride(b_dim),
            stride_kszb=k_scales_zeros.stride(s_dim),
            stride_kszh=k_scales_zeros.stride(h_dim),
            stride_kszd=k_scales_zeros.stride(d_dim),
            stride_vszn=v_scales_zeros.stride(b_dim),
            stride_vszb=v_scales_zeros.stride(s_dim),
            stride_vszh=v_scales_zeros.stride(h_dim),
            stride_vszd=v_scales_zeros.stride(d_dim),
            quant_policy=quant_policy,
            stride_boff=block_offsets.stride(0),
            BLOCK=BLOCK,
            BLOCK_D=BLOCK_D,
            BLOCK_DV=BLOCK_DV,
            BLOCK_H=BLOCK_H,
            num_warps=4,
            num_stages=3,
            **kernel_meta,
        )
