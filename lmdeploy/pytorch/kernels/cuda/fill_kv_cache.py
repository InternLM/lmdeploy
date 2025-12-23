# Copyright (c) OpenMMLab. All rights reserved.
from typing import Literal, Optional

import torch
import triton
import triton.language as tl
from torch import Tensor


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
    is_decoding: tl.constexpr,
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
):
    """Fill kv cache kernel."""
    batch_id = tl.program_id(2)
    head_id = tl.program_id(0)
    block_id = tl.program_id(1)

    q_startloc = tl.load(QStartLoc + batch_id)
    q_seqlen = tl.load(QSeqLens + batch_id)
    kv_seqlen = tl.load(KVSeqLens + batch_id)
    history_seqlen = kv_seqlen - q_seqlen

    kv_block_id = history_seqlen // BLOCK + block_id

    if kv_seqlen <= 0:
        return

    if kv_block_id * BLOCK >= kv_seqlen:
        return

    if is_decoding:
        page_offs = tl.full((1, ), history_seqlen % BLOCK, dtype=tl.int32)
        kv_mask = tl.full((1, ), 1, dtype=tl.int1)
        q_offs = tl.full((1, ), q_startloc, dtype=tl.int32)
    else:
        page_offs = tl.arange(0, BLOCK)
        kv_offs = kv_block_id * BLOCK + page_offs
        kv_mask = (kv_offs >= history_seqlen) & (kv_offs < kv_seqlen)
        token_off = q_startloc + kv_block_id * BLOCK - history_seqlen
        q_offs = token_off + page_offs

    block_off = tl.load(BlockOffsets + batch_id * stride_boff + kv_block_id)

    d_off = tl.arange(0, BLOCK_D)
    mask_ks = kv_mask[:, None]
    mask_kc = mask_ks & (d_off[None, :] < head_dim)
    d_off = d_off % head_dim

    ks_ptr = KStates + head_id * stride_ksh
    ks_ptrs = ks_ptr + q_offs[:, None] * stride_kss + d_off[None, :] * stride_ksd
    kc_ptr = KCaches + block_off * stride_kcn + head_id * stride_kch
    kc_ptrs = kc_ptr + page_offs[:, None] * stride_kcb + d_off[None, :] * stride_kcd

    if BLOCK_DV > 0:
        dv_off = tl.arange(0, BLOCK_DV)
        mask_vs = kv_mask[:, None]
        mask_vc = mask_vs & (dv_off[None, :] < head_dim_v)
        dv_off = dv_off % head_dim_v
        vs_ptr = VStates + head_id * stride_vsh
        vs_ptrs = vs_ptr + q_offs[:, None] * stride_vss + dv_off[None, :] * stride_vsd
        vc_ptr = VCaches + block_off * stride_vcn + head_id * stride_vch
        vc_ptrs = vc_ptr + page_offs[:, None] * stride_vcb + dv_off[None, :] * stride_vcd

    k = tl.load(ks_ptrs, mask=mask_ks)
    if BLOCK_DV > 0:
        v = tl.load(vs_ptrs, mask=mask_vs)
    tl.store(kc_ptrs, k, mask=mask_kc)
    if BLOCK_DV > 0:
        tl.store(vc_ptrs, v, mask=mask_vc)


@triton.jit
def _fill_page_quant_int8(
    state_ptr,
    cache_ptr,
    scales_zeros_ptr,
    block_off,
    head_id,
    page_offs,
    q_offs,
    kv_mask,
    head_dim: tl.constexpr,
    stride_ss,
    stride_sh,
    stride_sd,
    stride_cn: tl.constexpr,
    stride_cb: tl.constexpr,
    stride_ch: tl.constexpr,
    stride_cd: tl.constexpr,
    stride_szn: tl.constexpr,
    stride_szb: tl.constexpr,
    stride_szh: tl.constexpr,
    stride_szd: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fill page int8."""
    d_off = tl.arange(0, BLOCK_D)
    mask_kc = kv_mask[:, None] & (d_off[None, :] < head_dim)
    d_off = d_off % head_dim
    state_ptr = state_ptr + head_id * stride_sh
    state_ptrs = state_ptr + q_offs[:, None] * stride_ss + d_off[None, :] * stride_sd
    cache_ptr = cache_ptr + block_off * stride_cn + head_id * stride_ch
    cache_ptrs = cache_ptr + page_offs[:, None] * stride_cb + d_off[None, :] * stride_cd
    scales_zeros_ptr = scales_zeros_ptr + block_off * stride_szn + head_id * stride_szh
    scales_ptrs = scales_zeros_ptr + page_offs[:, None] * stride_szb
    zeros_ptrs = scales_ptrs + stride_szd

    state = tl.load(state_ptrs, mask=kv_mask[:, None])
    state, scales, zeros = _quant_int8(state)

    tl.store(cache_ptrs, state, mask=mask_kc)
    tl.store(scales_ptrs, scales[:, None], mask=kv_mask[:, None])
    tl.store(zeros_ptrs, zeros[:, None], mask=kv_mask[:, None])


@triton.jit
def _fill_page_quant_int4(
    state_ptr,
    cache_ptr,
    scales_zeros_ptr,
    block_off,
    head_id,
    page_offs,
    q_offs,
    kv_mask,
    head_dim: tl.constexpr,
    stride_ss,
    stride_sh,
    stride_sd,
    stride_cn: tl.constexpr,
    stride_cb: tl.constexpr,
    stride_ch: tl.constexpr,
    stride_cd: tl.constexpr,
    stride_szn: tl.constexpr,
    stride_szb: tl.constexpr,
    stride_szh: tl.constexpr,
    stride_szd: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fill page int4."""
    d_off = tl.arange(0, BLOCK_D)
    mask_kc = kv_mask[:, None] & (d_off[None, :] < head_dim)
    state_ptr = state_ptr + head_id * stride_sh
    state0_ptrs = state_ptr + q_offs[:, None] * stride_ss + d_off[None, :] * stride_sd
    state1_ptrs = state0_ptrs + head_dim * stride_sd
    cache_ptr = cache_ptr + block_off * stride_cn + head_id * stride_ch
    cache_ptrs = cache_ptr + page_offs[:, None] * stride_cb + d_off[None, :] * stride_cd
    scales_zeros_ptr = scales_zeros_ptr + block_off * stride_szn + head_id * stride_szh
    scales_ptrs = scales_zeros_ptr + page_offs[:, None] * stride_szb
    zeros_ptrs = scales_ptrs + stride_szd

    state0 = tl.load(state0_ptrs, mask=mask_kc)
    state1 = tl.load(state1_ptrs, mask=mask_kc)
    state, scales, zeros = _quant_int4(state0, state1)

    tl.store(cache_ptrs, state, mask=mask_kc)
    tl.store(scales_ptrs, scales[:, None], mask=kv_mask[:, None])
    tl.store(zeros_ptrs, zeros[:, None], mask=kv_mask[:, None])


@triton.jit
def _fill_page_quant(state_ptr, cache_ptr, scales_zeros_ptr, block_off, head_id, page_offs, q_offs, kv_mask,
                     head_dim: tl.constexpr, stride_ss, stride_sh, stride_sd, stride_cn: tl.constexpr,
                     stride_cb: tl.constexpr, stride_ch: tl.constexpr, stride_cd: tl.constexpr,
                     stride_szn: tl.constexpr, stride_szb: tl.constexpr, stride_szh: tl.constexpr,
                     stride_szd: tl.constexpr, BLOCK_D: tl.constexpr, quant_policy: tl.constexpr):
    """Fill page."""
    if quant_policy == 8:
        return _fill_page_quant_int8(state_ptr,
                                     cache_ptr,
                                     scales_zeros_ptr,
                                     block_off,
                                     head_id,
                                     page_offs,
                                     q_offs,
                                     kv_mask,
                                     head_dim=head_dim,
                                     stride_ss=stride_ss,
                                     stride_sh=stride_sh,
                                     stride_sd=stride_sd,
                                     stride_cn=stride_cn,
                                     stride_cb=stride_cb,
                                     stride_ch=stride_ch,
                                     stride_cd=stride_cd,
                                     stride_szn=stride_szn,
                                     stride_szb=stride_szb,
                                     stride_szh=stride_szh,
                                     stride_szd=stride_szd,
                                     BLOCK_D=BLOCK_D)
    elif quant_policy == 4:
        return _fill_page_quant_int4(state_ptr,
                                     cache_ptr,
                                     scales_zeros_ptr,
                                     block_off,
                                     head_id,
                                     page_offs,
                                     q_offs,
                                     kv_mask,
                                     head_dim=head_dim,
                                     stride_ss=stride_ss,
                                     stride_sh=stride_sh,
                                     stride_sd=stride_sd,
                                     stride_cn=stride_cn,
                                     stride_cb=stride_cb,
                                     stride_ch=stride_ch,
                                     stride_cd=stride_cd,
                                     stride_szn=stride_szn,
                                     stride_szb=stride_szb,
                                     stride_szh=stride_szh,
                                     stride_szd=stride_szd,
                                     BLOCK_D=BLOCK_D)
    else:
        tl.static_assert(False, 'Unsupported quant policy')


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
    is_decoding: tl.constexpr,
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
):
    """Fill kv cache kernel with int4 and int8 quant fuzed.

    Args:
        stride_xss: stride of sequence length dim of key or value states
        stride_xsh: stride of head_num dim of key or value states
        stride_xsh: stride of head_size dim of key or value states
        stride_xn: stride of page num dim
        stride_xb: stride of block size dim
        stride_xh: stride of head_num dim
        stride_xd: stride of head_size dim
    """
    batch_id = tl.program_id(2)
    head_id = tl.program_id(0)
    block_id = tl.program_id(1)

    q_startloc = tl.load(QStartLoc + batch_id)
    q_seqlen = tl.load(QSeqLens + batch_id)
    kv_seqlen = tl.load(KVSeqLens + batch_id)
    history_seqlen = kv_seqlen - q_seqlen

    kv_block_id = history_seqlen // BLOCK + block_id

    if kv_seqlen <= 0:
        return

    if kv_block_id * BLOCK >= kv_seqlen:
        return

    if is_decoding:
        page_offs = tl.full((1, ), history_seqlen % BLOCK, dtype=tl.int32)
        kv_mask = tl.full((1, ), 1, dtype=tl.int1)
        q_offs = tl.full((1, ), q_startloc, dtype=tl.int32)
    else:
        page_offs = tl.arange(0, BLOCK)
        kv_offs = kv_block_id * BLOCK + page_offs
        kv_mask = (kv_offs >= history_seqlen) & (kv_offs < kv_seqlen)
        token_off = q_startloc + kv_block_id * BLOCK - history_seqlen
        q_offs = token_off + page_offs

    block_off = tl.load(BlockOffsets + batch_id * stride_boff + kv_block_id)

    _fill_page_quant(KStates,
                     KCaches,
                     KScalesZeros,
                     block_off,
                     head_id,
                     page_offs,
                     q_offs,
                     kv_mask,
                     head_dim=head_dim,
                     stride_ss=stride_kss,
                     stride_sh=stride_ksh,
                     stride_sd=stride_ksd,
                     stride_cn=stride_kcn,
                     stride_cb=stride_kcb,
                     stride_ch=stride_kch,
                     stride_cd=stride_kcd,
                     stride_szn=stride_kszn,
                     stride_szb=stride_kszb,
                     stride_szh=stride_kszh,
                     stride_szd=stride_kszd,
                     BLOCK_D=BLOCK_D,
                     quant_policy=quant_policy)

    if BLOCK_DV > 0:
        _fill_page_quant(VStates,
                         VCaches,
                         VScalesZeros,
                         block_off,
                         head_id,
                         page_offs,
                         q_offs,
                         kv_mask,
                         head_dim=head_dim_v,
                         stride_ss=stride_vss,
                         stride_sh=stride_vsh,
                         stride_sd=stride_vsd,
                         stride_cn=stride_vcn,
                         stride_cb=stride_vcb,
                         stride_ch=stride_vch,
                         stride_cd=stride_vcd,
                         stride_szn=stride_vszn,
                         stride_szb=stride_vszb,
                         stride_szh=stride_vszh,
                         stride_szd=stride_vszd,
                         BLOCK_D=BLOCK_DV,
                         quant_policy=quant_policy)


def fill_kv_cache(k_states: Tensor,
                  v_states: Optional[Tensor],
                  k_caches: Tensor,
                  v_caches: Optional[Tensor],
                  q_start_loc: Tensor,
                  q_seq_length: Tensor,
                  kv_seq_length: Tensor,
                  max_q_seq_length: int,
                  block_offsets: Tensor,
                  k_scales_zeros: Tensor = None,
                  v_scales_zeros: Tensor = None,
                  quant_policy: Literal[0, 4, 8] = 0,
                  kv_layout: str = 'bshd'):
    """Fill key/value state to cache for paged attention."""
    if kv_layout == 'bshd':
        b_dim, s_dim, h_dim, d_dim = (0, 1, 2, 3)
    elif kv_layout == 'bhsd':
        b_dim, s_dim, h_dim, d_dim = (0, 2, 1, 3)
    else:
        raise RuntimeError('Unsupported layout.')
    if v_states is None:
        v_states = k_states[..., :0]
    if v_caches is None:
        v_caches = k_caches[..., :0]

    block_offsets = block_offsets.contiguous()
    batch_size = block_offsets.size(0)
    block_size = k_caches.size(s_dim)
    num_heads = k_caches.size(h_dim)
    head_dim = k_caches.size(d_dim)
    head_dim_v = v_caches.size(d_dim)
    if v_states.size(-1) == 0:
        head_dim_v = 0
    if max_q_seq_length == 1:
        max_num_blocks = 1
    else:
        max_num_blocks = triton.cdiv(max_q_seq_length, block_size) + 1

    BLOCK = block_size
    BLOCK_D = triton.next_power_of_2(head_dim)
    BLOCK_DV = triton.next_power_of_2(head_dim_v)
    if k_caches.data_ptr() == v_caches.data_ptr() and head_dim_v <= head_dim:
        BLOCK_DV = 0
    grid = (num_heads, max_num_blocks, batch_size)
    is_decoding = max_num_blocks == 1
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
            is_decoding=is_decoding,
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
            num_warps=4,
            num_stages=3,
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
            is_decoding=is_decoding,
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
            num_warps=4,
            num_stages=1,
        )


@triton.jit
def fast_log2_ceil(x):
    bits_x = tl.cast(x, tl.uint32, bitcast=True)
    exp_x = (bits_x >> 23) & 0xFF
    man_bits = bits_x & ((1 << 23) - 1)
    tmp = exp_x - 127 + tl.where(man_bits != 0, 1, 0)
    return tl.cast(tmp, tl.int32)


@triton.jit
def fast_pow2(x):
    bits_x = (x + 127) << 23
    return tl.cast(bits_x, tl.float32, bitcast=True)


@triton.jit
def fast_round_scale(amax, fp8_max_inv):
    return fast_pow2(fast_log2_ceil(amax * fp8_max_inv))


@triton.jit
def _quant_blocked_fp8(x,
                       fp8_min: tl.constexpr,
                       fp8_max: tl.constexpr,
                       dtype: tl.constexpr,
                       GROUP_SIZE: tl.constexpr = 128,
                       ROUND_SCALE: tl.constexpr = 0):
    x = x.to(tl.float32)
    M: tl.constexpr = x.shape[0]
    N: tl.constexpr = x.shape[1]
    rfp8_max: tl.constexpr = 1 / fp8_max
    x = x.reshape(M, N // GROUP_SIZE, GROUP_SIZE)
    amax = tl.maximum(tl.max(tl.abs(x), axis=2, keep_dims=True), 1e-6)
    if ROUND_SCALE == 1:
        scale = fast_round_scale(amax, rfp8_max)
    else:
        scale = amax * rfp8_max
    out = x / scale

    out = tl.clamp(out, fp8_min, fp8_max)
    out = out.to(dtype)
    out = out.reshape(M, N)
    scale = scale.reshape(M, N // GROUP_SIZE)
    return out, scale


@triton.jit
def _fill_kv_cache_blocked_fp8_kernel(
    KStates,
    VStates,
    KCaches,
    VCaches,
    KSCaches,
    VSCaches,
    cu_seqlen_q_ptr,
    KVSeqLens,
    BlockOffsets,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    is_decoding: tl.constexpr,
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
    stride_kscn: tl.constexpr,
    stride_kscb: tl.constexpr,
    stride_ksch: tl.constexpr,
    stride_kscd: tl.constexpr,
    stride_vscn: tl.constexpr,
    stride_vscb: tl.constexpr,
    stride_vsch: tl.constexpr,
    stride_vscd: tl.constexpr,
    stride_boff,
    ROUND_SCALE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    """Fill kv cache kernel."""
    batch_id = tl.program_id(2)
    head_id = tl.program_id(0)
    block_id = tl.program_id(1)

    q_startloc = tl.load(cu_seqlen_q_ptr + batch_id)
    q_seqlen = tl.load(cu_seqlen_q_ptr + batch_id + 1) - q_startloc
    kv_seqlen = tl.load(KVSeqLens + batch_id)
    history_seqlen = kv_seqlen - q_seqlen

    kv_block_id = history_seqlen // BLOCK + block_id

    if kv_seqlen <= 0:
        return

    if kv_block_id * BLOCK >= kv_seqlen:
        return

    if is_decoding:
        page_offs = tl.full((1, ), history_seqlen % BLOCK, dtype=tl.int32)
        kv_mask = tl.full((1, ), 1, dtype=tl.int1)
        q_offs = tl.full((1, ), q_startloc, dtype=tl.int32)
    else:
        page_offs = tl.arange(0, BLOCK)
        kv_offs = kv_block_id * BLOCK + page_offs
        kv_mask = (kv_offs >= history_seqlen) & (kv_offs < kv_seqlen)
        token_off = q_startloc + kv_block_id * BLOCK - history_seqlen
        q_offs = token_off + page_offs

    block_off = tl.load(BlockOffsets + batch_id * stride_boff + kv_block_id)

    d_off = tl.arange(0, BLOCK_D)
    mask_ks = kv_mask[:, None]
    mask_kc = mask_ks & (d_off[None, :] < head_dim)
    d_off = d_off % head_dim

    BLOCK_DS: tl.constexpr = (BLOCK_D + GROUP_SIZE - 1) // GROUP_SIZE
    ds_off = tl.arange(0, BLOCK_DS)

    ks_ptr = KStates + head_id * stride_ksh
    ks_ptrs = ks_ptr + q_offs[:, None] * stride_kss + d_off[None, :] * stride_ksd
    kc_ptr = KCaches + block_off * stride_kcn + head_id * stride_kch
    kc_ptrs = kc_ptr + page_offs[:, None] * stride_kcb + d_off[None, :] * stride_kcd
    ksc_ptr = KSCaches + block_off * stride_kscn + head_id * stride_ksch
    ksc_ptrs = ksc_ptr + page_offs[:, None] * stride_kscb + ds_off[None, :] * stride_kscd

    if BLOCK_DV > 0:
        dv_off = tl.arange(0, BLOCK_DV)
        mask_vs = kv_mask[:, None]
        mask_vc = mask_vs & (dv_off[None, :] < head_dim_v)

        BLOCK_DVS: tl.constexpr = (BLOCK_DV + GROUP_SIZE - 1) // GROUP_SIZE
        dvs_off = tl.arange(0, BLOCK_DVS)

        dv_off = dv_off % head_dim_v
        vs_ptr = VStates + head_id * stride_vsh
        vs_ptrs = vs_ptr + q_offs[:, None] * stride_vss + dv_off[None, :] * stride_vsd
        vc_ptr = VCaches + block_off * stride_vcn + head_id * stride_vch
        vc_ptrs = vc_ptr + page_offs[:, None] * stride_vcb + dv_off[None, :] * stride_vcd
        vsc_ptr = VSCaches + block_off * stride_vscn + head_id * stride_vsch
        vsc_ptrs = vsc_ptr + page_offs[:, None] * stride_vscb + dvs_off[None, :] * stride_vscd

    k = tl.load(ks_ptrs, mask=mask_ks)
    if BLOCK_DV > 0:
        v = tl.load(vs_ptrs, mask=mask_vs)
    kc, kcs = _quant_blocked_fp8(k, fp8_min, fp8_max, KCaches.dtype.element_ty, GROUP_SIZE, ROUND_SCALE)
    tl.store(kc_ptrs, kc, mask=mask_kc)
    tl.store(ksc_ptrs, kcs, mask=kv_mask[:, None] & (ds_off[None, :] < tl.cdiv(head_dim, GROUP_SIZE)))
    if BLOCK_DV > 0:
        vc, vcs = _quant_blocked_fp8(v, fp8_min, fp8_max, VCaches.dtype.element_ty, GROUP_SIZE, ROUND_SCALE)
        tl.store(vc_ptrs, vc, mask=mask_vc)
        tl.store(vsc_ptrs, vcs, mask=kv_mask[:, None] & (ds_off[None, :] < tl.cdiv(head_dim_v, GROUP_SIZE)))


def fill_kv_cache_blocked_fp8(k_states: Tensor,
                              v_states: Optional[Tensor],
                              k_caches: Tensor,
                              v_caches: Optional[Tensor],
                              ks_caches: Tensor,
                              vs_caches: Optional[Tensor],
                              cu_seqlen_q: Tensor,
                              kv_seqlens: Tensor,
                              max_q_seqlen: int,
                              block_offsets: Tensor,
                              group_size: int = 128,
                              kv_layout: str = 'bshd',
                              scale_fmt: Optional[str] = None):
    """Fill key/value state to cache for paged attention with fp8 quantization.

    Args:
        k_states (Tensor): Key states of shape
            (seq_length, num_heads, head_dim).
        v_states (Optional[Tensor]): Value states of shape
            (seq_length, num_heads, head_dim_v). If None, no value states
            are processed.
        k_caches (Tensor): 4D k cache, shape depends on ``kv_layout``.
        v_caches (Optional[Tensor]): 4D v cache, shape depends on
            ``kv_layout``. If None, no value caches are processed.
        ks_caches (Tensor): 4D k scale cache, shape depends on
            ``kv_layout``.
        vs_caches (Optional[Tensor]): 4D v scale cache, shape depends on
            ``kv_layout``. If None, no value scale caches are processed.
        cu_seqlen_q (Tensor): Cumulative sequence lengths of queries,
            shape (batch_size + 1, ).
        kv_seqlens (Tensor): Sequence lengths of key/values, shape
            (batch_size, ).
        max_q_seqlen (int): Maximum sequence length of queries.
        block_offsets (Tensor): Block offsets for each batch, shape
            (batch_size, ).
        group_size (int, optional): Group size for fp8 quantization. Default
            is 128.
        kv_layout (str, optional): Layout of key/value caches. Valid values
            are ``'bshd'`` and ``'bhsd'``. Default is ``'bshd'``.
        scale_fmt (str, optional): Format of the fp8 scaling factors. Valid
            values are ``None`` and ``'ue8m0'``. When set to ``'ue8m0'``,
            scaling factors are stored/interpreted using the UE8M0 fp8 scale
            format; when ``None``, the default scale layout for this kernel
            is used.
    """
    assert scale_fmt in (None, 'ue8m0'), f'Unsupported scale format: {scale_fmt}.'

    if kv_layout == 'bshd':
        b_dim, s_dim, h_dim, d_dim = (0, 1, 2, 3)
    elif kv_layout == 'bhsd':
        b_dim, s_dim, h_dim, d_dim = (0, 2, 1, 3)
    else:
        raise RuntimeError('Unsupported layout.')

    if v_states is None:
        v_states = k_states[..., :0]
    if v_caches is None:
        v_caches = k_caches[..., :0]
    if vs_caches is None:
        vs_caches = ks_caches[..., :0]

    block_offsets = block_offsets.contiguous()
    batch_size = block_offsets.size(0)
    block_size = k_caches.size(s_dim)
    num_heads = k_caches.size(h_dim)
    head_dim = k_caches.size(d_dim)
    head_dim_v = v_states.size(-1)
    if max_q_seqlen == 1:
        max_num_blocks = 1
    else:
        max_num_blocks = triton.cdiv(max_q_seqlen, block_size) + 1

    BLOCK = block_size
    BLOCK_D = triton.next_power_of_2(head_dim)
    BLOCK_DV = triton.next_power_of_2(head_dim_v)
    if k_caches.data_ptr() == v_caches.data_ptr() and head_dim_v <= head_dim:
        BLOCK_DV = 0

    dtype = k_caches.dtype
    finfo = torch.finfo(dtype)
    fmin = finfo.min
    fmax = finfo.max

    grid = (num_heads, max_num_blocks, batch_size)
    ROUND_SCALE = 1 if scale_fmt == 'ue8m0' else 0
    is_decoding = max_q_seqlen == 1
    _fill_kv_cache_blocked_fp8_kernel[grid](
        k_states,
        v_states,
        k_caches,
        v_caches,
        ks_caches,
        vs_caches,
        cu_seqlen_q,
        kv_seqlens,
        block_offsets,
        fp8_min=fmin,
        fp8_max=fmax,
        is_decoding=is_decoding,
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
        stride_kscn=ks_caches.stride(b_dim),
        stride_kscb=ks_caches.stride(s_dim),
        stride_ksch=ks_caches.stride(h_dim),
        stride_kscd=ks_caches.stride(d_dim),
        stride_vscn=vs_caches.stride(b_dim),
        stride_vscb=vs_caches.stride(s_dim),
        stride_vsch=vs_caches.stride(h_dim),
        stride_vscd=vs_caches.stride(d_dim),
        stride_boff=block_offsets.stride(0),
        ROUND_SCALE=ROUND_SCALE,
        GROUP_SIZE=group_size,
        BLOCK=BLOCK,
        BLOCK_D=BLOCK_D,
        BLOCK_DV=BLOCK_DV,
        num_warps=4,
    )
