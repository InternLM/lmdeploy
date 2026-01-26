# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Sequence

import torch
import triton
import triton.language as tl
from packaging import version
from torch import Tensor

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

TRITON_VERSION = version.parse(triton.__version__)
VERSION_300 = version.parse('3.0.0')
VERSION_320 = version.parse('3.2.0')
assert TRITON_VERSION >= VERSION_300

# TODO: fast op might not work on non-nv device
tanh = tl.extra.cuda.libdevice.tanh
tl_log2 = tl.log2
tl_exp2 = tl.exp2


def _get_block_d(head_dim_k, head_dim_v):
    """Get block d."""
    BLOCK_DK = triton.next_power_of_2(head_dim_k)
    BLOCK_DK1 = 0
    if BLOCK_DK != head_dim_k:
        BLOCK_DK = BLOCK_DK // 2
        BLOCK_DK1 = max(16, triton.next_power_of_2(head_dim_k - BLOCK_DK))
    BLOCK_DV = triton.next_power_of_2(head_dim_v)
    return BLOCK_DK, BLOCK_DK1, BLOCK_DV


@triton.jit
def softcapping(qk, logit_softcapping: tl.constexpr):
    """Soft capping."""
    if logit_softcapping > 0.0:
        qk = qk / logit_softcapping
        qk = tanh(qk)
        qk = qk * logit_softcapping
    return qk


@triton.jit
def _load_kv(ptrs, boundary_check: tl.constexpr):
    """Load kv."""
    if boundary_check is not None:
        return tl.load(ptrs, boundary_check=boundary_check, padding_option='zero')
    else:
        return tl.load(ptrs)


@triton.jit
def _prefill_fwd_inner(acc, l_i, m_i, q, k_ptrs, v_ptrs, q1, k1_ptrs, loop_start, loop_end, sm_scale, alibi_slope,
                       global_offs_m, history_mask, kv_min_loc, causal_mask: tl.constexpr, window_size: tl.constexpr,
                       logit_softcapping: tl.constexpr, k_bound: tl.constexpr, v_bound: tl.constexpr,
                       shared_kv: tl.constexpr, block_sparse_size: tl.constexpr, BLOCK_N: tl.constexpr,
                       BLOCK_DK1: tl.constexpr):
    k_ptrs = tl.advance(k_ptrs, (0, loop_start))
    v_ptrs = tl.advance(v_ptrs, (loop_start, 0))
    if BLOCK_DK1:
        k1_ptrs = tl.advance(k1_ptrs, (0, loop_start))

    offs_n = tl.arange(0, BLOCK_N)
    for start_n in range(loop_start, loop_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        k = _load_kv(k_ptrs, boundary_check=k_bound)
        qk = tl.dot(q, k)

        if BLOCK_DK1 != 0:
            k1 = _load_kv(k1_ptrs, boundary_check=k_bound)
            qk += tl.dot(q1, k1)

        if causal_mask:
            qk *= sm_scale
            qk = softcapping(qk, logit_softcapping)
            qk = qk * tl_log2(math.e)
            if block_sparse_size > 1:
                offs_mask = (start_n + offs_n) // block_sparse_size * block_sparse_size
                qk_mask = (history_mask[:, None]) >= offs_mask[None, :]
            else:
                qk_mask = (history_mask[:, None]) >= (start_n + offs_n[None, :])
            if window_size > 0:
                qk_mask = qk_mask & ((start_n + offs_n[None, :]) >= kv_min_loc[:, None])
            qk = tl.where(
                qk_mask,
                qk,
                float(-1e30),
            )
            m_i_new = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_i_new[:, None]
        elif window_size > 0:
            qk *= sm_scale
            qk = softcapping(qk, logit_softcapping)
            qk = qk * tl_log2(math.e)
            qk_mask = ((start_n + offs_n[None, :]) >= kv_min_loc[:, None])
            qk = tl.where(
                qk_mask,
                qk,
                float(-1e30),
            )
            m_i_new = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_i_new[:, None]
        elif logit_softcapping > 0:
            qk *= sm_scale
            qk = softcapping(qk, logit_softcapping)
            qk = qk * tl_log2(math.e)
            m_i_new = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_i_new[:, None]
        else:
            qk_scale = sm_scale * tl_log2(math.e)
            m_i_new = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_i_new[:, None]

        if alibi_slope is not None:
            relative_pos = start_n + offs_n[None, :] - global_offs_m[:, None]
            bias = -tl.abs(relative_pos).to(tl.float32) * alibi_slope * tl_log2(math.e)
            qk += bias

        # -- compute p, m_i and l_i
        p = tl_exp2(qk)
        alpha = tl_exp2(m_i - m_i_new)
        l_i = alpha * l_i + tl.sum(p, 1)
        # -- update output accumulator --
        # scale acc
        acc = acc * alpha[:, None]

        # update acc
        if shared_kv:
            v = tl.trans(k)
        else:
            v = _load_kv(v_ptrs, boundary_check=v_bound)
        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        # update m_i and l_i
        m_i = m_i_new

        k_ptrs = tl.advance(k_ptrs, (0, BLOCK_N))
        v_ptrs = tl.advance(v_ptrs, (BLOCK_N, 0))
        if BLOCK_DK1:
            k1_ptrs = tl.advance(k1_ptrs, (0, BLOCK_N))

    return acc, l_i, m_i


# # FOR DEBUG, DON'T REMOVE
# import itertools
# configs = [
#     triton.Config({
#         'BLOCK_M': BM,
#         'BLOCK_N': BN
#     }, num_stages=s, num_warps=w)
#     for BM, BN, s, w in itertools.product([64, 128], [32, 64], [3, 4], [4])
# ]


# @triton.autotune(list(configs),
#                  key=['head_dim_k', 'head_dim_v'])
@triton.jit
def _flash_prefill_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    q_start_loc_ptr,
    q_seqlens_ptr,
    kv_start_loc_ptr,
    kv_seqlens_ptr,
    sinks,
    alibi_slopes_ptr,
    sm_scale,
    stride_qs: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kh,
    stride_kd: tl.constexpr,
    stride_vs: tl.constexpr,
    stride_vh,
    stride_vd: tl.constexpr,
    stride_os: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_od: tl.constexpr,
    kv_group_num,
    head_dim_k: tl.constexpr,
    head_dim_v: tl.constexpr,
    causal: tl.constexpr,
    window_size: tl.constexpr,
    logit_softcapping: tl.constexpr,
    shared_kv: tl.constexpr,
    block_sparse_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DK: tl.constexpr,
    BLOCK_DK1: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    """Flash attention kernel."""
    start_m = tl.program_id(0)
    head_id = tl.program_id(1)
    batch_id = tl.program_id(2)

    if cu_seqlens_q_ptr is not None:
        q_start_loc = tl.load(cu_seqlens_q_ptr + batch_id).to(tl.int32)
        q_seqlen = tl.load(cu_seqlens_q_ptr + batch_id + 1).to(tl.int32) - q_start_loc
    else:
        q_start_loc = tl.load(q_start_loc_ptr + batch_id).to(tl.int32)
        q_seqlen = tl.load(q_seqlens_ptr + batch_id).to(tl.int32)

    if cu_seqlens_k_ptr is not None:
        kv_start_loc = tl.load(cu_seqlens_k_ptr + batch_id).to(tl.int32)
        kv_seqlen = tl.load(cu_seqlens_k_ptr + batch_id + 1).to(tl.int32) - kv_start_loc
    else:
        kv_start_loc = tl.load(kv_start_loc_ptr + batch_id).to(tl.int32)
        kv_seqlen = tl.load(kv_seqlens_ptr + batch_id).to(tl.int32)

    if BLOCK_M * start_m >= q_seqlen:
        return

    kv_head_id = head_id // kv_group_num
    history_len = kv_seqlen - q_seqlen

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    loop_start = 0
    kv_min_loc = tl.zeros([BLOCK_M], dtype=tl.int32)
    if window_size > 0:
        start_block_id = tl.maximum(history_len + start_m * BLOCK_M - window_size, 0) // BLOCK_N
        kv_min_loc = tl.maximum(history_len + offs_m - window_size, 0)
        loop_start = start_block_id * BLOCK_N

    offs_dk = tl.arange(0, BLOCK_DK)
    mask_dk = offs_dk < head_dim_k
    offs_dk = tl.multiple_of(tl.max_contiguous(offs_dk % head_dim_k, BLOCK_DK), BLOCK_DK)
    off_q = ((q_start_loc + offs_m[:, None]) * stride_qs + head_id * stride_qh + offs_dk[None, :] * stride_qd)
    q_ptrs = q_ptr + off_q
    q = tl.load(q_ptrs, mask=((offs_m[:, None] < q_seqlen) & mask_dk[None, :]))

    k_ptrs = tl.make_block_ptr(
        base=k_ptr + kv_start_loc * stride_ks + kv_head_id * stride_kh,
        shape=(head_dim_k, kv_seqlen),
        strides=(stride_kd, stride_ks),
        offsets=(0, 0),
        block_shape=(BLOCK_DK, BLOCK_N),
        order=(0, 1),
    )
    v_ptrs = tl.make_block_ptr(
        base=v_ptr + kv_start_loc * stride_vs + kv_head_id * stride_vh,
        shape=(kv_seqlen, head_dim_v),
        strides=(stride_vs, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DV),
        order=(1, 0),
    )

    # for alibi
    if alibi_slopes_ptr is not None:
        alibi_slope = tl.load(alibi_slopes_ptr + head_id)
    else:
        alibi_slope = None
    global_offs_m = history_len + offs_m

    if BLOCK_DK + BLOCK_DK1 == head_dim_k:
        k_bound0: tl.constexpr = None
        k_bound1: tl.constexpr = (1, )
    else:
        k_bound0: tl.constexpr = (1, )
        k_bound1: tl.constexpr = (0, 1)
    if head_dim_v == BLOCK_DV:
        v_bound0: tl.constexpr = None
        v_bound1: tl.constexpr = (0, )
    else:
        v_bound0: tl.constexpr = (1, )
        v_bound1: tl.constexpr = (0, 1)

    if BLOCK_DK1 != 0:
        offs_dk1 = BLOCK_DK + tl.arange(0, BLOCK_DK1)
        mask_dk1 = offs_dk1 < head_dim_k
        offs_dk1 = tl.multiple_of(tl.max_contiguous(offs_dk1 % head_dim_k, BLOCK_DK1), BLOCK_DK1)
        offs_q1 = ((q_start_loc + offs_m[:, None]) * stride_qs + head_id * stride_qh + offs_dk1[None, :] * stride_qd)
        q1_ptrs = q_ptr + offs_q1
        q1 = tl.load(q1_ptrs, mask=((offs_m[:, None] < q_seqlen) & mask_dk1[None, :]))
        k1_ptrs = tl.make_block_ptr(
            base=k_ptr + kv_start_loc * stride_ks + kv_head_id * stride_kh,
            shape=(head_dim_k, kv_seqlen),
            strides=(stride_kd, stride_ks),
            offsets=(BLOCK_DK, 0),
            block_shape=(BLOCK_DK1, BLOCK_N),
            order=(0, 1),
        )
    else:
        q1 = q
        k1_ptrs = k_ptrs

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)

    if causal:
        history_mask = history_len + start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        loop_end = (history_len + start_m * BLOCK_M) // BLOCK_N * BLOCK_N
    else:
        history_mask = tl.full([BLOCK_M], kv_seqlen - 1, dtype=tl.int32)
        loop_end = kv_seqlen // BLOCK_N * BLOCK_N

    acc, l_i, m_i = _prefill_fwd_inner(acc,
                                       l_i,
                                       m_i,
                                       q,
                                       k_ptrs,
                                       v_ptrs,
                                       q1,
                                       k1_ptrs,
                                       loop_start,
                                       loop_end,
                                       sm_scale,
                                       alibi_slope,
                                       global_offs_m,
                                       history_mask,
                                       kv_min_loc,
                                       causal_mask=False,
                                       window_size=window_size,
                                       logit_softcapping=logit_softcapping,
                                       k_bound=k_bound0,
                                       v_bound=v_bound0,
                                       shared_kv=shared_kv,
                                       block_sparse_size=block_sparse_size,
                                       BLOCK_N=BLOCK_N,
                                       BLOCK_DK1=BLOCK_DK1)

    loop_start = loop_end
    if causal:
        loop_end = tl.minimum(kv_seqlen, loop_start + BLOCK_M + BLOCK_N)
    else:
        loop_end = kv_seqlen
    acc, l_i, m_i = _prefill_fwd_inner(acc,
                                       l_i,
                                       m_i,
                                       q,
                                       k_ptrs,
                                       v_ptrs,
                                       q1,
                                       k1_ptrs,
                                       loop_start,
                                       loop_end,
                                       sm_scale,
                                       alibi_slope,
                                       global_offs_m,
                                       history_mask,
                                       kv_min_loc,
                                       causal_mask=True,
                                       window_size=window_size,
                                       logit_softcapping=logit_softcapping,
                                       k_bound=k_bound1,
                                       v_bound=v_bound1,
                                       shared_kv=shared_kv,
                                       block_sparse_size=block_sparse_size,
                                       BLOCK_N=BLOCK_N,
                                       BLOCK_DK1=BLOCK_DK1)
    # epilogue
    if sinks is not None:
        sink = tl.load(sinks + head_id).to(l_i.dtype)
        l_i = l_i + tl.exp2(sink * tl_log2(math.e) - m_i)

    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]

    # initialize pointers to output
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_dv = offs_dv < head_dim_v
    off_o = ((q_start_loc + offs_m[:, None]) * stride_os + head_id * stride_oh + offs_dv[None, :] * stride_od)
    out_ptrs = o_ptr + off_o
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < q_seqlen) & mask_dv[None, :])


_nv_cap = None


def _kernel_meta_sm7x(BLOCK_DK):
    num_warps = 4
    num_stages = min(4, max(2, 768 // BLOCK_DK))
    BLOCK_M = max(16, 8192 // BLOCK_DK)
    BLOCK_N = 32
    return BLOCK_M, BLOCK_N, num_warps, num_stages


def _kernel_meta_sm8x(BLOCK_DK: int, shared_kv: bool):
    num_warps = 8
    min_m = 64 if shared_kv else 16
    BLOCK_M = max(min_m, 16384 // BLOCK_DK)
    BLOCK_M = min(128, BLOCK_M)
    BLOCK_N = BLOCK_M
    num_stages = 3 if BLOCK_DK <= 128 else 2

    return BLOCK_M, BLOCK_N, num_warps, num_stages


def _kernel_meta_sm86(BLOCK_DK: int, shared_kv: bool):
    """Sm86 has different smem size with sm80."""
    num_warps = 4
    if BLOCK_DK <= 128:
        BLOCK_M = 128
        BLOCK_N = 64
        num_stages = 3
    elif BLOCK_DK <= 256:
        BLOCK_M = 64
        BLOCK_N = 32
        num_stages = 2
    else:
        BLOCK_M = 32
        BLOCK_N = 32
        num_stages = 2

    return BLOCK_M, BLOCK_N, num_warps, num_stages


def _kernel_meta_sm9x(BLOCK_DK: int, shared_kv: bool):

    num_warps = 8
    BLOCK_M = 128 if BLOCK_DK <= 256 else 64
    if not shared_kv and BLOCK_DK >= 512:
        BLOCK_M = 32

    # fix crash on triton<3.2.0
    if BLOCK_DK >= 512 and TRITON_VERSION < VERSION_320:
        BLOCK_M = 32
        num_warps = 4

    BLOCK_N = 128 if BLOCK_DK <= 128 else 64

    num_stages = 3 if BLOCK_DK <= 128 else 2
    return BLOCK_M, BLOCK_N, num_warps, num_stages


def _kernel_meta_sm12x(BLOCK_DK: int, shared_kv: bool):
    # Blackwell (sm_120, cc 12.x) + B200/B100 variants
    if BLOCK_DK <= 128:
        BLOCK_M = 128
        BLOCK_N = 128 if shared_kv else 64
        num_warps = 8
        num_stages = 3
    elif BLOCK_DK <= 256:
        BLOCK_M = 64
        BLOCK_N = 128 if shared_kv else 64
        num_warps = 8
        num_stages = 3
    elif BLOCK_DK <= 512:
        BLOCK_M = 64 if shared_kv else 32
        BLOCK_N = 64
        num_warps = 4
        num_stages = 2
    else:
        BLOCK_M = 32
        BLOCK_N = 32 if not shared_kv else 64
        num_warps = 4
        num_stages = 2

    return BLOCK_M, BLOCK_N, num_warps, num_stages


def _kernel_meta_rocm(BLOCK_DK: int, shared_kv: bool):
    BLOCK_N = 32
    BLOCK_M = 32 if BLOCK_DK > 128 else 64
    num_warps = 4
    num_stages = 1
    return BLOCK_M, BLOCK_N, num_warps, num_stages


def flash_attn_varlen_func(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor = None,
    cu_seqlens_k: Tensor = None,
    max_seqlen_q: int = None,
    max_seqlen_k: int = None,  # not used, just for align with fa interface
    softmax_scale: float = None,
    causal: bool = False,
    window_size: int = (-1, -1),
    softcap: float = 0.0,
    # old seqlens
    q_start_loc: Tensor = None,
    q_seqlens: Tensor = None,
    kv_start_loc: Tensor = None,
    kv_seqlens: Tensor = None,
    # args not in fa
    alibi_slopes: Tensor = None,
    sinks: Tensor = None,
    block_sparse_size: int = 1,
    kv_layout: str = 'hsd',
):
    """Varlen flash Attention forward.

    Support sliding window, softcapping.
    """

    global _nv_cap
    if _nv_cap is None:
        _nv_cap = torch.cuda.get_device_capability()

    def grid(args):
        return (triton.cdiv(max_seqlen_q, args['BLOCK_M']), num_heads, batch)

    if kv_layout == 'shd':
        s_dim, h_dim, d_dim = (0, 1, 2)
    elif kv_layout == 'hsd':
        s_dim, h_dim, d_dim = (1, 0, 2)
    else:
        raise RuntimeError('Unsupported layout.')

    if max_seqlen_q is None:
        max_seqlen_q = q.size(0)

    if window_size is None:
        window_size = -1
    elif isinstance(window_size, Sequence):
        window_size = window_size[0]

    if softcap is None:
        softcap = -1.0

    head_dim_q = q.size(-1)
    head_dim_k = k.size(d_dim)
    head_dim_v = v.size(d_dim)

    o = q.new_empty(*q.size()[:-1], head_dim_v)
    assert head_dim_q == head_dim_k and head_dim_v == o.size(-1)

    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim_q**0.5)

    if cu_seqlens_k is None:
        assert kv_start_loc is not None and kv_seqlens is not None
    if cu_seqlens_q is None:
        assert q_start_loc is not None and q_seqlens is not None
        batch = q_seqlens.size(0)
    else:
        batch = cu_seqlens_q.size(0) - 1
    num_heads = q.size(-2)
    num_kv_heads = k.size(h_dim)
    kv_group_num = num_heads // num_kv_heads

    if sinks is not None:
        assert sinks.is_contiguous()
        assert sinks.numel() == num_heads

    BLOCK_DK, BLOCK_DK1, BLOCK_DV = _get_block_d(head_dim_k, head_dim_v)

    shared_kv = k.data_ptr() == v.data_ptr() and BLOCK_DK == BLOCK_DV

    num_warps = 4
    hip_mode = getattr(torch.version, 'hip', None) is not None
    if hip_mode:
        BLOCK_M, BLOCK_N, num_warps, num_stages = _kernel_meta_rocm(BLOCK_DK, shared_kv)
    else:
        if _nv_cap[0] < 8:
            BLOCK_M, BLOCK_N, num_warps, num_stages = _kernel_meta_sm7x(BLOCK_DK)
        elif _nv_cap[0] < 9:
            if _nv_cap[1] in [6, 9]:
                BLOCK_M, BLOCK_N, num_warps, num_stages = _kernel_meta_sm86(BLOCK_DK, shared_kv)
            else:
                BLOCK_M, BLOCK_N, num_warps, num_stages = _kernel_meta_sm8x(BLOCK_DK, shared_kv)
        elif _nv_cap[0] < 10:
            BLOCK_M, BLOCK_N, num_warps, num_stages = _kernel_meta_sm9x(BLOCK_DK, shared_kv)
        else:
            BLOCK_M, BLOCK_N, num_warps, num_stages = _kernel_meta_sm12x(BLOCK_DK, shared_kv)

    BLOCK_M = min(128, BLOCK_M)
    _flash_prefill_fwd_kernel[grid](
        q,
        k,
        v,
        o,
        cu_seqlens_q,
        cu_seqlens_k,
        q_start_loc,
        q_seqlens,
        kv_start_loc,
        kv_seqlens,
        sinks,
        alibi_slopes,
        sm_scale=softmax_scale,
        stride_qs=q.stride(0),
        stride_qh=q.stride(1),
        stride_qd=q.stride(2),
        stride_ks=k.stride(s_dim),
        stride_kh=k.stride(h_dim),
        stride_kd=k.stride(d_dim),
        stride_vs=v.stride(s_dim),
        stride_vh=v.stride(h_dim),
        stride_vd=v.stride(d_dim),
        stride_os=o.stride(0),
        stride_oh=o.stride(1),
        stride_od=o.stride(2),
        kv_group_num=kv_group_num,
        head_dim_k=head_dim_k,
        head_dim_v=head_dim_v,
        causal=causal,
        window_size=window_size,
        logit_softcapping=softcap,
        shared_kv=shared_kv,
        block_sparse_size=block_sparse_size,
        BLOCK_DK=BLOCK_DK,
        BLOCK_DK1=BLOCK_DK1,
        BLOCK_DV=BLOCK_DV,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return o
