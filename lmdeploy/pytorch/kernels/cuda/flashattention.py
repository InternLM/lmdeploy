# Copyright (c) OpenMMLab. All rights reserved.
import math
from functools import lru_cache

import torch
import triton
import triton.language as tl
from packaging import version
from torch import Tensor

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

TRITON_VERSION = version.parse(triton.__version__)
VERSION_300 = version.parse('3.0.0')
assert TRITON_VERSION >= version.parse('2.2.0')

# TODO: fast op might not work on non-nv device
if TRITON_VERSION >= VERSION_300:
    tanh = tl.extra.cuda.libdevice.tanh
    fast_dividef = tl.extra.cuda.libdevice.fast_dividef
else:
    tanh = tl.math.tanh
    fast_dividef = tl.math.fast_dividef


@lru_cache
def _get_nv_cap():
    """return nv cap."""
    return torch.cuda.get_device_capability()


def _get_block_d(head_dim_k, head_dim_v):
    """get block d."""
    BLOCK_DK = triton.next_power_of_2(head_dim_k)
    BLOCK_DK1 = 0
    if BLOCK_DK != head_dim_k:
        BLOCK_DK = BLOCK_DK // 2
        BLOCK_DK1 = max(16, triton.next_power_of_2(head_dim_k - BLOCK_DK))
    BLOCK_DV = triton.next_power_of_2(head_dim_v)
    return BLOCK_DK, BLOCK_DK1, BLOCK_DV


@triton.jit
def _prefill_fwd_inner(acc, l_i, m_i, q, k_ptrs, v_ptrs, q1, k1_ptrs,
                       loop_start, loop_end, qk_scale, history_mask,
                       kv_min_loc, causal_mask: tl.constexpr,
                       window_size: tl.constexpr,
                       logit_softcapping: tl.constexpr, BLOCK_N: tl.constexpr,
                       BLOCK_DK1: tl.constexpr):
    offs_n = tl.arange(0, BLOCK_N)
    for start_n in range(loop_start, loop_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        k = tl.load(k_ptrs)
        qk = tl.dot(q, k)

        if BLOCK_DK1 != 0:
            k1 = tl.load(k1_ptrs)
            qk += tl.dot(q1, k1)

        qk *= qk_scale
        if logit_softcapping > 0.0:
            qk = qk / logit_softcapping
            qk = tanh(qk)
            qk = qk * logit_softcapping

        if causal_mask:
            qk_mask = (history_mask[:, None]) >= (start_n + offs_n[None, :])
            if window_size > 0:
                qk_mask = qk_mask and (
                    (start_n + offs_n[None, :]) >= kv_min_loc[:, None])
            qk = tl.where(
                qk_mask,
                qk,
                float(-1e30),
            )
        else:
            if window_size > 0:
                qk_mask = ((start_n + offs_n[None, :]) >= kv_min_loc[:, None])
                qk = tl.where(
                    qk_mask,
                    qk,
                    float(-1e30),
                )

        # -- compute p, m_i and l_i
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp2(qk - m_i_new[:, None])
        alpha = tl.exp2(m_i - m_i_new)
        l_i_new = alpha * l_i + tl.sum(p, 1)
        # -- update output accumulator --
        # scale acc
        acc = acc * alpha[:, None]

        # update acc
        v = tl.load(v_ptrs)
        p = tl.cast(p, v.dtype)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

        k_ptrs = tl.advance(k_ptrs, (0, BLOCK_N))
        v_ptrs = tl.advance(v_ptrs, (BLOCK_N, 0))
        if BLOCK_DK1:
            k1_ptrs = tl.advance(k1_ptrs, (0, BLOCK_N))

    return acc, l_i, m_i


@triton.jit
def _flash_prefill_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    q_start_loc_ptr,
    q_seqlens_ptr,
    kv_start_loc_ptr,
    kv_seqlens_ptr,
    sm_scale: tl.constexpr,
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
    kv_group_num: tl.constexpr,
    window_size: tl.constexpr,
    head_dim_k: tl.constexpr,
    head_dim_v: tl.constexpr,
    logit_softcapping: tl.constexpr,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DK: tl.constexpr,
    BLOCK_DK1: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    """flash attention kernel."""
    batch_id = tl.program_id(2)
    kv_head_id = tl.program_id(1)
    start_m = tl.program_id(0)

    q_seqlen = tl.load(q_seqlens_ptr + batch_id)

    if BLOCK_M * start_m >= q_seqlen * kv_group_num:
        return

    q_start_loc = tl.load(q_start_loc_ptr + batch_id)
    kv_start_loc = tl.load(kv_start_loc_ptr + batch_id)
    kv_seqlen = tl.load(kv_seqlens_ptr + batch_id)

    history_len = kv_seqlen - q_seqlen

    offs_dk = tl.arange(0, BLOCK_DK)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_dk = offs_dk < head_dim_k
    offs_mh = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m = offs_mh // kv_group_num
    offs_h = offs_mh % kv_group_num + kv_head_id * kv_group_num
    offs_q = ((q_start_loc + offs_m[:, None]) * stride_qs +
              offs_h[:, None] * stride_qh + offs_dk[None, :] * stride_qd)

    loop_start = 0
    kv_min_loc = offs_m
    if window_size > 0:
        start_block_id = tl.maximum(history_len - window_size, 0) // BLOCK_N
        kv_min_loc = tl.maximum(history_len + offs_m - window_size, 0)
        loop_start = start_block_id * BLOCK_N
        kv_start_loc += loop_start

    q = tl.load(
        q_ptr + offs_q,
        mask=(offs_m[:, None] < q_seqlen) & mask_dk[None, :],
        other=0.0,
        eviction_policy='evict_first',
    )
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + kv_start_loc * stride_ks + kv_head_id * stride_kh,
        shape=(head_dim_k, N_CTX),
        strides=(stride_kd, stride_ks),
        offsets=(0, 0),
        block_shape=(BLOCK_DK, BLOCK_N),
        order=(0, 1),
    )
    v_ptrs = tl.make_block_ptr(
        base=v_ptr + kv_start_loc * stride_vs + kv_head_id * stride_vh,
        shape=(N_CTX, head_dim_v),
        strides=(stride_vs, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DV),
        order=(1, 0),
    )

    if BLOCK_DK1 != 0:
        offs_dk1 = BLOCK_DK + tl.arange(0, BLOCK_DK1)
        mask_dk1 = offs_dk1 < head_dim_k
        offs_q1 = ((q_start_loc + offs_m[:, None]) * stride_qs +
                   offs_h[:, None] * stride_qh + offs_dk1[None, :] * stride_qd)
        q1 = tl.load(
            q_ptr + offs_q1,
            mask=(offs_m[:, None] < q_seqlen) & mask_dk1[None, :],
            other=0.0,
            eviction_policy='evict_first',
        )
        k1_ptrs = tl.make_block_ptr(
            base=k_ptr + kv_start_loc * stride_ks + kv_head_id * stride_kh,
            shape=(head_dim_k, N_CTX),
            strides=(stride_kd, stride_ks),
            offsets=(BLOCK_DK, 0),
            block_shape=(BLOCK_DK1, BLOCK_N),
            order=(0, 1),
        )
    else:
        q1 = tl.zeros([1], dtype=q.dtype)
        k1_ptrs = k_ptrs

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)

    qk_scale = sm_scale * tl.log2(math.e)
    history_mask = history_len + offs_m

    loop_end = history_len // BLOCK_N * BLOCK_N
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
                                       qk_scale,
                                       history_mask,
                                       kv_min_loc,
                                       causal_mask=False,
                                       window_size=window_size,
                                       logit_softcapping=logit_softcapping,
                                       BLOCK_N=BLOCK_N,
                                       BLOCK_DK1=BLOCK_DK1)

    loop_start = loop_end
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
                                       qk_scale,
                                       history_mask,
                                       kv_min_loc,
                                       causal_mask=True,
                                       window_size=window_size,
                                       logit_softcapping=logit_softcapping,
                                       BLOCK_N=BLOCK_N,
                                       BLOCK_DK1=BLOCK_DK1)

    acc = fast_dividef(acc, l_i[:, None])
    # initialize pointers to output
    off_o = ((q_start_loc + offs_m[:, None]) * stride_os +
             offs_h[:, None] * stride_oh + offs_dv[None, :] * stride_od)
    out_ptrs = o_ptr + off_o
    mask_dv = offs_dv < head_dim_v
    tl.store(out_ptrs,
             acc,
             mask=(offs_m[:, None] < q_seqlen) & mask_dv[None, :])


def flash_attention_fwd(
    q_states: Tensor,
    k_states: Tensor,
    v_states: Tensor,
    o_states: Tensor,
    q_start_loc: Tensor,
    q_seqlens: Tensor,
    kv_start_loc: Tensor,
    kv_seqlens: Tensor,
    max_seqlen: int = None,
    window_size: int = None,
    sm_scale: float = None,
    logit_softcapping: float = None,
    kv_layout: str = 'hsd',
):
    """varlen flash Attention forward.

    Support sliding window, softcapping.
    """
    _nv_cap = _get_nv_cap()

    if kv_layout == 'shd':
        s_dim, h_dim, d_dim = (0, 1, 2)
    elif kv_layout == 'hsd':
        s_dim, h_dim, d_dim = (1, 0, 2)
    else:
        raise RuntimeError('Unsupported layout.')

    if max_seqlen is None:
        max_seqlen = q_states.size(0)

    if window_size is None:
        window_size = -1

    if logit_softcapping is None:
        logit_softcapping = -1.0

    head_dim_q = q_states.size(-1)
    head_dim_k = k_states.size(d_dim)
    head_dim_v = v_states.size(d_dim)
    N_CTX = k_states.size(s_dim)
    assert head_dim_q == head_dim_k and head_dim_v == o_states.size(-1)

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim_q**0.5)

    batch, num_heads = q_seqlens.size(0), q_states.size(-2)
    num_kv_heads = k_states.size(h_dim)
    kv_group_num = num_heads // num_kv_heads

    BLOCK_DK, BLOCK_DK1, BLOCK_DV = _get_block_d(head_dim_k, head_dim_v)

    if _nv_cap[0] < 8:
        BLOCK_M = max(16, min(64, 8192 // BLOCK_DK))
    else:
        BLOCK_M = max(16, min(64, 16384 // BLOCK_DK))
    BLOCK_N = 64
    num_warps = 4
    num_stages = 2
    grid = (triton.cdiv(max_seqlen * kv_group_num,
                        BLOCK_M), num_kv_heads, batch)
    _flash_prefill_fwd_kernel[grid](
        q_states,
        k_states,
        v_states,
        o_states,
        q_start_loc,
        q_seqlens,
        kv_start_loc,
        kv_seqlens,
        sm_scale=sm_scale,
        stride_qs=q_states.stride(0),
        stride_qh=q_states.stride(1),
        stride_qd=q_states.stride(2),
        stride_ks=k_states.stride(s_dim),
        stride_kh=k_states.stride(h_dim),
        stride_kd=k_states.stride(d_dim),
        stride_vs=v_states.stride(s_dim),
        stride_vh=v_states.stride(h_dim),
        stride_vd=v_states.stride(d_dim),
        stride_os=o_states.stride(0),
        stride_oh=o_states.stride(1),
        stride_od=o_states.stride(2),
        kv_group_num=kv_group_num,
        window_size=window_size,
        head_dim_k=head_dim_k,
        head_dim_v=head_dim_v,
        logit_softcapping=logit_softcapping,
        N_CTX=N_CTX,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DK=BLOCK_DK,
        BLOCK_DK1=BLOCK_DK1,
        BLOCK_DV=BLOCK_DV,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return o_states
