# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/ModelTC/lightllm
import math
from typing import Literal

import torch
import triton
import triton.language as tl
from packaging import version
from torch import Tensor

from lmdeploy.utils import get_logger

from .triton_utils import get_kernel_meta, wrap_jit_func

logger = get_logger('lmdeploy')

TRITON_VERSION = version.parse(triton.__version__)
VERSION_300 = version.parse('3.0.0')

assert TRITON_VERSION >= version.parse('2.2.0')

# TODO: fast op might not work on non-nv device
if TRITON_VERSION >= VERSION_300:
    tanh = tl.extra.cuda.libdevice.tanh
    fast_dividef = tl.extra.cuda.libdevice.fast_dividef
    tl_log2 = tl.log2
    tl_exp2 = tl.exp2
else:
    tanh = tl.math.tanh
    fast_dividef = tl.math.fast_dividef
    tl_log2 = tl.math.log2
    tl_exp2 = tl.math.exp2


@triton.autotune(configs=[
    triton.Config({}, num_stages=2, num_warps=16),
    triton.Config({}, num_stages=2, num_warps=8),
    triton.Config({}, num_stages=2, num_warps=4),
],
                 key=['BLOCK_H', 'BLOCK_N', 'BLOCK_DMODEL', 'BLOCK_DV'],
                 warmup=10,
                 rep=25)
@wrap_jit_func(type_hint=dict(
    Q=torch.Tensor,
    K=torch.Tensor,
    V=torch.Tensor,
    sm_scale=float,
    KV_seqlens=torch.Tensor,
    Block_offsets=torch.Tensor,
    Acc_out=torch.Tensor,
    stride_qbs=int,
    stride_qh=int,
    stride_qd=int,
    stride_kbs=int,
    stride_kh=int,
    stride_kd=int,
    stride_vbs=int,
    stride_vh=int,
    stride_vd=int,
    stride_ok=int,
    stride_obs=int,
    stride_oh=int,
    stride_od=int,
    stride_boffb=int,
    kv_group_num=torch.int32,
    block_per_cta=torch.int32,
    window_size=torch.int32,
    head_size=torch.int32,
    head_size_v=torch.int32,
    BLOCK_DMODEL=torch.int32,
    BLOCK_DV=torch.int32,
    BLOCK_N=torch.int32,
    BLOCK_H=torch.int32,
    BLOCK_DMODEL1=torch.int32,
))
@triton.jit
def _fwd_grouped_split_kernel(
    Q,
    K,
    V,
    sm_scale,
    KV_seqlens,
    Block_offsets,
    Acc_out,
    stride_qbs: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kp: tl.constexpr,
    stride_kbs: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vp: tl.constexpr,
    stride_vbs: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ok: tl.constexpr,
    stride_obs: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_od: tl.constexpr,
    stride_boffb,
    kv_group_num: tl.constexpr,
    window_size: tl.constexpr,
    head_size: tl.constexpr,
    head_size_v: tl.constexpr,
    num_heads_q: tl.constexpr,
    logit_softcapping: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_DMODEL1: tl.constexpr,
):
    """first step kernel of split k attention."""
    cur_batch = tl.program_id(2)
    cur_kv_head = tl.program_id(0)
    split_k_id = tl.program_id(1)

    if BLOCK_H < kv_group_num:
        HEAD_PER_CTA: tl.constexpr = BLOCK_H
    else:
        HEAD_PER_CTA: tl.constexpr = kv_group_num
    cur_head = cur_kv_head * HEAD_PER_CTA + tl.arange(0, BLOCK_H)
    mask_h = cur_head < cur_kv_head * HEAD_PER_CTA + HEAD_PER_CTA
    mask_h = mask_h & (cur_head < num_heads_q)
    if BLOCK_H < kv_group_num:
        cur_kv_head = (cur_kv_head * HEAD_PER_CTA) // kv_group_num

    q_seqlen = 1
    kv_seqlen = tl.load(KV_seqlens + cur_batch)
    if kv_seqlen <= 0:
        return
    history_len = kv_seqlen - q_seqlen

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_d = offs_d < head_size
    offs_d = offs_d % head_size
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_dv = offs_dv < head_size_v
    offs_dv = offs_dv % head_size_v
    off_k = (cur_kv_head * stride_kh + offs_d[:, None] * stride_kd +
             offs_n[None, :] * stride_kbs)
    off_v = (cur_kv_head * stride_vh + offs_dv[None, :] * stride_vd +
             offs_n[:, None] * stride_vbs)

    off_q = (cur_batch * stride_qbs + cur_head[:, None] * stride_qh +
             offs_d[None, :] * stride_qd)
    q = tl.load(Q + off_q, mask=mask_h[:, None] & mask_d[None, :], other=0)

    k_ptrs = K + off_k
    v_ptrs = V + off_v

    if BLOCK_DMODEL1 != 0:
        offs_d1 = BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL1)
        mask_d1 = offs_d1 < head_size
        offs_d1 = offs_d1 % head_size
        off_q1 = (cur_batch * stride_qbs + cur_head[:, None] * stride_qh +
                  offs_d1[None, :] * stride_qd)
        q1 = tl.load(Q + off_q1,
                     mask=mask_h[:, None] & mask_d1[None, :],
                     other=0)
        off_k1 = (cur_kv_head * stride_kh + offs_d1[:, None] * stride_kd +
                  offs_n[None, :] * stride_kbs)
        k1_ptrs = K + off_k1

    block_offset_ptrs = Block_offsets + cur_batch * stride_boffb

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_H], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    num_total_blocks = tl.cdiv(kv_seqlen, BLOCK_N)
    BLOCK_PER_CTA = tl.cdiv(num_total_blocks, SPLIT_K)
    kv_len_per_prog = BLOCK_PER_CTA * BLOCK_N
    loop_start = kv_len_per_prog * split_k_id
    loop_end = tl.minimum(loop_start + kv_len_per_prog, kv_seqlen)

    # load block offset
    # dirty
    start_block_id = loop_start // BLOCK_N
    if window_size > 0:
        start_block_id = tl.maximum(history_len - window_size,
                                    loop_start) // BLOCK_N
        kv_min_loc = tl.maximum(history_len - window_size, 0)

    loop_start = start_block_id * BLOCK_N
    block_offset_ptrs += start_block_id
    for start_n in range(loop_start, loop_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        b_offset = tl.load(block_offset_ptrs)
        block_offset_ptrs += 1

        # -- compute qk ----
        k = tl.load(k_ptrs + b_offset * stride_kp)
        if BLOCK_DMODEL1 != 0:
            k1 = tl.load(k1_ptrs + b_offset * stride_kp)

        v = tl.load(v_ptrs + b_offset * stride_vp)

        qk = tl.zeros([BLOCK_H, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        if BLOCK_DMODEL1 != 0:
            qk += tl.dot(q1, k1)
        qk *= sm_scale * tl_log2(math.e)
        if logit_softcapping > 0.0:
            qk = qk / logit_softcapping
            qk = tanh(qk)
            qk = qk * logit_softcapping
        # NOTE: inf - inf = nan, and nan will leads to error
        if start_n + BLOCK_N > history_len or window_size > 0:
            qk_mask = history_len >= (start_n + offs_n)
            if window_size > 0:
                qk_mask = qk_mask and ((start_n + offs_n) >= kv_min_loc)
            qk = tl.where(
                qk_mask[None, :],
                qk,
                -float('inf'),
            )

        # -- compute p, m_i and l_i
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl_exp2(qk - m_i_new[:, None])
        alpha = tl_exp2(m_i - m_i_new)
        l_i_new = alpha * l_i + tl.sum(p, 1)

        # -- update output accumulator --
        # scale acc
        acc = acc * alpha[:, None]

        # update acc
        p, v = _convert_pv(p, v)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    # initialize pointers to output
    if loop_end > loop_start:
        off_acc = (cur_batch * stride_obs + split_k_id * stride_ok +
                   cur_head[:, None] * stride_oh +
                   offs_dv[None, :] * stride_od)
        tl.store(Acc_out + off_acc,
                 acc,
                 mask=mask_h[:, None] & mask_dv[None, :])

    off_meta = (cur_batch * stride_obs + split_k_id * stride_ok +
                cur_head * stride_oh + head_size_v)
    tl.store(Acc_out + off_meta, m_i, mask=mask_h)
    tl.store(Acc_out + off_meta + 1, l_i, mask=mask_h)


@triton.autotune(configs=[
    triton.Config({}, num_stages=2, num_warps=16),
    triton.Config({}, num_stages=2, num_warps=8),
    triton.Config({}, num_stages=2, num_warps=4),
],
                 key=['BLOCK_H', 'BLOCK_N', 'BLOCK_DMODEL', 'BLOCK_DV'])
@wrap_jit_func(type_hint=dict(
    Q=torch.Tensor,
    K=torch.Tensor,
    V=torch.Tensor,
    KScalesZeros=torch.Tensor,
    VScalesZeros=torch.Tensor,
    sm_scale=float,
    KV_seqlens=torch.Tensor,
    Block_offsets=torch.Tensor,
    Acc_out=torch.Tensor,
    stride_qbs=int,
    stride_qh=int,
    stride_qd=int,
    stride_kbs=int,
    stride_kh=int,
    stride_kd=int,
    stride_vbs=int,
    stride_vh=int,
    stride_vd=int,
    stride_kszp=int,
    stride_kszbs=int,
    stride_kszh=int,
    stride_kszd=int,
    stride_vszp=int,
    stride_vszbs=int,
    stride_vszh=int,
    stride_vszd=int,
    quant_policy=int,
    stride_ok=int,
    stride_obs=int,
    stride_oh=int,
    stride_od=int,
    stride_boffb=int,
    kv_group_num=torch.int32,
    block_per_cta=torch.int32,
    window_size=torch.int32,
    head_size=torch.int32,
    head_size_v=torch.int32,
    BLOCK_DMODEL=torch.int32,
    BLOCK_DV=torch.int32,
    BLOCK_N=torch.int32,
    BLOCK_H=torch.int32,
    BLOCK_DMODEL1=torch.int32,
))
@triton.jit
def _fwd_grouped_split_quant_kernel(
    Q,
    K,
    V,
    KScalesZeros,
    VScalesZeros,
    sm_scale,
    KV_seqlens,
    Block_offsets,
    Acc_out,
    stride_qbs: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kp: tl.constexpr,
    stride_kbs: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vp: tl.constexpr,
    stride_vbs: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_kszp: tl.constexpr,
    stride_kszbs: tl.constexpr,
    stride_kszh: tl.constexpr,
    stride_kszd: tl.constexpr,
    stride_vszp: tl.constexpr,
    stride_vszbs: tl.constexpr,
    stride_vszh: tl.constexpr,
    stride_vszd: tl.constexpr,
    quant_policy: tl.constexpr,
    stride_ok: tl.constexpr,
    stride_obs: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_od: tl.constexpr,
    stride_boffb,
    kv_group_num: tl.constexpr,
    window_size: tl.constexpr,
    head_size: tl.constexpr,
    head_size_v: tl.constexpr,
    num_heads_q: tl.constexpr,
    logit_softcapping: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_DMODEL1: tl.constexpr,
):
    """first step kernel of split k attention.

    Args:
        stride_xp: stride of page num dim
        stride_xbs: stride of block size dim
        stride_h: stride of head num dim
        stride_d: stride of head size dim
    """
    cur_batch = tl.program_id(2)
    cur_kv_head = tl.program_id(0)
    split_k_id = tl.program_id(1)

    if BLOCK_H < kv_group_num:
        HEAD_PER_CTA: tl.constexpr = BLOCK_H
    else:
        HEAD_PER_CTA: tl.constexpr = kv_group_num
    cur_head = cur_kv_head * HEAD_PER_CTA + tl.arange(0, BLOCK_H)
    mask_h = cur_head < cur_kv_head * HEAD_PER_CTA + HEAD_PER_CTA
    mask_h = mask_h & (cur_head < num_heads_q)
    if BLOCK_H < kv_group_num:
        cur_kv_head = (cur_kv_head * HEAD_PER_CTA) // kv_group_num

    q_seqlen = 1
    kv_seqlen = tl.load(KV_seqlens + cur_batch)
    if kv_seqlen <= 0:
        return
    history_len = kv_seqlen - q_seqlen

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dsz = tl.arange(0, 1)
    mask_d = offs_d < head_size
    offs_d = offs_d % head_size
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_dv = offs_dv < head_size_v
    offs_dv = offs_dv % head_size_v
    off_k = (cur_kv_head * stride_kh + offs_d[:, None] * stride_kd +
             offs_n[None, :] * stride_kbs)
    off_v = (cur_kv_head * stride_vh + offs_dv[None, :] * stride_vd +
             offs_n[:, None] * stride_vbs)
    off_ksz = (cur_kv_head * stride_kszh + offs_dsz[:, None] * stride_kszd +
               offs_n[None, :] * stride_kszbs)
    off_vsz = (cur_kv_head * stride_vszh + offs_dsz[None, :] * stride_vszd +
               offs_n[:, None] * stride_vszbs)

    off_q = (cur_batch * stride_qbs + cur_head[:, None] * stride_qh +
             offs_d[None, :] * stride_qd)
    q = tl.load(Q + off_q, mask=mask_h[:, None] & mask_d[None, :], other=0)

    ksz_ptrs = KScalesZeros + off_ksz
    vsz_ptrs = VScalesZeros + off_vsz

    if BLOCK_DMODEL1 != 0:
        offs_d1 = BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL1)
        mask_d1 = offs_d1 < head_size
        offs_d1 = offs_d1 % head_size
        off_q1 = (cur_batch * stride_qbs + cur_head[:, None] * stride_qh +
                  offs_d1[None, :] * stride_qd)
        q1 = tl.load(Q + off_q1,
                     mask=mask_h[:, None] & mask_d1[None, :],
                     other=0)
        off_k1 = (cur_kv_head * stride_kh + offs_d1[:, None] * stride_kd +
                  offs_n[None, :] * stride_kbs)

    block_offset_ptrs = Block_offsets + cur_batch * stride_boffb

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_H], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    if quant_policy == 4:
        if BLOCK_DMODEL1 != 0:
            offs_d1 = BLOCK_DMODEL // 2 + tl.arange(0, BLOCK_DMODEL1)
            shift_k1d = (offs_d1 // (head_size // 2) * 4)[:, None]
            offs_d1 = offs_d1 % (head_size // 2)
            off_k1 = (cur_kv_head * stride_kh + offs_d1[:, None] * stride_kd +
                      offs_n[None, :] * stride_kbs)
        offs_d = tl.arange(0, BLOCK_DMODEL) % (head_size // 2)
        shift_kd = (tl.arange(0, BLOCK_DMODEL) // (head_size // 2) * 4)[:,
                                                                        None]
        off_k = (cur_kv_head * stride_kh + offs_d[:, None] * stride_kd +
                 offs_n[None, :] * stride_kbs)
        offs_dv = tl.arange(0, BLOCK_DV * 2) % head_size_v
        shift_vd = (tl.arange(0, BLOCK_DV * 2) // head_size_v * 4)
        off_v = (cur_kv_head * stride_vh + offs_dv[None, :] * stride_vd +
                 offs_n[:, None] * stride_vbs)
        acc = tl.zeros([BLOCK_H, BLOCK_DV * 2],
                       dtype=tl.float32)  # v head_dim packed
        mask_dv = tl.arange(0, BLOCK_DV * 2) < (head_size_v * 2)
        offs_dv = tl.arange(0, BLOCK_DV * 2) % (head_size_v * 2)
    else:
        acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    num_total_blocks = tl.cdiv(kv_seqlen, BLOCK_N)
    BLOCK_PER_CTA = tl.cdiv(num_total_blocks, SPLIT_K)
    kv_len_per_prog = BLOCK_PER_CTA * BLOCK_N
    loop_start = kv_len_per_prog * split_k_id
    loop_end = tl.minimum(loop_start + kv_len_per_prog, kv_seqlen)

    # load block offset
    # dirty
    start_block_id = loop_start // BLOCK_N
    if window_size > 0:
        start_block_id = tl.maximum(history_len - window_size,
                                    loop_start) // BLOCK_N
        kv_min_loc = tl.maximum(history_len - window_size, 0)

    loop_start = start_block_id * BLOCK_N
    for start_n in range(loop_start, loop_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        b_offset = tl.load(block_offset_ptrs + start_n // BLOCK_N)

        # -- compute qk ----
        # k = tl.load(k_ptrs + b_offset * stride_kp)
        k = tl.load(K + off_k + b_offset * stride_kp)
        if quant_policy == 4:
            k = (k >> shift_kd) & 0x0F
        ks = tl.load(ksz_ptrs + b_offset * stride_kszp)
        kz = tl.load(ksz_ptrs + b_offset * stride_kszp + 1)
        if BLOCK_DMODEL1 != 0:
            k1 = tl.load(K + off_k1 + b_offset * stride_kp)
            if quant_policy == 4:
                k1 = (k1 >> shift_k1d) & 0x0F
            k1 = ((k1 - kz) * ks).to(q.dtype)

        if quant_policy == 4:
            v = tl.load(V + off_v + b_offset * stride_vp)
            v = (v >> shift_vd) & 0x0F
        else:
            v = tl.load(V + off_v + b_offset * stride_vp)
        vs = tl.load(vsz_ptrs + b_offset * stride_vszp)
        vz = tl.load(vsz_ptrs + b_offset * stride_vszp + 1)

        k = ((k - kz) * ks).to(q.dtype)
        v = ((v - vz) * vs).to(q.dtype)
        qk = tl.zeros([BLOCK_H, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        if BLOCK_DMODEL1 != 0:
            qk += tl.dot(q1, k1)
        qk *= sm_scale * tl_log2(math.e)
        if logit_softcapping > 0.0:
            qk = qk / logit_softcapping
            qk = tanh(qk)
            qk = qk * logit_softcapping
        # NOTE: inf - inf = nan, and nan will leads to error
        if start_n + BLOCK_N > history_len or window_size > 0:
            qk_mask = history_len >= (start_n + offs_n)
            if window_size > 0:
                qk_mask = qk_mask and ((start_n + offs_n) >= kv_min_loc)
            qk = tl.where(
                qk_mask[None, :],
                qk,
                -float('inf'),
            )

        # -- compute p, m_i and l_i
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl_exp2(qk - m_i_new[:, None])
        alpha = tl_exp2(m_i - m_i_new)
        l_i_new = alpha * l_i + tl.sum(p, 1)

        # -- update output accumulator --
        # scale acc
        acc = acc * alpha[:, None]

        # update acc
        p, v = _convert_pv(p, v)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    # initialize pointers to output
    if loop_end > loop_start:
        off_acc = (cur_batch * stride_obs + split_k_id * stride_ok +
                   cur_head[:, None] * stride_oh +
                   offs_dv[None, :] * stride_od)
        tl.store(Acc_out + off_acc,
                 acc,
                 mask=mask_h[:, None] & mask_dv[None, :])

    if quant_policy == 4:
        off_meta = (cur_batch * stride_obs + split_k_id * stride_ok +
                    cur_head * stride_oh + head_size_v * 2)
    else:
        off_meta = (cur_batch * stride_obs + split_k_id * stride_ok +
                    cur_head * stride_oh + head_size_v)
    tl.store(Acc_out + off_meta, m_i, mask=mask_h)
    tl.store(Acc_out + off_meta + 1, l_i, mask=mask_h)


@wrap_jit_func(type_hint=dict(
    Acc=torch.Tensor,
    Out=torch.Tensor,
    stride_ak=int,
    stride_abs=int,
    stride_ah=int,
    stride_ad=int,
    stride_obs=int,
    stride_oh=int,
    stride_od=int,
    head_size_v=torch.int32,
    SPLIT_K=torch.int32,
    BLOCK_DV=torch.int32,
))
@triton.jit
def _reduce_split_kernel(
    Acc,
    Out,
    stride_ak,
    stride_abs,
    stride_ah,
    stride_ad,
    stride_obs,
    stride_oh,
    stride_od,
    head_size_v: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    """second step kernel of split k attention."""
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    # initialize offsets
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_k = tl.arange(0, SPLIT_K)
    mask_dv = offs_dv < head_size_v

    offs_acc = (cur_batch * stride_abs + cur_head * stride_ah +
                offs_k[:, None] * stride_ak + offs_dv[None, :] * stride_ad)
    offs_mi = (cur_batch * stride_abs + cur_head * stride_ah +
               stride_ak * offs_k + head_size_v)

    m_k = tl.load(Acc + offs_mi)
    l_k = tl.load(Acc + offs_mi + 1)
    acc_k = tl.load(Acc + offs_acc,
                    mask=mask_dv[None, :] & (m_k[:, None] > -float('inf')),
                    other=0.0)

    m_max = tl.max(m_k, 0)
    alpha = tl_exp2(m_k - m_max)
    acc_k = acc_k * alpha[:, None]
    l_k = l_k * alpha

    acc = tl.sum(acc_k, 0)
    l_sum = tl.sum(l_k, 0)
    acc = acc / l_sum

    out_offs = (cur_batch * stride_obs + cur_head * stride_oh +
                offs_dv * stride_od)
    tl.store(Out + out_offs, acc, mask=mask_dv)


def _get_convert_pv(nv_capability):
    """lazy load convert_pv."""
    global TRITON_VERSION, VERSION_300
    if TRITON_VERSION >= VERSION_300 or nv_capability[0] >= 8:

        @triton.jit
        def convert_pv(p, v):
            """convert pv."""
            p = p.to(v.dtype)
            return p, v
    else:

        @triton.jit
        def convert_pv(p, v):
            """convert pv."""
            v = v.to(p.dtype)
            return p, v

    return convert_pv


_convert_pv = None
_nv_cap = None


def paged_attention_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    block_offsets: Tensor,
    kv_seqlens: Tensor,
    k_scales_zeros: Tensor = None,
    v_scales_zeros: Tensor = None,
    quant_policy: Literal[0, 4, 8] = 0,
    window_size: int = None,
    sm_scale: float = None,
    logit_softcapping: float = None,
    kv_layout: str = 'bshd',
):
    """Paged Attention forward.

    Args:
        q (Tensor): Query state.
        k (Tensor): Key state caches.
        v (Tensor): Value state caches.
        o (Tensor): Output state.
        block_offsets (Tensor): The block offset of key and value.
        q_start_loc (Tensor): Start token location of each data in batch.
        kv_seqlens (Tensor): Key/Value length for each data in batch.
        max_seqlen (int): The max input length.
        BLOCK (int): The kernel block size.
    """
    global _convert_pv, _nv_cap
    if _convert_pv is None:
        nv_cap = torch.cuda.get_device_capability()
        _nv_cap = nv_cap
        _convert_pv = _get_convert_pv(nv_cap)

    if kv_layout == 'bshd':
        b_dim, s_dim, h_dim, d_dim = (0, 1, 2, 3)
    elif kv_layout == 'bhsd':
        b_dim, s_dim, h_dim, d_dim = (0, 2, 1, 3)
    else:
        raise RuntimeError('Unsupported layout.')

    if window_size is None:
        window_size = -1

    if logit_softcapping is None:
        logit_softcapping = -1.0

    def _get_block_d(Lk):
        """get block d."""
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DMODEL1 = 0
        if BLOCK_DMODEL != Lk:
            BLOCK_DMODEL = BLOCK_DMODEL // 2
            BLOCK_DMODEL1 = max(16, triton.next_power_of_2(Lk - BLOCK_DMODEL))
        BLOCK_DV = triton.next_power_of_2(Lv)
        return BLOCK_DMODEL, BLOCK_DMODEL1, BLOCK_DV

    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[d_dim], v.shape[d_dim]
    if quant_policy == 4:
        assert Lq == Lk * 2 and Lv * 2 == o.shape[-1]
    else:
        assert Lq == Lk and Lv == o.shape[-1]

    if sm_scale is None:
        sm_scale = 1.0 / (Lq**0.5)
    batch, head = kv_seqlens.shape[0], q.shape[-2]
    kv_group_num = q.shape[-2] // k.shape[h_dim]

    BLOCK = k.size(s_dim)
    assert BLOCK >= 16
    if Lq > 512 and BLOCK > 32:
        logger.warning(f'`head_dim={Lq}` and `block_size={BLOCK}` '
                       'might leads to bad performance. '
                       'Please reduce `block_size`.')

    kernel_meta = get_kernel_meta(q)
    is_decoding = q.shape[-3] == kv_seqlens.size(0)
    assert is_decoding, 'we only support decoding paged attention.'

    SPLIT_K = 4
    if quant_policy != 4:
        acc = q.new_empty(batch, head, SPLIT_K, Lv + 2, dtype=torch.float32)
    else:
        acc = q.new_empty(batch,
                          head,
                          SPLIT_K,
                          o.shape[-1] + 2,
                          dtype=torch.float32)
    BLOCK_DMODEL, BLOCK_DMODEL1, BLOCK_DV = _get_block_d(Lq)
    p2_kv_group_num = triton.next_power_of_2(kv_group_num)
    BLOCK_H = max(16, min(BLOCK, p2_kv_group_num))
    grid_1 = triton.cdiv(head, min(BLOCK_H, kv_group_num))
    grid = (
        grid_1,
        SPLIT_K,
        batch,
    )
    if quant_policy > 0:
        _fwd_grouped_split_quant_kernel[grid](
            q,
            k,
            v,
            k_scales_zeros,
            v_scales_zeros,
            sm_scale,
            kv_seqlens,
            block_offsets,
            acc,
            stride_qbs=q.stride(-3),
            stride_qh=q.stride(-2),
            stride_qd=q.stride(-1),
            stride_kp=k.stride(b_dim),
            stride_kbs=k.stride(s_dim),
            stride_kh=k.stride(h_dim),
            stride_kd=k.stride(d_dim),
            stride_vp=v.stride(b_dim),
            stride_vbs=v.stride(s_dim),
            stride_vh=v.stride(h_dim),
            stride_vd=v.stride(d_dim),
            stride_kszp=k_scales_zeros.stride(b_dim),
            stride_kszbs=k_scales_zeros.stride(s_dim),
            stride_kszh=k_scales_zeros.stride(h_dim),
            stride_kszd=k_scales_zeros.stride(d_dim),
            stride_vszp=v_scales_zeros.stride(b_dim),
            stride_vszbs=v_scales_zeros.stride(s_dim),
            stride_vszh=v_scales_zeros.stride(h_dim),
            stride_vszd=v_scales_zeros.stride(d_dim),
            quant_policy=quant_policy,
            stride_ok=acc.stride(-2),
            stride_obs=acc.stride(-4),
            stride_oh=acc.stride(-3),
            stride_od=acc.stride(-1),
            stride_boffb=block_offsets.stride(0),
            kv_group_num=kv_group_num,
            window_size=window_size,
            head_size=Lq,
            head_size_v=Lv,
            num_heads_q=head,
            logit_softcapping=logit_softcapping,
            SPLIT_K=SPLIT_K,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_DV=BLOCK_DV,
            BLOCK_N=BLOCK,
            BLOCK_H=BLOCK_H,
            BLOCK_DMODEL1=BLOCK_DMODEL1,
            **kernel_meta)

    else:
        _fwd_grouped_split_kernel[grid](q,
                                        k,
                                        v,
                                        sm_scale,
                                        kv_seqlens,
                                        block_offsets,
                                        acc,
                                        stride_qbs=q.stride(-3),
                                        stride_qh=q.stride(-2),
                                        stride_qd=q.stride(-1),
                                        stride_kp=k.stride(b_dim),
                                        stride_kbs=k.stride(s_dim),
                                        stride_kh=k.stride(h_dim),
                                        stride_kd=k.stride(d_dim),
                                        stride_vp=v.stride(b_dim),
                                        stride_vbs=v.stride(s_dim),
                                        stride_vh=v.stride(h_dim),
                                        stride_vd=v.stride(d_dim),
                                        stride_ok=acc.stride(-2),
                                        stride_obs=acc.stride(-4),
                                        stride_oh=acc.stride(-3),
                                        stride_od=acc.stride(-1),
                                        stride_boffb=block_offsets.stride(0),
                                        kv_group_num=kv_group_num,
                                        window_size=window_size,
                                        head_size=Lk,
                                        head_size_v=Lv,
                                        num_heads_q=head,
                                        logit_softcapping=logit_softcapping,
                                        SPLIT_K=SPLIT_K,
                                        BLOCK_DMODEL=BLOCK_DMODEL,
                                        BLOCK_DV=BLOCK_DV,
                                        BLOCK_N=BLOCK,
                                        BLOCK_H=BLOCK_H,
                                        BLOCK_DMODEL1=BLOCK_DMODEL1,
                                        **kernel_meta)

    num_warps = 4
    grid = (batch, head)
    if quant_policy == 4:
        Lv *= 2
        BLOCK_DV *= 2
    _reduce_split_kernel[grid](acc,
                               o,
                               stride_ak=acc.stride(2),
                               stride_abs=acc.stride(0),
                               stride_ah=acc.stride(1),
                               stride_ad=acc.stride(3),
                               stride_obs=o.stride(0),
                               stride_oh=o.stride(1),
                               stride_od=o.stride(2),
                               SPLIT_K=SPLIT_K,
                               head_size_v=Lv,
                               BLOCK_DV=BLOCK_DV,
                               num_warps=num_warps,
                               num_stages=1,
                               **kernel_meta)
