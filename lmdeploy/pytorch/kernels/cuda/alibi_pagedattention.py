# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/ModelTC/lightllm
import math
from typing import Literal

import torch
import triton
import triton.language as tl
from torch import Tensor

assert triton.__version__ >= '2.1.0'

LOG2 = tl.constexpr(math.log(2))


@triton.jit
def tl_pow(a, b):
    """Triton pow."""
    return tl.exp(b * tl.log(a))


@triton.jit
def tl_2pow(b):
    """Triton pow2."""
    return tl.exp(b * LOG2)


@triton.jit
def tl_log2(a):
    """Triton log2."""
    return tl.log(a) / LOG2


@triton.jit
def _get_interleave_power_of_2(i, n):
    """Get interleave power of 2."""
    start = -tl_2pow(3 - tl_log2(n))
    start = tl_2pow(start)
    ratio = start
    return start * tl_pow(ratio, i)


@triton.jit
def get_slope(i, n):
    """Get slope."""
    closest_power_of_2 = tl_2pow(tl_log2(n).to(tl.int32))
    if i < closest_power_of_2:
        return _get_interleave_power_of_2(i, closest_power_of_2)
    else:
        return _get_interleave_power_of_2((i - closest_power_of_2) * 2, 2 * closest_power_of_2)


@triton.jit
def _load_block_offsets(offset_ptr, block_id, BLOCK: tl.constexpr):
    offs_n = tl.arange(0, BLOCK)
    return tl.load(offset_ptr + block_id) * BLOCK + offs_n


@triton.jit
def _fwd_split_kernel(
    Q,
    K,
    V,
    sm_scale,
    alibi_scale,
    B_kvlen,
    Block_offsets,
    Acc_out,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kp,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vp,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_ok,
    stride_obs,
    stride_oh,
    stride_od,
    stride_boffb,
    head_offset,
    num_heads,
    kv_group_num,
    block_per_cta,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """First step kernel of split k attention."""
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_k_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = 1
    cur_batch_kv_len = tl.load(B_kvlen + cur_batch)
    history_len = cur_batch_kv_len - cur_batch_seq_len

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = (cur_batch * stride_qbs + cur_head * stride_qh + offs_d * stride_qd)
    off_k = (cur_kv_head * stride_kh + offs_d[None, :] * stride_kd)
    off_v = (cur_kv_head * stride_vh + offs_d[None, :] * stride_vd)

    q = tl.load(Q + off_q).to(tl.float32)

    k_ptrs = K + off_k + offs_n[:, None] * stride_kbs
    v_ptrs = V + off_v + offs_n[:, None] * stride_vbs

    block_offset_ptrs = Block_offsets + cur_batch * stride_boffb
    head_slope = get_slope(cur_head.to(tl.float32) + head_offset, num_heads.to(tl.float32))

    # initialize pointer to m and l
    m_i = -float('inf')
    l_i = float(0)
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    kv_len_per_prog = block_per_cta * BLOCK_N
    loop_start = kv_len_per_prog * split_k_id
    loop_end = tl.minimum(loop_start + kv_len_per_prog, cur_batch_kv_len)

    # load block offset
    start_block_id = loop_start // BLOCK_N
    block_offset_ptrs += start_block_id

    for start_n in range(loop_start, loop_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        b_off = tl.load(block_offset_ptrs)
        block_offset_ptrs += 1

        mask = (start_n + offs_n[:, None]) < cur_batch_kv_len

        # -- compute qk ----
        k = tl.load(
            k_ptrs + b_off * stride_kp,
            mask=mask,
            other=0.0,
        )

        v = tl.load(
            v_ptrs + b_off * stride_vp,
            mask=mask,
            other=0.0,
        )

        qk = tl.sum(q[None, :] * k, 1)
        qk *= sm_scale

        mask = start_n + offs_n
        bias = mask.to(tl.float32) * (head_slope * alibi_scale)
        qk += bias

        # NOTE: inf - inf = nan, and nan will leads to error
        qk = tl.where(
            history_len >= (start_n + offs_n),
            qk,
            -float('inf'),
        )

        # -- compute p, m_i and l_i
        m_i_new = tl.maximum(m_i, tl.max(qk, 0))
        p = tl.exp(qk - m_i_new)
        alpha = tl.exp(m_i - m_i_new)
        l_i_new = alpha * l_i + tl.sum(p, 0)

        # -- update output accumulator --
        # scale acc
        acc = acc * alpha

        # update acc
        p_new = p.to(v.dtype)
        acc += tl.sum(p_new[:, None] * v, 0)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    # initialize pointers to output
    off_acc = (cur_batch * stride_obs + split_k_id * stride_ok + cur_head * stride_oh + offs_d * stride_od)
    tl.store(Acc_out + off_acc, acc)

    off_meta = (cur_batch * stride_obs + split_k_id * stride_ok + cur_head * stride_oh + BLOCK_DMODEL)
    tl.store(Acc_out + off_meta + tl.arange(0, 1), m_i)
    tl.store(Acc_out + off_meta + 1 + tl.arange(0, 1), l_i)


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
    SPLIT_K: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """Second step kernel of split k attention."""
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    # initialize offsets
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_k = tl.arange(0, SPLIT_K)

    offs_acc = (cur_batch * stride_abs + cur_head * stride_ah + offs_k[:, None] * stride_ak +
                offs_d[None, :] * stride_ad)
    offs_mi = (cur_batch * stride_abs + cur_head * stride_ah + stride_ak * offs_k + BLOCK_DMODEL)

    acc_k = tl.load(Acc + offs_acc)
    m_k = tl.load(Acc + offs_mi)
    l_k = tl.load(Acc + offs_mi + 1)

    m_max = tl.max(m_k, 0)
    alpha = tl.exp(m_k - m_max)
    acc_k = acc_k * alpha[:, None]
    l_k = l_k * alpha

    acc = tl.sum(acc_k, 0)
    l_sum = tl.sum(l_k, 0)
    acc = acc / l_sum

    out_offs = (cur_batch * stride_obs + cur_head * stride_oh + offs_d * stride_od)
    tl.store(Out + out_offs, acc)


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    alibi_scale,
    B_Start_Loc,
    B_Seqlen,
    B_kvlen,
    Block_offsets,
    Out,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kp,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vp,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_obs,
    stride_oh,
    stride_od,
    stride_boffb,
    head_offset,
    num_heads,
    kv_group_num,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Forward kernel."""
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_kv_len = tl.load(B_kvlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    history_len = cur_batch_kv_len - cur_batch_seq_len

    block_start_loc = BLOCK_M * start_m
    head_slope = get_slope(cur_head.to(tl.float32) + head_offset, num_heads.to(tl.float32))

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = ((cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs + cur_head * stride_qh +
             offs_d[None, :] * stride_qd)
    off_k = (cur_kv_head * stride_kh + offs_d[:, None] * stride_kd)
    off_v = (cur_kv_head * stride_vh + offs_d[None, :] * stride_vd)

    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    k_ptrs = K + off_k + offs_n[None, :] * stride_kbs
    v_ptrs = V + off_v + offs_n[:, None] * stride_vbs

    block_offset_ptrs = Block_offsets + cur_batch * stride_boffb

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)

    # b_offset = _load_block_offsets(block_offset_ptrs, 0, BLOCK_N)
    for start_n in range(0, block_mask * cur_batch_kv_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        b_off = tl.load(block_offset_ptrs)
        block_offset_ptrs += 1

        # -- compute qk ----
        k = tl.load(
            k_ptrs + b_off * stride_kp,
            mask=(start_n + offs_n[None, :]) < cur_batch_kv_len,
            other=0.0,
        )

        v = tl.load(
            v_ptrs + b_off * stride_vp,
            mask=(start_n + offs_n[:, None]) < cur_batch_kv_len,
            other=0.0,
        )
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale

        mask = start_n + offs_n[None, :]
        bias = mask.to(tl.float32) * (head_slope * alibi_scale)
        qk += bias

        # NOTE: inf - inf = nan, and nan will leads to error
        qk = tl.where(
            (history_len + offs_m[:, None]) >= mask,
            qk,
            float(-1e30),
        )

        # -- compute p, m_i and l_i
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_i_new[:, None])
        alpha = tl.exp(m_i - m_i_new)
        l_i_new = alpha * l_i + tl.sum(p, 1)
        # -- update output accumulator --
        # scale acc
        acc = acc * alpha[:, None]

        # update acc
        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    acc = acc / l_i[:, None]
    # initialize pointers to output
    off_o = ((cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs + cur_head * stride_oh +
             offs_d[None, :] * stride_od)
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)


@triton.jit
def _fwd_split_kernel_quant(
    Q,
    K,
    V,
    KScalesZeros,
    VScalesZeros,
    sm_scale,
    alibi_scale,
    B_kvlen,
    Block_offsets,
    Acc_out,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kp,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vp,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_ksp: tl.constexpr,
    stride_kszbs: tl.constexpr,
    stride_kszh: tl.constexpr,
    stride_kszd: tl.constexpr,
    stride_vsp: tl.constexpr,
    stride_vszbs: tl.constexpr,
    stride_vszh: tl.constexpr,
    stride_vszd: tl.constexpr,
    quant_policy: tl.constexpr,
    stride_ok,
    stride_obs,
    stride_oh,
    stride_od,
    stride_boffb,
    head_offset,
    num_heads,
    kv_group_num,
    block_per_cta,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """First step kernel of split k attention with dequant fused.

    Args:
        stride_xbs: stride of block size dim
        stride_h: stride of head num dim
        stride_d: stride of head size dim
    """
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_k_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = 1
    cur_batch_kv_len = tl.load(B_kvlen + cur_batch)
    history_len = cur_batch_kv_len - cur_batch_seq_len

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dsz = tl.arange(0, 1)
    off_q = (cur_batch * stride_qbs + cur_head * stride_qh + offs_d * stride_qd)
    if quant_policy == 4:
        shift_d = offs_d // (BLOCK_DMODEL // 2) * 4
        off_k = (cur_kv_head * stride_kh + (offs_d % (BLOCK_DMODEL // 2))[None, :] * stride_kd)
        off_v = (cur_kv_head * stride_vh + (offs_d % (BLOCK_DMODEL // 2))[None, :] * stride_vd)
    else:
        off_k = (cur_kv_head * stride_kh + offs_d[None, :] * stride_kd)
        off_v = (cur_kv_head * stride_vh + offs_d[None, :] * stride_vd)
    off_ksz = (cur_kv_head * stride_kszh + offs_dsz[None, :] * stride_kszd)
    off_vsz = (cur_kv_head * stride_vszh + offs_dsz[None, :] * stride_vszd)

    q = tl.load(Q + off_q).to(tl.float32)

    k_ptrs = K + off_k + offs_n[:, None] * stride_kbs
    v_ptrs = V + off_v + offs_n[:, None] * stride_vbs
    ksz_ptrs = KScalesZeros + off_ksz + offs_n[:, None] * stride_kszbs
    vsz_ptrs = VScalesZeros + off_vsz + offs_n[:, None] * stride_vszbs

    block_offset_ptrs = Block_offsets + cur_batch * stride_boffb
    head_slope = get_slope(cur_head.to(tl.float32) + head_offset, num_heads.to(tl.float32))

    # initialize pointer to m and l
    m_i = -float('inf')
    l_i = float(0)
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    kv_len_per_prog = block_per_cta * BLOCK_N
    loop_start = kv_len_per_prog * split_k_id
    loop_end = tl.minimum(loop_start + kv_len_per_prog, cur_batch_kv_len)

    for start_n in range(loop_start, loop_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        b_off = tl.load(block_offset_ptrs)
        block_offset_ptrs += 1

        mask = (start_n + offs_n[:, None]) < cur_batch_kv_len

        # -- compute qk ----
        k = tl.load(
            k_ptrs + b_off * stride_kp,
            mask=mask,
            other=0.0,
        )
        if quant_policy == 4:
            k = (k >> shift_d) & 0x0F
        ks = tl.load(
            ksz_ptrs + b_off * stride_ksp,
            mask=mask,
            other=0.0,
        )
        kz = tl.load(
            ksz_ptrs + b_off * stride_ksp + 1,
            mask=mask,
            other=0.0,
        )

        v = tl.load(
            v_ptrs + b_off * stride_vp,
            mask=mask,
            other=0.0,
        )
        if quant_policy == 4:
            v = (v >> shift_d) & 0x0F
        vs = tl.load(
            vsz_ptrs + b_off * stride_vsp,
            mask=mask,
            other=0.0,
        )
        vz = tl.load(
            vsz_ptrs + b_off * stride_vsp + 1,
            mask=mask,
            other=0.0,
        )

        k = (k - kz) * ks
        v = (v - vz) * vs

        qk = tl.sum(q[None, :] * k, 1)
        qk *= sm_scale

        mask = start_n + offs_n
        bias = mask.to(tl.float32) * (head_slope * alibi_scale)
        qk += bias

        # NOTE: inf - inf = nan, and nan will leads to error
        qk = tl.where(
            history_len >= (start_n + offs_n),
            qk,
            -float('inf'),
        )

        # -- compute p, m_i and l_i
        m_i_new = tl.maximum(m_i, tl.max(qk, 0))
        p = tl.exp(qk - m_i_new)
        alpha = tl.exp(m_i - m_i_new)
        l_i_new = alpha * l_i + tl.sum(p, 0)

        # -- update output accumulator --
        # scale acc
        acc = acc * alpha

        # update acc
        p_new = p.to(v.dtype)
        acc += tl.sum(p_new[:, None] * v, 0)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    # initialize pointers to output
    off_acc = (cur_batch * stride_obs + split_k_id * stride_ok + cur_head * stride_oh + offs_d * stride_od)
    tl.store(Acc_out + off_acc, acc)

    off_meta = (cur_batch * stride_obs + split_k_id * stride_ok + cur_head * stride_oh + BLOCK_DMODEL)
    tl.store(Acc_out + off_meta + tl.arange(0, 1), m_i)
    tl.store(Acc_out + off_meta + 1 + tl.arange(0, 1), l_i)


@triton.jit
def _fwd_kernel_quant(
    Q,
    K,
    V,
    KScalesZeros,
    VScalesZeros,
    sm_scale,
    alibi_scale,
    B_Start_Loc,
    B_Seqlen,
    B_kvlen,
    Block_offsets,
    Out,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kp,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vp,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_ksp: tl.constexpr,
    stride_kszbs: tl.constexpr,
    stride_kszh: tl.constexpr,
    stride_kszd: tl.constexpr,
    stride_vsp: tl.constexpr,
    stride_vszbs: tl.constexpr,
    stride_vszh: tl.constexpr,
    stride_vszd: tl.constexpr,
    quant_policy: tl.constexpr,
    stride_obs,
    stride_oh,
    stride_od,
    stride_boffb,
    head_offset,
    num_heads,
    kv_group_num,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Forward kernel with dequant fused.

    Args:
        stride_xbs: stride of block size dim
        stride_h: stride of head num dim
        stride_d: stride of head size dim
    """
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_kv_len = tl.load(B_kvlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    history_len = cur_batch_kv_len - cur_batch_seq_len

    block_start_loc = BLOCK_M * start_m
    head_slope = get_slope(cur_head.to(tl.float32) + head_offset, num_heads.to(tl.float32))

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dsz = tl.arange(0, 1)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = ((cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs + cur_head * stride_qh +
             offs_d[None, :] * stride_qd)
    if quant_policy == 4:
        shift_kd = (offs_d // (BLOCK_DMODEL // 2) * 4)[:, None]
        shift_vd = (offs_d // (BLOCK_DMODEL // 2) * 4)[None, :]
        off_k = (cur_kv_head * stride_kh + (offs_d % (BLOCK_DMODEL // 2))[:, None] * stride_kd)
        off_v = (cur_kv_head * stride_vh + (offs_d % (BLOCK_DMODEL // 2))[None, :] * stride_vd)
    else:
        off_k = (cur_kv_head * stride_kh + offs_d[:, None] * stride_kd)
        off_v = (cur_kv_head * stride_vh + offs_d[None, :] * stride_vd)
    off_ksz = (cur_kv_head * stride_kszh + offs_dsz[:, None] * stride_kszd)
    off_vsz = (cur_kv_head * stride_vszh + offs_dsz[None, :] * stride_vszd)

    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    k_ptrs = K + off_k + offs_n[None, :] * stride_kbs
    v_ptrs = V + off_v + offs_n[:, None] * stride_vbs
    ksz_ptrs = KScalesZeros + off_ksz + offs_n[None, :] * stride_kszbs
    vsz_ptrs = VScalesZeros + off_vsz + offs_n[:, None] * stride_vszbs

    block_offset_ptrs = Block_offsets + cur_batch * stride_boffb

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)

    for start_n in range(0, block_mask * cur_batch_kv_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        b_off = tl.load(block_offset_ptrs)
        block_offset_ptrs += 1

        # -- compute qk ----
        k = tl.load(
            k_ptrs + b_off * stride_kp,
            mask=(start_n + offs_n[None, :]) < cur_batch_kv_len,
            other=0.0,
        )
        if quant_policy == 4:
            k = (k >> shift_kd) & 0x0F
        ks = tl.load(
            ksz_ptrs + b_off * stride_ksp,
            mask=(start_n + offs_n[None, :]) < cur_batch_kv_len,
            other=0.0,
        )
        kz = tl.load(
            ksz_ptrs + b_off * stride_ksp + 1,
            mask=(start_n + offs_n[None, :]) < cur_batch_kv_len,
            other=0.0,
        )

        v = tl.load(
            v_ptrs + b_off * stride_vp,
            mask=(start_n + offs_n[:, None]) < cur_batch_kv_len,
            other=0.0,
        )
        if quant_policy == 4:
            v = (v >> shift_vd) & 0x0F
        vs = tl.load(
            vsz_ptrs + b_off * stride_vsp,
            mask=(start_n + offs_n[:, None]) < cur_batch_kv_len,
            other=0.0,
        )
        vz = tl.load(
            vsz_ptrs + b_off * stride_vsp + 1,
            mask=(start_n + offs_n[:, None]) < cur_batch_kv_len,
            other=0.0,
        )

        v = ((v - vz) * vs).to(q.dtype)
        k = ((k - kz) * ks).to(q.dtype)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale

        mask = start_n + offs_n[None, :]
        bias = mask.to(tl.float32) * (head_slope * alibi_scale)
        qk += bias

        # NOTE: inf - inf = nan, and nan will leads to error
        qk = tl.where(
            (history_len + offs_m[:, None]) >= mask,
            qk,
            float(-1e30),
        )

        # -- compute p, m_i and l_i
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_i_new[:, None])
        alpha = tl.exp(m_i - m_i_new)
        l_i_new = alpha * l_i + tl.sum(p, 1)
        # -- update output accumulator --
        # scale acc
        acc = acc * alpha[:, None]

        # update acc
        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    acc = acc / l_i[:, None]
    # initialize pointers to output
    off_o = ((cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs + cur_head * stride_oh +
             offs_d[None, :] * stride_od)
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)


def alibi_paged_attention_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    block_offsets: Tensor,
    b_start_loc: Tensor,
    b_seq_len: Tensor,
    b_kv_seq_len: Tensor,
    max_input_len: int,
    head_offset: int = 0,
    num_heads: int = -1,
    alibi_scale: float = 1.0,
    k_scales_zeros: Tensor = None,
    v_scales_zeros: Tensor = None,
    quant_policy: Literal[0, 4, 8] = 0,
):
    """Paged attention forward with alibi bias.

    Args:
        q (Tensor): Query state.
        k (Tensor): Key state caches.
        v (Tensor): Value state caches.
        o (Tensor): Output state.
        block_offsets (Tensor): The block offset of key and value.
        b_start_loc (Tensor): Start token location of each data in batch.
        b_seq_len (Tensor): Query length for each data in batch.
        b_kv_seq_len (Tensor): Key/Value length for each data in batch.
        max_input_len (int): The max input length.
        head_offset (int): The offset of the start head. Head might be
            partitioned when tensor parallel inference.
        num_heads (int): The number of heads. Head might be partitioned when
            tensor parallel inference.
        BLOCK (int): The kernel block size.
    """

    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    if quant_policy == 4:
        assert Lq == Lk * 2 and Lk == Lv
        assert Lk in {8, 16, 32, 64}
    else:
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
    assert q.dim() == 3
    assert k.dim() == 4
    assert v.dim() == 4
    if k_scales_zeros is not None:
        assert k_scales_zeros.dim() == 4
    if v_scales_zeros is not None:
        assert v_scales_zeros.dim() == 4

    sm_scale = 1.0 / (Lq**0.5)  # 计算scale系数
    batch, head = b_seq_len.shape[0], q.shape[-2]
    kv_group_num = q.shape[-2] // k[0].shape[-2]
    if num_heads <= 0:
        num_heads = head

    BLOCK = k.size(1)

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))  # batch, head,

    num_warps = 4 if Lq <= 64 else 8
    is_decoding = q.shape[-3] == b_seq_len.size(0)
    if not is_decoding:
        if quant_policy > 0:
            _fwd_kernel_quant[grid](q,
                                    k,
                                    v,
                                    k_scales_zeros,
                                    v_scales_zeros,
                                    sm_scale,
                                    alibi_scale,
                                    b_start_loc,
                                    b_seq_len,
                                    b_kv_seq_len,
                                    block_offsets,
                                    o,
                                    *q.stride(),
                                    *k.stride(),
                                    *v.stride(),
                                    *k_scales_zeros.stride(),
                                    *v_scales_zeros.stride(),
                                    quant_policy,
                                    o.stride(-3),
                                    o.stride(-2),
                                    o.stride(-1),
                                    block_offsets.stride(0),
                                    head_offset=head_offset,
                                    num_heads=num_heads,
                                    kv_group_num=kv_group_num,
                                    BLOCK_M=BLOCK,
                                    BLOCK_DMODEL=Lq,
                                    BLOCK_N=BLOCK,
                                    num_warps=num_warps,
                                    num_stages=1)
        else:
            _fwd_kernel[grid](q,
                              k,
                              v,
                              sm_scale,
                              alibi_scale,
                              b_start_loc,
                              b_seq_len,
                              b_kv_seq_len,
                              block_offsets,
                              o,
                              *q.stride(),
                              *k.stride(),
                              *v.stride(),
                              o.stride(-3),
                              o.stride(-2),
                              o.stride(-1),
                              block_offsets.stride(0),
                              head_offset=head_offset,
                              num_heads=num_heads,
                              kv_group_num=kv_group_num,
                              BLOCK_M=BLOCK,
                              BLOCK_DMODEL=Lq,
                              BLOCK_N=BLOCK,
                              num_warps=num_warps,
                              num_stages=1)
    else:
        SPLIT_K = 4
        grid = (batch, head, SPLIT_K)
        block_per_cta = triton.cdiv(block_offsets.size(-1), SPLIT_K)
        acc = q.new_empty(batch, head, SPLIT_K, Lq + 2, dtype=torch.float32)
        if quant_policy > 0:
            _fwd_split_kernel_quant[grid](q,
                                          k,
                                          v,
                                          k_scales_zeros,
                                          v_scales_zeros,
                                          sm_scale,
                                          alibi_scale,
                                          b_kv_seq_len,
                                          block_offsets,
                                          acc,
                                          *q.stride(),
                                          *k.stride(),
                                          *v.stride(),
                                          *k_scales_zeros.stride(),
                                          *v_scales_zeros.stride(),
                                          quant_policy=quant_policy,
                                          stride_ok=acc.stride(-2),
                                          stride_obs=acc.stride(-4),
                                          stride_oh=acc.stride(-3),
                                          stride_od=acc.stride(-1),
                                          stride_boffb=block_offsets.stride(0),
                                          head_offset=head_offset,
                                          num_heads=num_heads,
                                          kv_group_num=kv_group_num,
                                          block_per_cta=block_per_cta,
                                          BLOCK_DMODEL=Lq,
                                          BLOCK_N=BLOCK,
                                          num_warps=4,
                                          num_stages=1)

        else:
            _fwd_split_kernel[grid](q,
                                    k,
                                    v,
                                    sm_scale,
                                    alibi_scale,
                                    b_kv_seq_len,
                                    block_offsets,
                                    acc,
                                    *q.stride(),
                                    *k.stride(),
                                    *v.stride(),
                                    stride_ok=acc.stride(-2),
                                    stride_obs=acc.stride(-4),
                                    stride_oh=acc.stride(-3),
                                    stride_od=acc.stride(-1),
                                    stride_boffb=block_offsets.stride(0),
                                    head_offset=head_offset,
                                    num_heads=num_heads,
                                    kv_group_num=kv_group_num,
                                    block_per_cta=block_per_cta,
                                    BLOCK_DMODEL=Lq,
                                    BLOCK_N=BLOCK,
                                    num_warps=4,
                                    num_stages=1)

        grid = (batch, head)
        _reduce_split_kernel[grid](acc,
                                   o,
                                   stride_ak=acc.stride(-2),
                                   stride_abs=acc.stride(-4),
                                   stride_ah=acc.stride(-3),
                                   stride_ad=acc.stride(-1),
                                   stride_obs=o.stride(-3),
                                   stride_oh=o.stride(-2),
                                   stride_od=o.stride(-1),
                                   SPLIT_K=SPLIT_K,
                                   BLOCK_DMODEL=Lq,
                                   num_warps=num_warps,
                                   num_stages=1)
