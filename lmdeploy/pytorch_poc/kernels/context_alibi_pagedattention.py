# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/ModelTC/lightllm
import math

import torch
import triton
import triton.language as tl
from torch import Tensor

assert triton.__version__ >= '2.1.0'

LOG2 = math.log(2)


@triton.jit
def tl_pow(a, b):
    return tl.exp(b * tl.log(a))


@triton.jit
def tl_2pow(b):
    return tl.exp(b * LOG2)


@triton.jit
def tl_log2(a):
    return tl.log(a) / LOG2


@triton.jit
def _get_interleave_power_of_2(i, n):
    start = -tl_2pow(3 - tl_log2(n))
    start = tl_2pow(start)
    ratio = start
    return start * tl_pow(ratio, i)


@triton.jit
def get_slope(i, n):
    closest_power_of_2 = tl_2pow(tl_log2(n).to(tl.int32))
    if i < closest_power_of_2:
        return _get_interleave_power_of_2(i, closest_power_of_2)
    else:
        return _get_interleave_power_of_2((i - closest_power_of_2) * 2,
                                          2 * closest_power_of_2)


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
    stride_kbs,
    stride_kh,
    stride_kd,
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
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_kv_len = tl.load(B_kvlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    history_len = cur_batch_kv_len - cur_batch_seq_len

    block_start_loc = BLOCK_M * start_m
    head_slope = get_slope(
        cur_head.to(tl.float32) + head_offset, num_heads.to(tl.float32))

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = ((cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs +
             cur_head * stride_qh + offs_d[None, :] * stride_qd)
    off_k = (offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh +
             offs_d[:, None] * stride_kd)
    off_v = (offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh +
             offs_d[None, :] * stride_vd)

    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    k_ptrs = K + off_k
    v_ptrs = V + off_v

    block_offset_ptrs = Block_offsets + cur_batch * stride_boffb

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)

    for start_n in range(0, block_mask * cur_batch_kv_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        start_block_id = start_n // BLOCK_N
        b_offset = tl.load(block_offset_ptrs + start_block_id)

        # -- compute qk ----
        k = tl.load(
            k_ptrs + b_offset * BLOCK_N * stride_kbs,
            mask=(start_n + offs_n[None, :]) < cur_batch_kv_len,
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

        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        # update acc
        v = tl.load(
            v_ptrs + b_offset * BLOCK_N * stride_vbs,
            mask=(start_n + offs_n[:, None]) < cur_batch_kv_len,
            other=0.0,
        )

        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
    # initialize pointers to output
    off_o = ((cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs +
             cur_head * stride_oh + offs_d[None, :] * stride_od)
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)


@torch.no_grad()
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
    BLOCK: int = 64,
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
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}

    sm_scale = 1.0 / (Lq**0.5)  # 计算scale系数
    batch, head = b_seq_len.shape[0], q.shape[-2]
    kv_group_num = q.shape[-2] // k[0].shape[-2]
    if num_heads <= 0:
        num_heads = head

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))  # batch, head,

    num_warps = 4 if Lk <= 64 else 8
    _fwd_kernel[grid](
        q,
        k,
        v,
        sm_scale,
        alibi_scale,
        b_start_loc,
        b_seq_len,
        b_kv_seq_len,
        block_offsets,
        o,
        q.stride(-3),
        q.stride(-2),
        q.stride(-1),
        k.stride(-3),
        k.stride(-2),
        k.stride(-1),
        v.stride(-3),
        v.stride(-2),
        v.stride(-1),
        o.stride(-3),
        o.stride(-2),
        o.stride(-1),
        block_offsets.stride(0),
        head_offset=head_offset,
        num_heads=num_heads,
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=Lk,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )

    return
