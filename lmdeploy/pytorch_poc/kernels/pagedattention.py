# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/ModelTC/lightllm
import torch
import triton
import triton.language as tl
from torch import Tensor

from lmdeploy.pytorch_poc.dist_utils import try_to_local

assert triton.__version__ >= '2.1.0'


@triton.jit
def _fwd_split_kernel(
    Q,
    K,
    V,
    sm_scale,
    B_kvlen,
    Block_offsets,
    Acc_out,
    Meta_out,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_ok,
    stride_obs,
    stride_oh,
    stride_od,
    stride_mk,
    stride_mbs,
    stride_mh,
    stride_md,
    stride_boffb,
    kv_group_num,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """first step kernel of split k attention."""
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_k_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = 1
    cur_batch_kv_len = tl.load(B_kvlen + cur_batch)
    history_len = cur_batch_kv_len - cur_batch_seq_len

    start_m = 0
    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = ((cur_batch + offs_m[:, None]) * stride_qbs +
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

    num_blocks = (cur_batch_kv_len + BLOCK_N - 1) // BLOCK_N
    num_blocks_per_prog = (num_blocks + SPLIT_K - 1) // SPLIT_K
    kv_len_per_prog = num_blocks_per_prog * BLOCK_N
    loop_start = kv_len_per_prog * split_k_id
    loop_end = tl.minimum(loop_start + kv_len_per_prog, cur_batch_kv_len)

    for start_n in range(loop_start, block_mask * loop_end, BLOCK_N):
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
        # NOTE: inf - inf = nan, and nan will leads to error
        qk = tl.where(
            (history_len + offs_m[:, None]) >= (start_n + offs_n[None, :]),
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
    off_acc = ((cur_batch + offs_m[:, None]) * stride_obs +
               split_k_id * stride_ok + cur_head * stride_oh +
               offs_d[None, :] * stride_od)
    tl.store(Acc_out + off_acc, acc, mask=offs_m[:, None] < cur_batch_seq_len)

    off_meta = ((cur_batch + offs_m) * stride_mbs + split_k_id * stride_mk +
                cur_head * stride_mh)
    tl.store(Meta_out + off_meta, m_i, mask=offs_m < cur_batch_seq_len)
    tl.store(Meta_out + off_meta + stride_md,
             l_i,
             mask=offs_m < cur_batch_seq_len)


@triton.jit
def _reduce_split_kernel(
    Acc,
    Meta,
    Out,
    stride_ak,
    stride_abs,
    stride_ah,
    stride_ad,
    stride_mk,
    stride_mbs,
    stride_mh,
    stride_md,
    stride_obs,
    stride_oh,
    stride_od,
    SPLIT_K: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """second step kernel of split k attention."""
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    # initialize offsets
    offs_d = tl.arange(0, BLOCK_DMODEL)

    offs_acc = (cur_batch * stride_abs + cur_head * stride_ah +
                offs_d * stride_ad)
    offs_mi = cur_batch * stride_mbs + cur_head * stride_mh

    m = tl.load(Meta + offs_mi)
    l_sum = tl.load(Meta + offs_mi + stride_md)
    acc = tl.load(Acc + offs_acc)
    for k_id in range(1, SPLIT_K):
        acc_k = tl.load(Acc + offs_acc + k_id * stride_ak)
        m_k = tl.load(Meta + offs_mi + k_id * stride_mk)
        l_k = tl.load(Meta + offs_mi + k_id * stride_mk + stride_md)

        m_new = tl.maximum(m, m_k)
        if m_k < m:
            # Scale incoming values
            alpha = tl.exp(m_k - m_new)
            acc_k = acc_k * alpha
            l_k = l_k * alpha
        else:
            # Scale our values
            alpha = tl.exp(m - m_new)
            acc = acc * alpha
            l_sum = l_sum * alpha

        m = m_new
        l_sum += l_k
        acc += acc_k

    acc = acc / l_sum
    out_offs = (cur_batch * stride_obs + cur_head * stride_oh +
                offs_d * stride_od)
    tl.store(Out + out_offs, acc)


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
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
    kv_group_num,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """paged attention kernel."""
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_kv_len = tl.load(B_kvlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    history_len = cur_batch_kv_len - cur_batch_seq_len

    block_start_loc = BLOCK_M * start_m

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
        # NOTE: inf - inf = nan, and nan will leads to error
        qk = tl.where(
            (history_len + offs_m[:, None]) >= (start_n + offs_n[None, :]),
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

    acc = acc / l_i[:, None]
    # initialize pointers to output
    off_o = ((cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs +
             cur_head * stride_oh + offs_d[None, :] * stride_od)
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)


@torch.no_grad()
def paged_attention_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    block_offsets: Tensor,
    b_start_loc: Tensor,
    b_seq_len: Tensor,
    b_kv_seq_len: Tensor,
    max_input_len: int,
    BLOCK: int = 64,
):
    """Paged Attention forward.

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
        BLOCK (int): The kernel block size.
    """
    q = try_to_local(q)
    k = try_to_local(k)
    v = try_to_local(v)
    o = try_to_local(o)
    block_offsets = try_to_local(block_offsets)
    b_start_loc = try_to_local(b_start_loc)
    b_seq_len = try_to_local(b_seq_len)
    b_kv_seq_len = try_to_local(b_kv_seq_len)

    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}

    sm_scale = 1.0 / (Lq**0.5)  # 计算scale系数
    batch, head = b_seq_len.shape[0], q.shape[-2]
    kv_group_num = q.shape[-2] // k[0].shape[-2]

    num_warps = 4 if Lk <= 64 else 8

    # is_decoding = q.shape[-3] == b_seq_len.size(0)
    is_decoding = False  # split k implementation is slower...
    if not is_decoding:
        grid = (batch, head, triton.cdiv(max_input_len, BLOCK))  # batch, head,
        _fwd_kernel[grid](
            q,
            k,
            v,
            sm_scale,
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
            kv_group_num=kv_group_num,
            BLOCK_M=BLOCK,
            BLOCK_DMODEL=Lk,
            BLOCK_N=BLOCK,
            num_warps=num_warps,
            num_stages=1,
        )
    else:
        SPLIT_K = 4
        grid = (batch, head, SPLIT_K)
        acc = q.new_empty(batch, head, SPLIT_K, Lq, dtype=torch.float32)
        meta = acc.new_empty(batch, head, SPLIT_K, 2)
        _fwd_split_kernel[grid](
            q,
            k,
            v,
            sm_scale,
            b_kv_seq_len,
            block_offsets,
            acc,
            meta,
            stride_qbs=q.stride(-3),
            stride_qh=q.stride(-2),
            stride_qd=q.stride(-1),
            stride_kbs=k.stride(-3),
            stride_kh=k.stride(-2),
            stride_kd=k.stride(-1),
            stride_vbs=v.stride(-3),
            stride_vh=v.stride(-2),
            stride_vd=v.stride(-1),
            stride_ok=acc.stride(-2),
            stride_obs=acc.stride(-4),
            stride_oh=acc.stride(-3),
            stride_od=acc.stride(-1),
            stride_mk=meta.stride(-2),
            stride_mbs=meta.stride(-4),
            stride_mh=meta.stride(-3),
            stride_md=meta.stride(-1),
            stride_boffb=block_offsets.stride(0),
            kv_group_num=kv_group_num,
            SPLIT_K=SPLIT_K,
            BLOCK_M=16,
            BLOCK_DMODEL=Lk,
            BLOCK_N=BLOCK,
            num_warps=num_warps,
            num_stages=1,
        )
        grid = (batch, head)
        _reduce_split_kernel[grid](
            acc,
            meta,
            o,
            stride_ak=acc.stride(-2),
            stride_abs=acc.stride(-4),
            stride_ah=acc.stride(-3),
            stride_ad=acc.stride(-1),
            stride_mk=meta.stride(-2),
            stride_mbs=meta.stride(-4),
            stride_mh=meta.stride(-3),
            stride_md=meta.stride(-1),
            stride_obs=o.stride(-3),
            stride_oh=o.stride(-2),
            stride_od=o.stride(-1),
            SPLIT_K=SPLIT_K,
            BLOCK_DMODEL=Lk,
            num_warps=num_warps,
            num_stages=1,
        )
