# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/ModelTC/lightllm
import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import get_cuda_stream

assert triton.__version__ >= '2.1.0'


@triton.jit
def _load_block_offsets(offset_ptr, block_id, num_sub_blocks: tl.constexpr,
                        BLOCK: tl.constexpr):
    if num_sub_blocks > 1:
        offs_sub = tl.arange(0, num_sub_blocks)
        offs_n = tl.arange(0, BLOCK // num_sub_blocks)
        ret = tl.load(offset_ptr + block_id * num_sub_blocks + offs_sub)[
            None, :] * BLOCK // num_sub_blocks + offs_n[:, None]
        return tl.ravel(ret)
    else:
        offs_n = tl.arange(0, BLOCK)
        return tl.load(offset_ptr + block_id) * BLOCK + offs_n


@triton.jit
def _fwd_split_kernel(
    Q,
    K,
    V,
    sm_scale,
    B_kvlen,
    Block_offsets,
    Acc_out,
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
    stride_boffb,
    kv_group_num,
    block_per_cta,
    num_sub_blocks: tl.constexpr,
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

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = (cur_batch * stride_qbs + cur_head * stride_qh +
             offs_d * stride_qd)
    off_k = (cur_kv_head * stride_kh + offs_d[None, :] * stride_kd)
    off_v = (cur_kv_head * stride_vh + offs_d[None, :] * stride_vd)

    q = tl.load(Q + off_q).to(tl.float32)

    k_ptrs = K + off_k
    v_ptrs = V + off_v

    block_offset_ptrs = Block_offsets + cur_batch * stride_boffb

    # initialize pointer to m and l
    m_i = -float('inf')
    l_i = float(0)
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    kv_len_per_prog = block_per_cta * BLOCK_N
    loop_start = kv_len_per_prog * split_k_id
    loop_end = tl.minimum(loop_start + kv_len_per_prog, cur_batch_kv_len)

    # load block offset
    start_block_id = loop_start // BLOCK_N
    b_offset = _load_block_offsets(block_offset_ptrs, start_block_id,
                                   num_sub_blocks, BLOCK_N)

    for start_n in range(loop_start, loop_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        mask = (start_n + offs_n[:, None]) < cur_batch_kv_len

        # -- compute qk ----
        k = tl.load(
            k_ptrs + b_offset[:, None] * stride_kbs,
            mask=mask,
            other=0.0,
        )

        v = tl.load(
            v_ptrs + b_offset[:, None] * stride_vbs,
            mask=mask,
            other=0.0,
        )

        # prefetch b_offset
        if start_n + BLOCK_N < loop_end:
            start_block_id += 1
            b_offset = _load_block_offsets(block_offset_ptrs, start_block_id,
                                           num_sub_blocks, BLOCK_N)

        qk = tl.sum(q[None, :] * k, 1)
        qk *= sm_scale
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
    off_acc = (cur_batch * stride_obs + split_k_id * stride_ok +
               cur_head * stride_oh + offs_d * stride_od)
    tl.store(Acc_out + off_acc, acc)

    off_meta = (cur_batch * stride_obs + split_k_id * stride_ok +
                cur_head * stride_oh + BLOCK_DMODEL)
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
    """second step kernel of split k attention."""
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    # initialize offsets
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_k = tl.arange(0, SPLIT_K)

    offs_acc = (cur_batch * stride_abs + cur_head * stride_ah +
                offs_k[:, None] * stride_ak + offs_d[None, :] * stride_ad)
    offs_mi = (cur_batch * stride_abs + cur_head * stride_ah +
               stride_ak * offs_k + BLOCK_DMODEL)

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
    num_sub_blocks: tl.constexpr,
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
    off_k = (cur_kv_head * stride_kh + offs_d[:, None] * stride_kd)
    off_v = (cur_kv_head * stride_vh + offs_d[None, :] * stride_vd)

    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    k_ptrs = K + off_k
    v_ptrs = V + off_v

    block_offset_ptrs = Block_offsets + cur_batch * stride_boffb

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)

    b_offset = _load_block_offsets(block_offset_ptrs, 0, num_sub_blocks,
                                   BLOCK_N)
    for start_n in range(0, block_mask * cur_batch_kv_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # -- compute qk ----
        k = tl.load(
            k_ptrs + b_offset[None, :] * stride_kbs,
            mask=(start_n + offs_n[None, :]) < cur_batch_kv_len,
            other=0.0,
        )

        v = tl.load(
            v_ptrs + b_offset[:, None] * stride_vbs,
            mask=(start_n + offs_n[:, None]) < cur_batch_kv_len,
            other=0.0,
        )
        if start_n + BLOCK_N < cur_batch_kv_len:
            start_block_id = start_n // BLOCK_N + 1
            b_offset = _load_block_offsets(block_offset_ptrs, start_block_id,
                                           num_sub_blocks, BLOCK_N)

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


@torch.inference_mode()
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

    def _kernel_meta():
        device = q.device
        device_idx = device.index
        device_type = device.type
        stream = get_cuda_stream(device_idx)
        return dict(device=device, device_type=device_type, stream=stream)

    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}

    sm_scale = 1.0 / (Lq**0.5)  # 计算scale系数
    batch, head = b_seq_len.shape[0], q.shape[-2]
    kv_group_num = q.shape[-2] // k.shape[-2]

    num_warps = 4 if Lk <= 64 else 8

    BLOCK = 64 if k.size(1) < 16 else k.size(1)
    num_sub_blocks = BLOCK // k.size(1)

    kernel_meta = _kernel_meta()
    is_decoding = q.shape[-3] == b_seq_len.size(0)
    if not is_decoding:
        grid = (batch, head, triton.cdiv(max_input_len, BLOCK))
        _fwd_kernel[grid](q,
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
                          num_sub_blocks=num_sub_blocks,
                          BLOCK_M=BLOCK,
                          BLOCK_DMODEL=Lk,
                          BLOCK_N=BLOCK,
                          num_warps=num_warps,
                          num_stages=1,
                          **kernel_meta)
    else:
        SPLIT_K = 4
        grid = (batch, head, SPLIT_K)
        block_per_cta = triton.cdiv(block_offsets.size(-1), SPLIT_K)
        acc = q.new_empty(batch, head, SPLIT_K, Lq + 2, dtype=torch.float32)
        _fwd_split_kernel[grid](q,
                                k,
                                v,
                                sm_scale,
                                b_kv_seq_len,
                                block_offsets,
                                acc,
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
                                stride_boffb=block_offsets.stride(0),
                                kv_group_num=kv_group_num,
                                block_per_cta=block_per_cta,
                                num_sub_blocks=num_sub_blocks,
                                BLOCK_DMODEL=Lk,
                                BLOCK_N=BLOCK,
                                num_warps=4,
                                num_stages=1,
                                **kernel_meta)

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
                                   BLOCK_DMODEL=Lk,
                                   num_warps=num_warps,
                                   num_stages=1,
                                   **kernel_meta)
