# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/ModelTC/lightllm
import torch
import triton
import triton.language as tl
from packaging import version
from torch import Tensor
from triton.runtime.jit import get_cuda_stream

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

TRITON_VERSION = version.parse(triton.__version__)

assert TRITON_VERSION >= version.parse('2.1.0')


@triton.jit
def _load_block_offsets(offset_ptr, block_id, BLOCK: tl.constexpr):
    """load block offsets."""
    offs_n = tl.arange(0, BLOCK)
    return tl.load(offset_ptr + block_id) * BLOCK + offs_n


@triton.autotune(configs=[
    triton.Config({}, num_stages=1, num_warps=16),
    triton.Config({}, num_stages=1, num_warps=8),
    triton.Config({}, num_stages=1, num_warps=4),
],
                 key=['BLOCK_N', 'BLOCK_DMODEL', 'BLOCK_DV'])
@triton.jit
def _fwd_split_kernel(
    Q,
    K,
    V,
    sm_scale,
    KV_seqlens,
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
    window_size: tl.constexpr,
    head_size: tl.constexpr,
    head_size_v: tl.constexpr,
    shared_kv: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """first step kernel of split k attention."""
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_k_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    q_seqlen = 1
    kv_seqlen = tl.load(KV_seqlens + cur_batch)
    history_len = kv_seqlen - q_seqlen

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_d = offs_d < head_size
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_dv = offs_dv < head_size_v
    off_q = (cur_batch * stride_qbs + cur_head * stride_qh +
             offs_d * stride_qd)
    off_k = (cur_kv_head * stride_kh + offs_d[None, :] * stride_kd)
    off_v = (cur_kv_head * stride_vh + offs_dv[None, :] * stride_vd)

    q = tl.load(Q + off_q, mask=mask_d, other=0).to(tl.float32)

    k_ptrs = K + off_k
    v_ptrs = V + off_v

    block_offset_ptrs = Block_offsets + cur_batch * stride_boffb

    # initialize pointer to m and l
    m_i = -float('inf')
    l_i = float(0)
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    kv_len_per_prog = block_per_cta * BLOCK_N
    loop_start = kv_len_per_prog * split_k_id
    loop_end = tl.minimum(loop_start + kv_len_per_prog, kv_seqlen)

    # load block offset
    # dirty
    start_block_id = loop_start // BLOCK_N
    if window_size > 0:
        start_block_id = tl.maximum(history_len - window_size,
                                    loop_start) // BLOCK_N
        kv_min_loc = tl.maximum(history_len - window_size, 0)
    b_offset = _load_block_offsets(block_offset_ptrs, start_block_id, BLOCK_N)

    loop_start = start_block_id * BLOCK_N
    for start_n in range(loop_start, loop_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        mask = (start_n + offs_n[:, None]) < kv_seqlen

        # -- compute qk ----
        k = tl.load(
            k_ptrs + b_offset[:, None] * stride_kbs,
            mask=mask & mask_d[None, :],
            other=0.0,
        )

        if shared_kv:
            v = k
        else:
            v = tl.load(
                v_ptrs + b_offset[:, None] * stride_vbs,
                mask=mask & mask_dv[None, :],
                other=0.0,
            )

        # prefetch b_offset
        if start_n + BLOCK_N < loop_end:
            start_block_id += 1
            b_offset = _load_block_offsets(block_offset_ptrs, start_block_id,
                                           BLOCK_N)

        qk = tl.sum(q[None, :] * k, 1)
        qk *= sm_scale
        # NOTE: inf - inf = nan, and nan will leads to error
        qk_mask = history_len >= (start_n + offs_n)
        if window_size > 0:
            qk_mask = qk_mask and ((start_n + offs_n) >= kv_min_loc)
        qk = tl.where(
            qk_mask,
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
               cur_head * stride_oh + offs_dv * stride_od)
    tl.store(Acc_out + off_acc, acc, mask=mask_dv)

    off_meta = (cur_batch * stride_obs + split_k_id * stride_ok +
                cur_head * stride_oh + head_size_v)
    tl.store(Acc_out + off_meta + tl.arange(0, 1), m_i)
    tl.store(Acc_out + off_meta + 1 + tl.arange(0, 1), l_i)


@triton.autotune(configs=[
    triton.Config({}, num_stages=1, num_warps=16),
    triton.Config({}, num_stages=1, num_warps=8),
    triton.Config({}, num_stages=1, num_warps=4),
],
                 key=['BLOCK_H', 'BLOCK_N', 'BLOCK_DMODEL', 'BLOCK_DV'])
@triton.jit
def _fwd_grouped_split_kernel(
    Q,
    K,
    V,
    sm_scale,
    KV_seqlens,
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
    kv_group_num: tl.constexpr,
    block_per_cta,
    window_size: tl.constexpr,
    head_size: tl.constexpr,
    head_size_v: tl.constexpr,
    shared_kv: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """first step kernel of split k attention."""
    cur_batch = tl.program_id(0)
    cur_kv_head = tl.program_id(1)
    split_k_id = tl.program_id(2)

    heads_per_cta = min(BLOCK_H, kv_group_num)
    cur_head = cur_kv_head * heads_per_cta + tl.arange(0, BLOCK_H)
    mask_h = cur_head < cur_kv_head * heads_per_cta + heads_per_cta

    q_seqlen = 1
    kv_seqlen = tl.load(KV_seqlens + cur_batch)
    history_len = kv_seqlen - q_seqlen

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_d = offs_d < head_size
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_dv = offs_dv < head_size_v
    off_k = (cur_kv_head * stride_kh + offs_d[:, None] * stride_kd)
    off_v = (cur_kv_head * stride_vh + offs_dv[None, :] * stride_vd)

    off_q = (cur_batch * stride_qbs + cur_head[:, None] * stride_qh +
             offs_d[None, :] * stride_qd)
    q = tl.load(Q + off_q, mask=mask_h[:, None] & mask_d[None, :], other=0)

    k_ptrs = K + off_k
    v_ptrs = V + off_v

    block_offset_ptrs = Block_offsets + cur_batch * stride_boffb

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_H], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    kv_len_per_prog = block_per_cta * BLOCK_N
    loop_start = kv_len_per_prog * split_k_id
    loop_end = tl.minimum(loop_start + kv_len_per_prog, kv_seqlen)

    # load block offset
    # dirty
    start_block_id = loop_start // BLOCK_N
    if window_size > 0:
        start_block_id = tl.maximum(history_len - window_size,
                                    loop_start) // BLOCK_N
        kv_min_loc = tl.maximum(history_len - window_size, 0)
    b_offset = _load_block_offsets(block_offset_ptrs, start_block_id, BLOCK_N)

    loop_start = start_block_id * BLOCK_N
    for start_n in range(loop_start, loop_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        mask = (start_n + offs_n) < kv_seqlen

        # -- compute qk ----
        k = tl.load(
            k_ptrs + b_offset[None, :] * stride_kbs,
            mask=mask[None, :] & mask_d[:, None],
            other=0.0,
        )

        if shared_kv:
            v = tl.trans(k)
        else:
            v = tl.load(
                v_ptrs + b_offset[:, None] * stride_vbs,
                mask=mask[:, None] & mask_dv[None, :],
                other=0.0,
            )

        # prefetch b_offset
        if start_n + BLOCK_N < loop_end:
            start_block_id += 1
            b_offset = _load_block_offsets(block_offset_ptrs, start_block_id,
                                           BLOCK_N)

        qk = tl.zeros([BLOCK_H, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        # NOTE: inf - inf = nan, and nan will leads to error
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
        p = tl.exp(qk - m_i_new[:, None])
        alpha = tl.exp(m_i - m_i_new)
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
    off_acc = (cur_batch * stride_obs + split_k_id * stride_ok +
               cur_head[:, None] * stride_oh + offs_dv[None, :] * stride_od)
    tl.store(Acc_out + off_acc, acc, mask=mask_h[:, None] & mask_dv[None, :])

    off_meta = (cur_batch * stride_obs + split_k_id * stride_ok +
                cur_head * stride_oh + head_size_v)
    tl.store(Acc_out + off_meta, m_i, mask=mask_h)
    tl.store(Acc_out + off_meta + 1, l_i, mask=mask_h)


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

    acc_k = tl.load(Acc + offs_acc, mask=mask_dv[None, :], other=0.0)
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
                offs_dv * stride_od)
    tl.store(Out + out_offs, acc, mask=mask_dv)


def _get_convert_pv(nv_capability):
    """lazy load convert_pv."""
    if nv_capability[0] >= 8:

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


@triton.autotune(configs=[
    triton.Config({}, num_stages=1, num_warps=16),
    triton.Config({}, num_stages=1, num_warps=8),
    triton.Config({}, num_stages=1, num_warps=4),
],
                 key=['BLOCK_M', 'BLOCK_N', 'BLOCK_DMODEL', 'BLOCK_DV'])
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    Q_start_loc,
    Q_seqlens,
    KV_seqlens,
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
    window_size: tl.constexpr,
    head_size: tl.constexpr,
    head_size_v: tl.constexpr,
    shared_kv: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """paged attention kernel."""
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    q_seqlen = tl.load(Q_seqlens + cur_batch)
    kv_seqlen = tl.load(KV_seqlens + cur_batch)
    q_start_loc = tl.load(Q_start_loc + cur_batch)
    history_len = kv_seqlen - q_seqlen

    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < head_size
    mask_dv = offs_dv < head_size_v
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = ((q_start_loc + offs_m[:, None]) * stride_qbs +
             cur_head * stride_qh + offs_d[None, :] * stride_qd)
    off_k = (cur_kv_head * stride_kh + offs_d[:, None] * stride_kd)
    off_v = (cur_kv_head * stride_vh + offs_dv[None, :] * stride_vd)

    q = tl.load(Q + off_q,
                mask=(offs_m[:, None] < q_seqlen) & mask_d[None, :],
                other=0.0)

    k_ptrs = K + off_k
    v_ptrs = V + off_v

    block_offset_ptrs = Block_offsets + cur_batch * stride_boffb

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < q_seqlen, 1, 0)

    # this is dirty
    start_block_id = kv_seqlen - kv_seqlen
    if window_size > 0:
        start_block_id = tl.maximum(history_len - window_size, 0) // BLOCK_N
        kv_min_loc = tl.maximum(history_len + offs_m - window_size, 0)
    b_offset = _load_block_offsets(block_offset_ptrs, start_block_id, BLOCK_N)
    kv_start_loc = start_block_id * BLOCK_N
    for start_n in range(kv_start_loc, block_mask * kv_seqlen, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # -- compute qk ----
        k = tl.load(
            k_ptrs + b_offset[None, :] * stride_kbs,
            mask=(start_n + offs_n[None, :] < kv_seqlen) & mask_d[:, None],
            other=0.0,
        )
        if shared_kv:
            v = tl.trans(k)
        else:
            v = tl.load(
                v_ptrs + b_offset[:, None] * stride_vbs,
                mask=(start_n + offs_n[:, None] < kv_seqlen)
                & mask_dv[None, :],
                other=0.0,
            )
        if start_n + BLOCK_N < kv_seqlen:
            start_block_id = start_n // BLOCK_N + 1
            b_offset = _load_block_offsets(block_offset_ptrs, start_block_id,
                                           BLOCK_N)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        # NOTE: inf - inf = nan, and nan will leads to error
        qk_mask = (history_len + offs_m[:, None]) >= (start_n +
                                                      offs_n[None, :])
        if window_size > 0:
            qk_mask = qk_mask and (
                (start_n + offs_n[None, :]) >= kv_min_loc[:, None])
        qk = tl.where(
            qk_mask,
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
        p, v = _convert_pv(p, v)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    acc = acc / l_i[:, None]
    # initialize pointers to output
    off_o = ((q_start_loc + offs_m[:, None]) * stride_obs +
             cur_head * stride_oh + offs_dv[None, :] * stride_od)
    out_ptrs = Out + off_o
    tl.store(out_ptrs,
             acc,
             mask=(offs_m[:, None] < q_seqlen) & mask_dv[None, :])


def paged_attention_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    block_offsets: Tensor,
    q_start_loc: Tensor,
    q_seqlens: Tensor,
    kv_seqlens: Tensor,
    max_seqlen: int,
    window_size: int = None,
    sm_scale: float = None,
    shared_kv: int = False,
):
    """Paged Attention forward.

    Args:
        q (Tensor): Query state.
        k (Tensor): Key state caches.
        v (Tensor): Value state caches.
        o (Tensor): Output state.
        block_offsets (Tensor): The block offset of key and value.
        q_start_loc (Tensor): Start token location of each data in batch.
        q_seqlens (Tensor): Query length for each data in batch.
        kv_seqlens (Tensor): Key/Value length for each data in batch.
        max_seqlen (int): The max input length.
        BLOCK (int): The kernel block size.
    """
    global _convert_pv
    if _convert_pv is None:
        nv_cap = torch.cuda.get_device_capability()
        _convert_pv = _get_convert_pv(nv_cap)

    if window_size is None:
        window_size = -1

    def _kernel_meta():
        """kernel meta."""
        device = q.device
        device_idx = device.index
        device_type = device.type
        stream = get_cuda_stream(device_idx)
        return dict(device=device, device_type=device_type, stream=stream)

    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk, Lv == o.shape[-1]

    if sm_scale is None:
        sm_scale = 1.0 / (Lq**0.5)
    batch, head = q_seqlens.shape[0], q.shape[-2]
    kv_group_num = q.shape[-2] // k.shape[-2]

    BLOCK = k.size(1)
    assert BLOCK >= 16
    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    if shared_kv:
        BLOCK_DV = BLOCK_DMODEL
    else:
        BLOCK_DV = triton.next_power_of_2(Lv)
    BLOCK_M = max(16, min(BLOCK, 16384 // BLOCK_DMODEL))
    if Lk > 512 and BLOCK > 32:
        logger.warning(f'`head_dim={Lk}` and `block_size={BLOCK}` '
                       'might leads to bad performance. '
                       'Please reduce `block_size`.')

    kernel_meta = _kernel_meta()
    is_decoding = q.shape[-3] == q_seqlens.size(0)
    if not is_decoding:
        grid = (batch, head, triton.cdiv(max_seqlen, BLOCK_M))
        _fwd_kernel[grid](q,
                          k,
                          v,
                          sm_scale,
                          q_start_loc,
                          q_seqlens,
                          kv_seqlens,
                          block_offsets,
                          o,
                          stride_qbs=q.stride(-3),
                          stride_qh=q.stride(-2),
                          stride_qd=q.stride(-1),
                          stride_kbs=k.stride(-3),
                          stride_kh=k.stride(-2),
                          stride_kd=k.stride(-1),
                          stride_vbs=v.stride(-3),
                          stride_vh=v.stride(-2),
                          stride_vd=v.stride(-1),
                          stride_obs=o.stride(-3),
                          stride_oh=o.stride(-2),
                          stride_od=o.stride(-1),
                          stride_boffb=block_offsets.stride(0),
                          kv_group_num=kv_group_num,
                          window_size=window_size,
                          head_size=Lk,
                          head_size_v=Lv,
                          shared_kv=shared_kv,
                          BLOCK_M=BLOCK_M,
                          BLOCK_DMODEL=BLOCK_DMODEL,
                          BLOCK_DV=BLOCK_DV,
                          BLOCK_N=BLOCK,
                          **kernel_meta)
    else:
        num_warps = max(4, BLOCK_DMODEL // 64)
        SPLIT_K = 4
        block_per_cta = triton.cdiv(block_offsets.size(-1), SPLIT_K)
        acc = q.new_empty(batch, head, SPLIT_K, Lv + 2, dtype=torch.float32)
        if kv_group_num <= 2 or shared_kv:
            grid = (batch, head, SPLIT_K)
            _fwd_split_kernel[grid](q,
                                    k,
                                    v,
                                    sm_scale,
                                    kv_seqlens,
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
                                    window_size=window_size,
                                    head_size=Lk,
                                    head_size_v=Lv,
                                    shared_kv=shared_kv,
                                    BLOCK_DMODEL=BLOCK_DMODEL,
                                    BLOCK_DV=BLOCK_DV,
                                    BLOCK_N=BLOCK,
                                    **kernel_meta)
        else:
            BLOCK_H = max(16, min(BLOCK, kv_group_num))
            grid = (batch, head // min(BLOCK_H, kv_group_num), SPLIT_K)
            _fwd_grouped_split_kernel[grid](
                q,
                k,
                v,
                sm_scale,
                kv_seqlens,
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
                window_size=window_size,
                head_size=Lk,
                head_size_v=Lv,
                shared_kv=shared_kv,
                BLOCK_DMODEL=BLOCK_DMODEL,
                BLOCK_DV=BLOCK_DV,
                BLOCK_N=BLOCK,
                BLOCK_H=BLOCK_H,
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
                                   head_size_v=Lv,
                                   BLOCK_DV=BLOCK_DV,
                                   num_warps=num_warps,
                                   num_stages=1,
                                   **kernel_meta)
