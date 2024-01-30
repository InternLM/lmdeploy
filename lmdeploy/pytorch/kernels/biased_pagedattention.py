# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/ModelTC/lightllm
import torch
import triton
import triton.language as tl
from torch import Tensor

assert triton.__version__ >= '2.1.0'

_NV_CAP = torch.cuda.get_device_capability()
if _NV_CAP[0] >= 8:

    @triton.jit
    def _convert_pv(p, v):
        """convert pv."""
        p = p.to(v.dtype)
        return p, v
else:

    @triton.jit
    def _convert_pv(p, v):
        """convert pv."""
        v = v.to(p.dtype)
        return p, v


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Bias,
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
    stride_biasbs,
    stride_biash,
    stride_biasq,
    stride_biask,
    stride_obs,
    stride_oh,
    stride_od,
    stride_boffb,
    kv_group_num,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """biased paged attention kernel."""
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_kv_len = tl.load(B_kvlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

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
    off_bias = (cur_batch * stride_biasbs + cur_head * stride_biash +
                offs_m[:, None] * stride_biasq +
                offs_n[None, :] * stride_biask)

    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    k_ptrs = K + off_k
    v_ptrs = V + off_v
    bias_ptrs = Bias + off_bias

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

        bias = tl.load(
            bias_ptrs + start_n,
            mask=(start_n + offs_n[None, :]) < cur_batch_kv_len
            and (offs_m[:, None] < cur_batch_seq_len),
            other=-1e30,
        )
        qk += bias

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

        p, v = _convert_pv(p, v)
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
def biased_paged_attention_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    bias: Tensor,
    o: Tensor,
    block_offsets: Tensor,
    b_start_loc: Tensor,
    b_seq_len: Tensor,
    b_kv_seq_len: Tensor,
    max_input_len: int,
    BLOCK: int = 64,
):
    """Paged attention forward with custom bias.

    Args:
        q (Tensor): Query state.
        k (Tensor): Key state caches.
        v (Tensor): Value state caches.
        bias (Tensor): Bias of the QK.
        o (Tensor): Output state.
        block_offsets (Tensor): The block offset of key and value.
        b_start_loc (Tensor): Start token location of each data in batch.
        b_seq_len (Tensor): Query length for each data in batch.
        b_kv_seq_len (Tensor): Key/Value length for each data in batch.
        max_input_len (int): The max input length.
        BLOCK (int): The kernel block size.
    """
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    assert bias.dtype == torch.float32

    if bias.dim() == 2:
        bias = bias.unsqueeze(0)

    if bias.dim() == 3:
        bias = bias.unsqueeze(1)

    sm_scale = 1.0 / (Lq**0.5)  # 计算scale系数
    batch, head = b_seq_len.shape[0], q.shape[-2]
    kv_group_num = q.shape[-2] // k[0].shape[-2]

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))  # batch, head,
    bias_head_stride = 0 if bias.size(1) == 1 else bias.stride(-3)

    num_warps = 4 if Lk <= 64 else 8
    _fwd_kernel[grid](
        q,
        k,
        v,
        bias,
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
        bias.stride(-4),
        bias_head_stride,
        bias.stride(-2),
        bias.stride(-1),
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
