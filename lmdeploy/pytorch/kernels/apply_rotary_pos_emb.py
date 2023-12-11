# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import get_cuda_stream


@triton.jit
def apply_rotary_pos_emb_kernel(
    Q,
    COS,
    SIN,
    POS,
    Q_EMB,
    seq_len,
    stride_qh: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """apply rotary on key OR query kernel."""
    seq_block_id = tl.program_id(0)
    head_id = tl.program_id(1)

    pos_offset = seq_block_id * BLOCK + tl.arange(0, BLOCK)
    pos_ids = tl.load(POS + pos_offset, pos_offset < seq_len, other=-1)

    feat_size = BLOCK_N * 2
    feat_offset_l = tl.arange(0, BLOCK_N)
    feat_offset_h = BLOCK_N + feat_offset_l
    cs_offset_l = pos_ids[:, None] * feat_size + feat_offset_l[None, :]
    cs_offset_h = pos_ids[:, None] * feat_size + feat_offset_h[None, :]
    pos_ids_mask = pos_ids[:, None] >= 0
    cos_l = tl.load(COS + cs_offset_l, mask=pos_ids_mask)
    cos_h = tl.load(COS + cs_offset_h, mask=pos_ids_mask)
    sin_l = tl.load(SIN + cs_offset_l, mask=pos_ids_mask)
    sin_h = tl.load(SIN + cs_offset_h, mask=pos_ids_mask)
    q_offset_seq = pos_offset[:, None] * stride_qh + head_id * feat_size
    q_offset_l = q_offset_seq + feat_offset_l[None, :]
    q_offset_h = q_offset_seq + feat_offset_h[None, :]

    pos_mask = pos_offset[:, None] < seq_len
    q_l = tl.load(Q + q_offset_l, mask=pos_mask)
    q_h = tl.load(Q + q_offset_h, mask=pos_mask)

    q_emb_l = q_l * cos_l - q_h * sin_l
    q_emb_h = q_h * cos_h + q_l * sin_h

    tl.store(Q_EMB + q_offset_l, q_emb_l, mask=pos_mask)
    tl.store(Q_EMB + q_offset_h, q_emb_h, mask=pos_mask)


@triton.jit
def apply_rotary_pos_emb_qk_kernel(
    Q,
    K,
    COS,
    SIN,
    POS,
    Q_EMB,
    K_EMB,
    seq_len,
    stride_qh: tl.constexpr,
    stride_kh: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """apply rotary on key AND query kernel."""
    seq_block_id = tl.program_id(0)
    head_id = tl.program_id(1)

    pos_offset = seq_block_id * BLOCK + tl.arange(0, BLOCK)
    pos_ids = tl.load(POS + pos_offset, pos_offset < seq_len, other=-1)

    feat_size = BLOCK_N * 2
    feat_offset_l = tl.arange(0, BLOCK_N)
    feat_offset_h = BLOCK_N + feat_offset_l
    cs_offset_l = pos_ids[:, None] * feat_size + feat_offset_l[None, :]
    cs_offset_h = pos_ids[:, None] * feat_size + feat_offset_h[None, :]
    pos_ids_mask = pos_ids[:, None] >= 0
    cos_l = tl.load(COS + cs_offset_l, mask=pos_ids_mask)
    cos_h = tl.load(COS + cs_offset_h, mask=pos_ids_mask)
    sin_l = tl.load(SIN + cs_offset_l, mask=pos_ids_mask)
    sin_h = tl.load(SIN + cs_offset_h, mask=pos_ids_mask)

    q_offset_seq = pos_offset[:, None] * stride_qh + head_id * feat_size
    q_offset_l = q_offset_seq + feat_offset_l[None, :]
    q_offset_h = q_offset_seq + feat_offset_h[None, :]
    k_offset_seq = pos_offset[:, None] * stride_kh + head_id * feat_size
    k_offset_l = k_offset_seq + feat_offset_l[None, :]
    k_offset_h = k_offset_seq + feat_offset_h[None, :]

    pos_mask = pos_offset[:, None] < seq_len
    q_l = tl.load(Q + q_offset_l, mask=pos_mask)
    q_h = tl.load(Q + q_offset_h, mask=pos_mask)
    k_l = tl.load(K + k_offset_l, mask=pos_mask)
    k_h = tl.load(K + k_offset_h, mask=pos_mask)

    q_emb_l = q_l * cos_l - q_h * sin_l
    q_emb_h = q_h * cos_h + q_l * sin_h
    k_emb_l = k_l * cos_l - k_h * sin_l
    k_emb_h = k_h * cos_h + k_l * sin_h

    tl.store(Q_EMB + q_offset_l, q_emb_l, mask=pos_mask)
    tl.store(Q_EMB + q_offset_h, q_emb_h, mask=pos_mask)
    tl.store(K_EMB + k_offset_l, k_emb_l, mask=pos_mask)
    tl.store(K_EMB + k_offset_h, k_emb_h, mask=pos_mask)


@torch.inference_mode()
def apply_rotary_pos_emb(q: Tensor,
                         k: Tensor,
                         cos: Tensor,
                         sin: Tensor,
                         position_ids: Tensor,
                         position_ids_1d: Tensor = None,
                         q_embed: Tensor = None,
                         k_embed: Tensor = None):
    """Apply rotary positional embedding on query and key.

    Args:
        q (Tensor): Query state.
        k (Tensor): Key state.
        cos (Tensor): cosine matrix (seq_len, dim).
        sin (Tensor): sine matrix (seq_len, dim).
        position_ids (Tensor): Position ids of q and k.
        position_ids_1d (Tensor): 1d Position ids.
        q_embed (Tensor): output q, can be same as q
        k_embed (Tensor): output k, can be same as k

    Returns:
        Tuple[Tensor, Tensor]: Embedded query and key.
    """
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if cos.device != q.device or cos.dtype != q.dtype:
        cos = cos.to(device=q.device, dtype=q.dtype)
    if sin.device != q.device or sin.dtype != q.dtype:
        sin = sin.to(device=q.device, dtype=q.dtype)
    if position_ids_1d is None:
        seq_length = position_ids[..., -1] + 1
        position_ids_1d = [ids[:l] for ids, l in zip(position_ids, seq_length)]
        position_ids_1d = torch.cat(position_ids_1d)

    if q_embed is None:
        q_embed = torch.empty_like(q)
    if k_embed is None:
        k_embed = torch.empty_like(k)

    seq_len = position_ids_1d.size(-1)
    BLOCK = 32
    num_heads_q = q.size(-2)
    num_heads_k = k.size(-2)
    num_warps = 4
    num_stages = 2

    device = q.device
    device_idx = device.index
    device_type = device.type
    stream = get_cuda_stream(device_idx)
    if num_heads_k == num_heads_q:
        grid = [triton.cdiv(seq_len, BLOCK), num_heads_q]
        apply_rotary_pos_emb_qk_kernel[grid](q,
                                             k,
                                             cos,
                                             sin,
                                             position_ids_1d,
                                             q_embed,
                                             k_embed,
                                             seq_len=seq_len,
                                             stride_qh=q.stride(-3),
                                             stride_kh=k.stride(-3),
                                             BLOCK=BLOCK,
                                             BLOCK_N=q.size(-1) // 2,
                                             num_warps=num_warps,
                                             num_stages=num_stages,
                                             stream=stream,
                                             device=device_idx,
                                             device_type=device_type)

    else:
        grid_q = [triton.cdiv(seq_len, BLOCK), num_heads_q]
        grid_k = [triton.cdiv(seq_len, BLOCK), num_heads_k]
        apply_rotary_pos_emb_kernel[grid_q](q,
                                            cos,
                                            sin,
                                            position_ids_1d,
                                            q_embed,
                                            seq_len=seq_len,
                                            stride_qh=q.stride(-3),
                                            BLOCK=BLOCK,
                                            BLOCK_N=q.size(-1) // 2,
                                            num_warps=num_warps,
                                            num_stages=num_stages,
                                            stream=stream,
                                            device=device_idx,
                                            device_type=device_type)
        apply_rotary_pos_emb_kernel[grid_k](k,
                                            cos,
                                            sin,
                                            position_ids_1d,
                                            k_embed,
                                            seq_len=seq_len,
                                            stride_qh=k.stride(-3),
                                            BLOCK=BLOCK,
                                            BLOCK_N=k.size(-1) // 2,
                                            num_warps=num_warps,
                                            num_stages=num_stages,
                                            stream=stream,
                                            device=device_idx,
                                            device_type=device_type)

    return q_embed, k_embed
