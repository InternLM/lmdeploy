# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl
from torch import Tensor

from .triton_utils import get_kernel_meta, wrap_jit_func


@wrap_jit_func(type_hint=dict(
    Q=Tensor,
    K=Tensor,
    COS=Tensor,
    SIN=Tensor,
    POS=Tensor,
    Q_EMB=Tensor,
    K_EMB=Tensor,
    seq_len=int,
    stride_qs=int,
    stride_qh=int,
    stride_qd=int,
    stride_ks=int,
    stride_kh=int,
    stride_kd=int,
    stride_qes=int,
    stride_qeh=int,
    stride_qed=int,
    stride_kes=int,
    stride_keh=int,
    stride_ked=int,
    half_size=torch.int32,
    BLOCK=torch.int32,
    BLOCK_QH=torch.int32,
    BLOCK_N=torch.int32,
))
@triton.jit(do_not_specialize=('seq_len', ))
def apply_rotary_pos_emb_qk_kernel(
    Q,
    K,
    COS,
    SIN,
    Q_EMB,
    K_EMB,
    seq_len,
    stride_qs: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_qes: tl.constexpr,
    stride_qeh: tl.constexpr,
    stride_qed: tl.constexpr,
    stride_kes: tl.constexpr,
    stride_keh: tl.constexpr,
    stride_ked: tl.constexpr,
    half_size: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_QH: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """apply rotary on key AND query kernel."""
    seq_block_id = tl.program_id(0)
    head_id = tl.program_id(1)

    pos_offset = seq_block_id * BLOCK + tl.arange(0, BLOCK)
    pos_mask = pos_offset < seq_len
    pos_offset = tl.max_contiguous(tl.multiple_of(pos_offset % seq_len, BLOCK),
                                   BLOCK)

    feat_size = half_size * 2
    feat_offset_l = tl.arange(0, BLOCK_N)
    feat_mask = feat_offset_l < half_size
    feat_offset_l = feat_offset_l % half_size
    feat_offset_h = half_size + feat_offset_l
    seq_mask = pos_mask[:, None] and feat_mask[None, :]
    cs_offset_l = pos_offset[:, None] * feat_size + feat_offset_l[None, :]
    cs_offset_h = pos_offset[:, None] * feat_size + feat_offset_h[None, :]
    q_elem_type = Q.dtype.element_ty
    cos_l = tl.load(COS + cs_offset_l).to(q_elem_type)
    cos_h = tl.load(COS + cs_offset_h).to(q_elem_type)
    sin_l = tl.load(SIN + cs_offset_l).to(q_elem_type)
    sin_h = tl.load(SIN + cs_offset_h).to(q_elem_type)

    if head_id < BLOCK_QH:
        q_ptr = Q + pos_offset * stride_qs
        qe_ptr = Q_EMB + pos_offset * stride_qes
        ql_ptrs = q_ptr[:, None] + feat_offset_l[None, :] * stride_qd
        qh_ptrs = q_ptr[:, None] + feat_offset_h[None, :] * stride_qd
        qel_ptrs = qe_ptr[:, None] + feat_offset_l[None, :] * stride_qed
        qeh_ptrs = qe_ptr[:, None] + feat_offset_h[None, :] * stride_qed
        ql_ptrs += head_id * stride_qh
        qh_ptrs += head_id * stride_qh
        qel_ptrs += head_id * stride_qeh
        qeh_ptrs += head_id * stride_qeh

        q_l = tl.load(ql_ptrs)
        q_h = tl.load(qh_ptrs)
        qe_l = q_l * cos_l - q_h * sin_l
        qe_h = q_h * cos_h + q_l * sin_h

        tl.store(qel_ptrs, qe_l, mask=seq_mask)
        tl.store(qeh_ptrs, qe_h, mask=seq_mask)
    else:
        head_id = head_id - BLOCK_QH
        k_ptr = K + pos_offset * stride_ks
        ke_ptr = K_EMB + pos_offset * stride_kes
        kl_ptrs = k_ptr[:, None] + feat_offset_l[None, :] * stride_kd
        kh_ptrs = k_ptr[:, None] + feat_offset_h[None, :] * stride_kd
        kel_ptrs = ke_ptr[:, None] + feat_offset_l[None, :] * stride_ked
        keh_ptrs = ke_ptr[:, None] + feat_offset_h[None, :] * stride_ked
        kl_ptrs += head_id * stride_kh
        kh_ptrs += head_id * stride_kh
        kel_ptrs += head_id * stride_keh
        keh_ptrs += head_id * stride_keh
        k_l = tl.load(kl_ptrs)
        k_h = tl.load(kh_ptrs)
        ke_l = k_l * cos_l - k_h * sin_l
        ke_h = k_h * cos_h + k_l * sin_h

        tl.store(kel_ptrs, ke_l, mask=seq_mask)
        tl.store(keh_ptrs, ke_h, mask=seq_mask)


def apply_rotary_pos_emb(q: Tensor,
                         k: Tensor,
                         cos: Tensor,
                         sin: Tensor,
                         q_embed: Tensor = None,
                         k_embed: Tensor = None):
    """Apply rotary positional embedding on query and key.

    Args:
        q (Tensor): Query state.
        k (Tensor): Key state.
        cos (Tensor): cosine matrix (seq_len, dim).
        sin (Tensor): sine matrix (seq_len, dim).
        q_embed (Tensor): output q, can be same as q
        k_embed (Tensor): output k, can be same as k

    Returns:
        Tuple[Tensor, Tensor]: Embedded query and key.
    """
    if cos.device != q.device:
        cos = cos.to(device=q.device)
    if sin.device != q.device:
        sin = sin.to(device=q.device)

    if q_embed is None:
        q_embed = torch.empty_like(q)
    if k_embed is None:
        k_embed = torch.empty_like(k)

    seq_len = cos.numel() // cos.size(-1)
    BLOCK = 16
    half_size = q.size(-1) // 2
    BLOCK_N = triton.next_power_of_2(half_size)
    num_heads_q = q.size(-2)
    num_heads_k = k.size(-2)
    num_warps = 4
    num_stages = 4

    kernel_meta = get_kernel_meta(q)
    grid = [triton.cdiv(seq_len, BLOCK), num_heads_q + num_heads_k]
    apply_rotary_pos_emb_qk_kernel[grid](q,
                                         k,
                                         cos,
                                         sin,
                                         q_embed,
                                         k_embed,
                                         seq_len=seq_len,
                                         stride_qs=q.stride(-3),
                                         stride_qh=q.stride(-2),
                                         stride_qd=q.stride(-1),
                                         stride_ks=k.stride(-3),
                                         stride_kh=k.stride(-2),
                                         stride_kd=k.stride(-1),
                                         stride_qes=q_embed.stride(-3),
                                         stride_qeh=q_embed.stride(-2),
                                         stride_qed=q_embed.stride(-1),
                                         stride_kes=k_embed.stride(-3),
                                         stride_keh=k_embed.stride(-2),
                                         stride_ked=k_embed.stride(-1),
                                         half_size=half_size,
                                         BLOCK=BLOCK,
                                         BLOCK_QH=num_heads_q,
                                         BLOCK_N=BLOCK_N,
                                         num_warps=num_warps,
                                         num_stages=num_stages,
                                         **kernel_meta)

    return q_embed, k_embed
