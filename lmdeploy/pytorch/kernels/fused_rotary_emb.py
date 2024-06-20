# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl
from torch import Tensor

from .triton_utils import get_kernel_meta, wrap_jit_func


@wrap_jit_func(type_hint=dict(Q=Tensor,
                              K=Tensor,
                              PostionIds=Tensor,
                              InvFreq=Tensor,
                              scaling_factor=float,
                              OutQ=Tensor,
                              OutK=Tensor,
                              stride_bq=int,
                              stride_sq=int,
                              stride_hq=int,
                              stride_dq=int,
                              stride_bk=int,
                              stride_sk=int,
                              stride_hk=int,
                              stride_dk=int,
                              stride_bp=int,
                              stride_sp=int,
                              max_seq_len=int,
                              BLOCK=torch.int32,
                              BLOCK_HQ=torch.int32,
                              BLOCK_HK=torch.int32,
                              BLOCK_F=torch.int32))
@triton.jit
def _fused_rotary_emb_kernel(
        Q, K, PostionIds, InvFreq, scaling_factor, OutQ, OutK, stride_bq,
        stride_sq, stride_hq: tl.constexpr, stride_dq: tl.constexpr, stride_bk,
        stride_sk, stride_hk: tl.constexpr, stride_dk: tl.constexpr, stride_bp,
        stride_sp, max_seq_len, BLOCK: tl.constexpr, BLOCK_HQ: tl.constexpr,
        BLOCK_HK: tl.constexpr, BLOCK_F: tl.constexpr):
    """fused rotary emb kernel."""
    batch_id = tl.program_id(0)
    seq_block_id = tl.program_id(1)

    s_off = seq_block_id * BLOCK + tl.arange(0, BLOCK)[:, None]
    f_off = tl.arange(0, BLOCK_F)[None, :]
    s_mask = s_off < max_seq_len

    bp_off = stride_bp * batch_id
    p_off = bp_off + stride_sp * s_off

    sq_off = batch_id * stride_bq + s_off * stride_sq
    q0_off = sq_off + f_off * stride_dq
    q1_off = q0_off + BLOCK_F * stride_dq

    sk_off = batch_id * stride_bk + s_off * stride_sk
    k0_off = sk_off + f_off * stride_dk
    k1_off = k0_off + BLOCK_F * stride_dk

    inv_freq = tl.load(InvFreq + f_off).to(tl.float32)
    position_ids = tl.load(PostionIds + p_off, mask=s_mask).to(tl.float32)
    position_ids = position_ids / scaling_factor

    # pos_freq = tl.dot(position_ids, inv_freq)
    pos_freq = position_ids * inv_freq
    cos = tl.cos(pos_freq).to(Q.dtype.element_ty)
    sin = tl.sin(pos_freq).to(Q.dtype.element_ty)

    for h in range(BLOCK_HQ):
        q0 = tl.load(Q + q0_off + h * stride_hq, mask=s_mask)
        q1 = tl.load(Q + q1_off + h * stride_hq, mask=s_mask)
        q0_out = q0 * cos - q1 * sin
        tl.store(OutQ + q0_off + h * stride_hq, q0_out, mask=s_mask)
        q1_out = q1 * cos + q0 * sin
        tl.store(OutQ + q1_off + h * stride_hq, q1_out, mask=s_mask)

    for h in range(BLOCK_HK):
        k0 = tl.load(K + k0_off + h * stride_hk, mask=s_mask)
        k1 = tl.load(K + k1_off + h * stride_hk, mask=s_mask)
        k0_out = k0 * cos - k1 * sin
        tl.store(OutK + k0_off + h * stride_hk, k0_out, mask=s_mask)
        k1_out = k1 * cos + k0 * sin
        tl.store(OutK + k1_off + h * stride_hk, k1_out, mask=s_mask)


def fused_rotary_emb(q: Tensor,
                     k: Tensor,
                     position_ids: torch.LongTensor,
                     inv_freq: Tensor,
                     scaling_factor: float,
                     out_q: Tensor = None,
                     out_k: Tensor = None):
    """Fuse `rotary_embedding` and `apply_rotary_pos_emb`."""

    if out_q is None:
        out_q = torch.empty_like(q)
    else:
        assert q.stride() == out_q.stride()
    if out_k is None:
        out_k = torch.empty_like(k)
    else:
        assert k.stride() == out_k.stride()

    assert q.dim() == 4
    assert k.dim() == 4
    assert q.size(0) == position_ids.size(0)

    BLOCK = 32
    BLOCK_HQ = q.size(-2)
    BLOCK_HK = k.size(-2)
    BLOCK_F = q.size(-1) // 2
    batch_size = q.size(0)
    max_seq_len = q.size(1)
    kernel_meta = get_kernel_meta(q)
    num_warps = 4

    grid = (batch_size, triton.cdiv(max_seq_len, BLOCK))
    _fused_rotary_emb_kernel[grid](q,
                                   k,
                                   position_ids,
                                   inv_freq,
                                   scaling_factor,
                                   out_q,
                                   out_k,
                                   stride_bq=q.stride(0),
                                   stride_sq=q.stride(1),
                                   stride_hq=q.stride(2),
                                   stride_dq=q.stride(3),
                                   stride_bk=k.stride(0),
                                   stride_sk=k.stride(1),
                                   stride_hk=k.stride(2),
                                   stride_dk=k.stride(3),
                                   stride_bp=position_ids.stride(0),
                                   stride_sp=position_ids.stride(1),
                                   max_seq_len=max_seq_len,
                                   BLOCK=BLOCK,
                                   BLOCK_HQ=BLOCK_HQ,
                                   BLOCK_HK=BLOCK_HK,
                                   BLOCK_F=BLOCK_F,
                                   num_warps=num_warps,
                                   num_stages=1,
                                   **kernel_meta)

    return out_q, out_k
