# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl
from torch import Tensor


def _next_pow_of_2(x):
    """get next power of 2."""
    return 1 << (x - 1).bit_length()


@triton.jit
def _x_a_mv_kernel(
    X,
    LoRA_A,
    XA,
    B_adapter_id,
    Rank_page_table,
    Ranks,
    stride_xs,
    stride_xh,
    stride_las,
    stride_lah,
    stride_xas,
    stride_xar,
    stride_ptb,
    BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """xa mv kernel."""
    cur_batch = tl.program_id(0)

    r_off = tl.arange(0, BLOCK_R)
    adapter_id = tl.load(B_adapter_id + cur_batch)
    rank = tl.load(Ranks + adapter_id)

    page_table_off = adapter_id * stride_ptb + r_off
    rank_mask = r_off < rank
    page_table = tl.load(Rank_page_table + page_table_off, mask=rank_mask)

    dm_off = tl.arange(0, BLOCK_DMODEL)
    rank_mask = r_off < rank

    x_off = cur_batch * stride_xs
    la_page_off = page_table * stride_las
    acc = tl.zeros((BLOCK_R, ), dtype=tl.float32)

    # compute acc
    for start_h in range(0, BLOCK_H, BLOCK_DMODEL):
        cur_dm_off = start_h + dm_off
        h_mask = cur_dm_off < BLOCK_H

        # load x
        xh_off = cur_dm_off * stride_xh
        x_mask = h_mask
        x = tl.load(X + x_off + xh_off, mask=x_mask, other=0.0).to(tl.float32)

        # load lora a
        lah_off = cur_dm_off * stride_lah
        la_mask = rank_mask[:, None] and h_mask[None, :]
        la = tl.load(LoRA_A + la_page_off[:, None] + lah_off[None, :],
                     mask=la_mask,
                     other=0.0)

        # compute
        acc += tl.sum(x[None, :] * la, 1)

    acc = acc.to(X.dtype.element_ty)
    xa_off = cur_batch * stride_xas
    tl.store(XA + xa_off + r_off * stride_xar, acc, mask=rank_mask)


@triton.jit
def _acc_b_mv_kernel(
    XA,
    LoRA_B,
    Out,
    B_adapter_id,
    Rank_page_table,
    Ranks,
    stride_xas,
    stride_xar,
    stride_os,
    stride_oh,
    stride_lbs,
    stride_lbh,
    stride_ptb,
    BLOCK_R: tl.constexpr,
    BLOCK_HO: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """acc b mv kernel."""
    cur_batch = tl.program_id(0)

    r_off = tl.arange(0, BLOCK_R)
    adapter_id = tl.load(B_adapter_id + cur_batch)
    rank = tl.load(Ranks + adapter_id)

    page_table_off = adapter_id * stride_ptb + r_off
    rank_mask = r_off < rank
    page_table = tl.load(Rank_page_table + page_table_off, mask=rank_mask)

    dm_off = tl.arange(0, BLOCK_DMODEL)
    rank_mask = r_off < rank
    lb_page_off = page_table * stride_lbs

    o_off = cur_batch * stride_os

    xa_off = cur_batch * stride_xas
    acc = tl.load(XA + xa_off + r_off * stride_xar, mask=rank_mask, other=0.0)

    # compute output
    for start_h in range(0, BLOCK_HO, BLOCK_DMODEL):
        cur_dm_off = start_h + dm_off
        h_mask = cur_dm_off < BLOCK_HO

        # load lora b
        lbh_off = cur_dm_off * stride_lbh
        lb_mask = rank_mask[:, None] and h_mask[None, :]
        lb = tl.load(LoRA_B + lb_page_off[:, None] + lbh_off[None, :],
                     mask=lb_mask,
                     other=0)

        # compute
        out = tl.sum(acc[:, None] * lb, 0)
        out = out.to(lb.dtype)

        # store o
        oh_off = cur_dm_off * stride_oh
        tl.store(Out + o_off + oh_off, out, mask=h_mask)


@torch.inference_mode()
def mbgmv_a(x: Tensor, lora_a: Tensor, b_adapter_ids: Tensor,
            rank_page_table: Tensor, ranks: Tensor, max_rank: int):
    """mbgmv_a."""

    assert x.dim() == 2
    assert lora_a.dim() == 2
    assert rank_page_table.dim() == 2

    head_size = x.size(-1)
    batch_size = x.size(0)

    BLOCK_R = _next_pow_of_2(max_rank)
    if BLOCK_R < 16:
        BLOCK_R = 16
    BLOCK_H = head_size
    BLOCK_DMODEL = 64

    num_warps = 4
    grid = [batch_size]
    xa = x.new_empty((x.size(0), BLOCK_R))

    _x_a_mv_kernel[grid](
        x,
        lora_a,
        xa,
        b_adapter_ids,
        Rank_page_table=rank_page_table,
        Ranks=ranks,
        stride_xs=x.stride(0),
        stride_xh=x.stride(1),
        stride_las=lora_a.stride(0),
        stride_lah=lora_a.stride(1),
        stride_xas=xa.stride(0),
        stride_xar=xa.stride(1),
        stride_ptb=rank_page_table.stride(0),
        BLOCK_R=BLOCK_R,
        BLOCK_H=BLOCK_H,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=num_warps,
        num_stages=1,
    )
    return xa


@torch.inference_mode()
def mbgmv_b(xa: Tensor, lora_b: Tensor, b_adapter_ids: Tensor,
            rank_page_table: Tensor, ranks: Tensor, max_rank: int):
    """mbgmv_b."""

    assert xa.dim() == 2
    assert lora_b.dim() == 2
    assert rank_page_table.dim() == 2

    head_o_size = lora_b.size(-1)
    batch_size = xa.size(0)

    BLOCK_R = _next_pow_of_2(max_rank)
    if BLOCK_R < 16:
        BLOCK_R = 16
    BLOCK_HO = head_o_size
    BLOCK_DMODEL = 64

    num_warps = 4
    grid = [batch_size]
    output = xa.new_empty((xa.size(0), BLOCK_HO))

    _acc_b_mv_kernel[grid](
        xa,
        lora_b,
        output,
        b_adapter_ids,
        Rank_page_table=rank_page_table,
        Ranks=ranks,
        stride_xas=xa.stride(0),
        stride_xar=xa.stride(1),
        stride_lbs=lora_b.stride(0),
        stride_lbh=lora_b.stride(1),
        stride_os=output.stride(0),
        stride_oh=output.stride(1),
        stride_ptb=rank_page_table.stride(0),
        BLOCK_R=BLOCK_R,
        BLOCK_HO=BLOCK_HO,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=num_warps,
        num_stages=1,
    )

    return output
