# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import get_cuda_stream


def _next_pow_of_2(x):
    """get next power of 2."""
    return 1 << (x - 1).bit_length()


@triton.jit
def _x_a_mm_kernel(
    X,
    LoRA_A,
    XA,
    B_start_loc,
    B_seq_lens,
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
    BLOCK_M: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """xa mm kernel."""
    cur_batch = tl.program_id(0)
    start_m = tl.program_id(1)

    r_off = tl.arange(0, BLOCK_R)

    seq_len = tl.load(B_seq_lens + cur_batch)
    if start_m * BLOCK_M >= seq_len:
        return

    start_loc = tl.load(B_start_loc + cur_batch)
    adapter_id = tl.load(B_adapter_id + cur_batch)
    rank = tl.load(Ranks + adapter_id)

    page_table_off = adapter_id * stride_ptb + r_off
    rank_mask = r_off < rank
    page_table = tl.load(Rank_page_table + page_table_off, mask=rank_mask)

    m_off = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    r_off = tl.arange(0, BLOCK_R)
    dm_off = tl.arange(0, BLOCK_DMODEL)
    rank_mask = r_off < rank

    x_off = (start_loc + m_off) * stride_xs
    xs_mask = m_off < seq_len
    la_page_off = page_table * stride_las
    acc = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)

    # compute acc
    for start_h in range(0, BLOCK_H, BLOCK_DMODEL):
        cur_dm_off = start_h + dm_off
        h_mask = cur_dm_off < BLOCK_H

        # load x
        xh_off = cur_dm_off * stride_xh
        x_mask = xs_mask[:, None] and h_mask[None, :]
        x = tl.load(X + x_off[:, None] + xh_off[None, :],
                    mask=x_mask,
                    other=0.0)

        # load lora a
        lah_off = cur_dm_off * stride_lah
        la_mask = rank_mask[None, :] and h_mask[:, None]
        la = tl.load(LoRA_A + la_page_off[None, :] + lah_off[:, None],
                     mask=la_mask,
                     other=0.0)

        # compute
        acc += tl.dot(x, la)

    acc = acc.to(X.dtype.element_ty)
    xa_off = (start_loc + m_off) * stride_xas
    xas_mask = xs_mask
    xa_mask = xas_mask[:, None] and rank_mask[None, :]
    tl.store(XA + xa_off[:, None] + r_off[None, :] * stride_xar,
             acc,
             mask=xa_mask)


@triton.jit
def _acc_b_mm_kernel(
    XA,
    LoRA_B,
    Out,
    B_start_loc,
    B_seq_lens,
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
    BLOCK_M: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_HO: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    start_m = tl.program_id(1)

    r_off = tl.arange(0, BLOCK_R)

    seq_len = tl.load(B_seq_lens + cur_batch)
    if start_m * BLOCK_M >= seq_len:
        return

    start_loc = tl.load(B_start_loc + cur_batch)
    adapter_id = tl.load(B_adapter_id + cur_batch)
    rank = tl.load(Ranks + adapter_id)

    page_table_off = adapter_id * stride_ptb + r_off
    rank_mask = r_off < rank
    page_table = tl.load(Rank_page_table + page_table_off, mask=rank_mask)

    m_off = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    dm_off = tl.arange(0, BLOCK_DMODEL)
    rank_mask = r_off < rank
    lb_page_off = page_table * stride_lbs

    xs_mask = m_off < seq_len
    o_off = (start_loc + m_off) * stride_os
    os_mask = xs_mask

    xa_off = (start_loc + m_off) * stride_xas
    xa_mask = xs_mask[:, None] and rank_mask[None, :]
    acc = tl.load(XA + xa_off[:, None] + r_off[None, :] * stride_xar,
                  mask=xa_mask,
                  other=0.0)
    acc = acc.to(LoRA_B.dtype.element_ty)

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
        out = tl.dot(acc, lb)
        out = out.to(lb.dtype)

        # store o
        oh_off = cur_dm_off * stride_oh
        o_mask = os_mask[:, None] and h_mask[None, :]
        tl.store(Out + o_off[:, None] + oh_off[None, :], out, mask=o_mask)


@torch.inference_mode()
def mbgmm_a(x: Tensor, lora_a: Tensor, b_start_loc: Tensor, b_seq_lens: Tensor,
            b_adapter_ids: Tensor, rank_page_table: Tensor, ranks: Tensor,
            max_seq_len: int, max_rank: int):
    """mbgmm_a."""

    def _kernel_meta():
        device = x.device
        device_idx = device.index
        device_type = device.type
        stream = get_cuda_stream(device_idx)
        return dict(device=device, device_type=device_type, stream=stream)

    assert x.dim() == 2
    assert lora_a.dim() == 2
    assert rank_page_table.dim() == 2

    head_size = x.size(-1)
    batch_size = len(b_seq_lens)

    BLOCK_M = 32
    BLOCK_R = _next_pow_of_2(max_rank)
    if BLOCK_R < 16:
        BLOCK_R = 16
    BLOCK_H = head_size
    BLOCK_DMODEL = 64

    num_warps = 4
    grid = [batch_size, triton.cdiv(max_seq_len, BLOCK_M)]
    xa = x.new_empty((x.size(0), BLOCK_R))
    kernel_meta = _kernel_meta()
    _x_a_mm_kernel[grid](x,
                         lora_a,
                         xa,
                         b_start_loc,
                         b_seq_lens,
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
                         BLOCK_M=BLOCK_M,
                         BLOCK_R=BLOCK_R,
                         BLOCK_H=BLOCK_H,
                         BLOCK_DMODEL=BLOCK_DMODEL,
                         num_warps=num_warps,
                         num_stages=1,
                         **kernel_meta)
    return xa


@torch.inference_mode()
def mbgmm_b(xa: Tensor, lora_b: Tensor, b_start_loc: Tensor,
            b_seq_lens: Tensor, b_adapter_ids: Tensor, rank_page_table: Tensor,
            ranks: Tensor, max_seq_len: int, max_rank: int):
    """mbgmm_b."""

    def _kernel_meta():
        device = xa.device
        device_idx = device.index
        device_type = device.type
        stream = get_cuda_stream(device_idx)
        return dict(device=device, device_type=device_type, stream=stream)

    assert xa.dim() == 2
    assert lora_b.dim() == 2
    assert rank_page_table.dim() == 2

    head_o_size = lora_b.size(-1)
    batch_size = len(b_seq_lens)

    BLOCK_M = 32
    BLOCK_R = _next_pow_of_2(max_rank)
    if BLOCK_R < 16:
        BLOCK_R = 16
    BLOCK_HO = head_o_size
    BLOCK_DMODEL = 64

    num_warps = 4
    grid = [batch_size, triton.cdiv(max_seq_len, BLOCK_M)]
    output = xa.new_empty((xa.size(0), BLOCK_HO))
    kernel_meta = _kernel_meta()
    _acc_b_mm_kernel[grid](xa,
                           lora_b,
                           output,
                           b_start_loc,
                           b_seq_lens,
                           b_adapter_ids,
                           Rank_page_table=rank_page_table,
                           Ranks=ranks,
                           stride_xas=xa.stride(0),
                           stride_xar=xa.stride(1),
                           stride_os=output.stride(0),
                           stride_oh=output.stride(1),
                           stride_lbs=lora_b.stride(0),
                           stride_lbh=lora_b.stride(1),
                           stride_ptb=rank_page_table.stride(0),
                           BLOCK_M=BLOCK_M,
                           BLOCK_R=BLOCK_R,
                           BLOCK_HO=BLOCK_HO,
                           BLOCK_DMODEL=BLOCK_DMODEL,
                           num_warps=num_warps,
                           num_stages=1,
                           **kernel_meta)

    return output
