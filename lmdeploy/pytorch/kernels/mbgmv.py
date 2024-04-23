# Copyright (c) OpenMMLab. All rights reserved.
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import get_cuda_stream


def _next_pow_of_2(x):
    """get next power of 2."""
    return 1 << (x - 1).bit_length()


@triton.jit
def _x_a_mv_kernel(
    X,
    LoRA_A,
    XA,
    B_adapter_id,
    Rank_offset,
    Ranks,
    stride_xs,
    stride_xh,
    stride_xas,
    stride_xar,
    stride_ptb,
    rank_step,
    BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """xa mv kernel."""
    cur_batch = tl.program_id(0)

    r_off = tl.arange(0, BLOCK_R)
    adapter_id = tl.load(B_adapter_id + cur_batch)
    rank = tl.load(Ranks + adapter_id) // rank_step

    rank_off = adapter_id * stride_ptb + r_off
    rank_mask = r_off < rank

    dm_off = tl.arange(0, BLOCK_DMODEL)

    x_off = cur_batch * stride_xs
    la_page_off = tl.load(Rank_offset + rank_off, mask=rank_mask)
    acc = tl.zeros((BLOCK_R, ), dtype=tl.float32)

    # compute acc
    for start_h in range(0, BLOCK_H, BLOCK_DMODEL):
        cur_dm_off = start_h + dm_off
        h_mask = cur_dm_off < BLOCK_H

        # load x
        xh_off = cur_dm_off * stride_xh
        x_mask = h_mask
        x = tl.load(X + x_off + xh_off, mask=x_mask, other=0.0)

        # load lora a
        lah_off = cur_dm_off
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
    B_scaling,
    Rank_offset,
    Ranks,
    stride_xas,
    stride_xar,
    stride_os,
    stride_oh,
    stride_ptb,
    BLOCK_R: tl.constexpr,
    BLOCK_HO: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """acc b mv kernel."""
    cur_batch = tl.program_id(0)

    r_off = tl.arange(0, BLOCK_R)
    adapter_id = tl.load(B_adapter_id + cur_batch)
    scaling = tl.load(B_scaling + adapter_id)
    rank = tl.load(Ranks + adapter_id)

    rank_off = adapter_id * stride_ptb + r_off
    rank_mask = r_off < rank

    dm_off = tl.arange(0, BLOCK_DMODEL)
    lb_page_off = tl.load(Rank_offset + rank_off, mask=rank_mask)

    o_off = cur_batch * stride_os

    xa_off = cur_batch * stride_xas
    acc = tl.load(XA + xa_off + r_off * stride_xar, mask=rank_mask, other=0.0)

    # compute output
    for start_h in range(0, BLOCK_HO, BLOCK_DMODEL):
        cur_dm_off = start_h + dm_off
        h_mask = cur_dm_off < BLOCK_HO

        # load lora b
        lbh_off = cur_dm_off
        lb_mask = rank_mask[:, None] and h_mask[None, :]
        lb = tl.load(LoRA_B + lb_page_off[:, None] + lbh_off[None, :],
                     mask=lb_mask,
                     other=0)

        # compute
        out = tl.sum(acc[:, None] * lb, 0)
        out = out.to(lb.dtype)
        out = out * scaling

        # store o
        oh_off = cur_dm_off * stride_oh
        tl.store(Out + o_off + oh_off, out, mask=h_mask)


def mbgmv_a(x: Tensor,
            lora_a: Tensor,
            adapter_ids: Tensor,
            rank_offset: Tensor,
            ranks: Tensor,
            max_rank: int,
            rank_step: int = 1):
    """mbgmv_a."""

    def _kernel_meta():
        device = x.device
        device_idx = device.index
        device_type = device.type
        stream = get_cuda_stream(device_idx)
        return dict(device=device, device_type=device_type, stream=stream)

    assert x.dim() == 2
    assert lora_a.dim() == 2
    assert rank_offset.dim() == 2

    head_size = x.size(-1)
    batch_size = x.size(0)
    max_rank = max_rank // rank_step

    BLOCK_R = _next_pow_of_2(max_rank)
    BLOCK_H = head_size
    BLOCK_DMODEL = 512

    num_warps = 4
    grid = [batch_size]
    xa = x.new_empty((x.size(0), BLOCK_R))
    kernel_meta = _kernel_meta()
    _x_a_mv_kernel[grid](x,
                         lora_a,
                         xa,
                         adapter_ids,
                         Rank_offset=rank_offset,
                         Ranks=ranks,
                         stride_xs=x.stride(0),
                         stride_xh=x.stride(1),
                         stride_xas=xa.stride(0),
                         stride_xar=xa.stride(1),
                         stride_ptb=rank_offset.stride(0),
                         rank_step=rank_step,
                         BLOCK_R=BLOCK_R,
                         BLOCK_H=BLOCK_H,
                         BLOCK_DMODEL=BLOCK_DMODEL,
                         num_warps=num_warps,
                         num_stages=1,
                         **kernel_meta)
    return xa


def mbgmv_b(xa: Tensor,
            lora_b: Tensor,
            adapter_ids: Tensor,
            scaling: Tensor,
            rank_offset: Tensor,
            ranks: Tensor,
            max_rank: int,
            out_size: int = None):
    """mbgmv_b."""

    def _kernel_meta():
        device = xa.device
        device_idx = device.index
        device_type = device.type
        stream = get_cuda_stream(device_idx)
        return dict(device=device, device_type=device_type, stream=stream)

    assert xa.dim() == 2
    assert lora_b.dim() == 2
    assert rank_offset.dim() == 2

    if out_size is None:
        out_size = lora_b.size(-1)
    batch_size = xa.size(0)

    BLOCK_R = _next_pow_of_2(max_rank)
    BLOCK_HO = out_size
    BLOCK_DMODEL = 512

    num_warps = 4
    grid = [batch_size]
    output = xa.new_empty((xa.size(0), BLOCK_HO))
    kernel_meta = _kernel_meta()
    _acc_b_mv_kernel[grid](xa,
                           lora_b,
                           output,
                           adapter_ids,
                           scaling,
                           Rank_offset=rank_offset,
                           Ranks=ranks,
                           stride_xas=xa.stride(0),
                           stride_xar=xa.stride(1),
                           stride_os=output.stride(0),
                           stride_oh=output.stride(1),
                           stride_ptb=rank_offset.stride(0),
                           BLOCK_R=BLOCK_R,
                           BLOCK_HO=BLOCK_HO,
                           BLOCK_DMODEL=BLOCK_DMODEL,
                           num_warps=num_warps,
                           num_stages=1,
                           **kernel_meta)

    return output
