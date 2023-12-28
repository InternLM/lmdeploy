# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl
from triton.runtime.jit import get_cuda_stream


@triton.jit
def _rearange_all_gather_kernel(X, StartLoc, SeqLen, AdapterIds, Ranks, Out,
                                stride_x, stride_o, world_size,
                                BLOCK: tl.constexpr, BLOCK_P: tl.constexpr):
    """rearange all gather kernel."""
    batch_id = tl.program_id(0)
    block_id = tl.program_id(1)

    start_loc = tl.load(StartLoc + batch_id) + block_id * BLOCK
    seq_len = tl.load(SeqLen + batch_id)

    if block_id * BLOCK >= seq_len:
        return

    block_off = start_loc + tl.arange(0, BLOCK)
    block_mask = block_id * BLOCK + tl.arange(0, BLOCK) < seq_len

    adapter_id = tl.load(AdapterIds + batch_id)
    rank = tl.load(Ranks + adapter_id)
    prank = rank // world_size
    p_off = tl.arange(0, BLOCK_P)

    for p_id in range(world_size):
        ip_off = p_id * BLOCK_P + p_off
        i_mask = block_mask[:, None] and (p_off < prank)[None, :]
        i_off = block_off[:, None] * stride_x + ip_off[None, :]
        x = tl.load(X + i_off, mask=i_mask)

        op_off = p_id * prank + p_off
        o_mask = i_mask
        o_off = block_off[:, None] * stride_o + op_off[None, :]
        tl.store(Out + o_off, x, mask=o_mask)


@triton.jit
def _rearange_all_gather_decoding_kernel(X, AdapterIds, Ranks, Out, stride_x,
                                         stride_o, world_size, seq_len,
                                         BLOCK: tl.constexpr,
                                         BLOCK_P: tl.constexpr):
    """rearange all gather kernel."""
    block_id = tl.program_id(0)
    block_off = block_id * BLOCK + tl.arange(0, BLOCK)
    block_mask = block_off < seq_len

    adapter_ids = tl.load(AdapterIds + block_off, mask=block_mask)
    ranks = tl.load(Ranks + adapter_ids)
    pranks = ranks // world_size
    p_off = tl.arange(0, BLOCK_P)

    for p_id in range(world_size):
        ip_off = p_id * BLOCK_P + p_off
        i_mask = block_mask[:, None] and (p_off[None, :] < pranks[:, None])
        i_off = block_off[:, None] * stride_x + ip_off[None, :]
        x = tl.load(X + i_off, mask=i_mask)

        op_off = p_id * pranks[:, None] + p_off[None, :]
        o_mask = i_mask
        o_off = block_off[:, None] * stride_o + op_off
        tl.store(Out + o_off, x, mask=o_mask)


def rearange_all_gather(x: torch.Tensor,
                        b_start_loc: torch.Tensor,
                        b_seq_lens: torch.Tensor,
                        adapter_ids: torch.LongTensor,
                        ranks: torch.Tensor,
                        world_size: int,
                        max_seq_len: int,
                        output: torch.Tensor = None):
    """rearange all gather."""

    def _kernel_meta():
        device = x.device
        device_idx = device.index
        device_type = device.type
        stream = get_cuda_stream(device_idx)
        return dict(device=device, device_type=device_type, stream=stream)

    max_rank = x.size(1)
    batch_size = len(b_seq_lens)
    partition_size = max_rank // world_size

    if output is None:
        output = torch.empty_like(x)

    num_warps = 4
    kernel_meta = _kernel_meta()

    is_decoding = batch_size == x.size(0)
    if not is_decoding:
        BLOCK = 128
        BLOCK_P = partition_size
        grid = (batch_size, triton.cdiv(max_seq_len, BLOCK))
        _rearange_all_gather_kernel[grid](x,
                                          b_start_loc,
                                          b_seq_lens,
                                          adapter_ids,
                                          ranks,
                                          output,
                                          stride_x=x.stride(0),
                                          stride_o=output.stride(0),
                                          world_size=world_size,
                                          BLOCK=BLOCK,
                                          BLOCK_P=BLOCK_P,
                                          num_warps=num_warps,
                                          num_stages=1,
                                          **kernel_meta)
    else:
        BLOCK = 64
        BLOCK_P = partition_size
        seq_len = x.size(0)
        grid = (triton.cdiv(seq_len, BLOCK), )
        _rearange_all_gather_decoding_kernel[grid](x,
                                                   adapter_ids,
                                                   ranks,
                                                   output,
                                                   stride_x=x.stride(0),
                                                   stride_o=output.stride(0),
                                                   world_size=world_size,
                                                   seq_len=seq_len,
                                                   BLOCK=BLOCK,
                                                   BLOCK_P=BLOCK_P,
                                                   num_warps=num_warps,
                                                   num_stages=1,
                                                   **kernel_meta)

    return output
