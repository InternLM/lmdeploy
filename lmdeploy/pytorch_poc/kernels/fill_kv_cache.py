# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Sequence

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import get_cuda_stream


@triton.jit
def _fill_kv_cache_kernel(
    k_states,
    v_states,
    k_caches,
    v_caches,
    state_start,
    state_len,
    cache_start,
    block_offsets1d,
    stride_kss,  # stride of key state token
    stride_vss,  # stride of value state token
    stride_kcs: tl.constexpr,  # stride of key cache token
    stride_vcs: tl.constexpr,  # stride of value cache token
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """fill kv cache kernel."""
    prog_id = tl.program_id(0)

    stride_kb = stride_kcs * BLOCK_M
    stride_vb = stride_vcs * BLOCK_M

    sstart = tl.load(state_start + prog_id)
    slen = tl.load(state_len + prog_id)
    cstart = tl.load(cache_start + prog_id)
    boffset = tl.load(block_offsets1d + prog_id)

    off_m = tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, BLOCK_N)

    ks_ptrs = k_states + (sstart +
                          off_m[:, None]) * stride_kss + off_n[None, :]
    vs_ptrs = v_states + (sstart +
                          off_m[:, None]) * stride_vss + off_n[None, :]
    kc_ptrs = k_caches + boffset * stride_kb + (
        cstart + off_m[:, None]) * stride_kcs + off_n[None, :]
    vc_ptrs = v_caches + boffset * stride_vb + (
        cstart + off_m[:, None]) * stride_vcs + off_n[None, :]

    mask = off_m[:, None] < slen

    for idx in range(0, stride_kcs, BLOCK_N):
        ks = tl.load(ks_ptrs + idx, mask=mask)
        tl.store(kc_ptrs + idx, ks, mask=mask)

    for idx in range(0, stride_vcs, BLOCK_N):
        vs = tl.load(vs_ptrs + idx, mask=mask)
        tl.store(vc_ptrs + idx, vs, mask=mask)


def _create_fill_cache_info(is_decoding: bool, block_size: int,
                            start_loc: Tensor, seq_length: Tensor,
                            block_offsets: Tensor, history_lengths: Sequence,
                            device: torch.device):
    """create information for cache filling.

    There are 4 tensor that we need to generate. Each one has shape (N) where
    N is the number of blocks that need to fill data.
    1. state_start: the token offset where we copy data from.
    2. cache_start: the token offset in block that we want to copy to.
    3. state_len: how many data (tokens) we want to copy
    4. block_offset1d: which block we want to perform the filling.
    """
    batch_size = block_offsets.size(0)

    # make sure history lengths is a tensor
    if not isinstance(history_lengths, Tensor):
        history_lengths = torch.tensor(history_lengths, device=device)

    first_block_ids = history_lengths // block_size
    token_ids_start = history_lengths % block_size
    if not is_decoding:
        # prefilling

        # initialize
        kv_seq_length = history_lengths + seq_length
        last_block_ids = kv_seq_length // block_size
        num_blocks = last_block_ids - first_block_ids + 1
        cum_num_blocks = num_blocks.cumsum(0)

        head_idx = torch.cat(
            [cum_num_blocks.new_zeros((1, )), cum_num_blocks[:-1]])
        tail_idx = head_idx + num_blocks - 1
        total_blocks = num_blocks.sum().item()

        # cache start
        cache_start = torch.zeros((total_blocks, ),
                                  dtype=torch.long,
                                  device=device)
        cache_start.index_put_((head_idx, ), token_ids_start)

        # state_len (cache_end - cache_start)
        cache_end = torch.full_like(cache_start, block_size)
        tail_len = (seq_length + token_ids_start + block_size) % block_size
        cache_end.index_put_((tail_idx, ), tail_len)
        state_len = cache_end - cache_start

        # state_start (0~cumed state len)
        cum_state_len = state_len.cumsum(0)
        state_start = torch.cat(
            [state_len.new_zeros((1, )), cum_state_len[:-1]])

        # block offsets
        block_offsets1d = [
            offs[first:last + 1] for offs, first, last in zip(
                block_offsets, first_block_ids.cpu(), last_block_ids.cpu())
        ]
        block_offsets1d = torch.cat(block_offsets1d)
    else:
        # decoding
        state_start = start_loc
        state_len = seq_length
        cache_start = token_ids_start
        batch_ids = torch.arange(batch_size, device=device)
        block_offsets1d = block_offsets[batch_ids, first_block_ids]

    fill_cache_info = dict()
    fill_cache_info['state_start'] = state_start
    fill_cache_info['state_len'] = state_len
    fill_cache_info['cache_start'] = cache_start
    fill_cache_info['block_offsets1d'] = block_offsets1d

    return fill_cache_info


def fill_kv_cache(k_states: Tensor,
                  v_states: Tensor,
                  k_caches: Tensor,
                  v_caches: Tensor,
                  start_loc: Tensor,
                  seq_length: Tensor,
                  block_offsets: Tensor,
                  history_lengths: Sequence,
                  context: Any = None):
    """fill key/value state to cache for paged attention.

    Paged attention required blocked layout key/value caches. This kernel
    fill states to cache blocks according to the block offsets.
    read https://vllm.ai/ for more detail.

    Args:
        k_states (Tensor): The key state in continuous batching layout.
        v_states (Tensor): The value state in continuous batching layout.
        k_caches (Tensor): The key cache in blocked layout.
        v_caches (Tensor): The value cache in blocked layout.
        start_loc (Tensor): The batch start sequence offset.
        seq_length (Tensor): The sequence of each data in batch.
        block_offsets (Tensor): The block offsets of kv caches.
        history_lengths (Sequence): The history lengths of each data in batch.
        context (Any): Context object of current step.
    """
    fill_cache_info = getattr(context, 'fill_cache_info', None)

    if fill_cache_info is None:
        is_decoding = k_states.size(-3) == seq_length.size(0)
        block_size = k_caches.size(1)
        fill_cache_info = _create_fill_cache_info(
            is_decoding,
            block_size,
            start_loc=start_loc,
            seq_length=seq_length,
            block_offsets=block_offsets,
            history_lengths=history_lengths,
            device=k_states.device)
        if context is not None:
            context.fill_cache_info = fill_cache_info

    state_start = fill_cache_info['state_start']
    state_len = fill_cache_info['state_len']
    cache_start = fill_cache_info['cache_start']
    block_offsets1d = fill_cache_info['block_offsets1d']

    grid = (state_start.size(0), )
    BLOCK_M = k_caches.size(-3)
    BLOCK_N = min(128, k_caches.stride(-3), v_caches.stride(-3))

    device = k_states.device
    device_idx = device.index
    device_type = device.type
    stream = get_cuda_stream(device_idx)
    _fill_kv_cache_kernel[grid](
        k_states,
        v_states,
        k_caches,
        v_caches,
        state_start=state_start,
        state_len=state_len,
        cache_start=cache_start,
        block_offsets1d=block_offsets1d,
        stride_kss=k_states.stride(-3),
        stride_vss=v_states.stride(-3),
        stride_kcs=k_caches.stride(-3),
        stride_vcs=v_caches.stride(-3),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=1,
        stream=stream,
        device=device_idx,
        device_type=device_type,
    )
