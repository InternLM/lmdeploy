# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Sequence

import torch
import triton
import triton.language as tl
from torch import Tensor


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


def fill_kv_cache(k_states: Tensor,
                  v_states: Tensor,
                  k_caches: Tensor,
                  v_caches: Tensor,
                  start_loc: Tensor,
                  seq_length: Tensor,
                  block_offsets: Tensor,
                  history_lengths: Sequence,
                  context: Any = None):
    """fill kv cache for paged attention."""
    fill_cache_info = getattr(context, 'fill_cache_info', None)

    if fill_cache_info is None:
        batch_size = block_offsets.size(0)
        block_size = k_caches.size(1)

        if not isinstance(history_lengths, Tensor):
            history_lengths = torch.tensor(history_lengths,
                                           device=k_states.device)

        batch_ids = torch.arange(batch_size, device=k_states.device)

        first_block_ids = history_lengths // block_size
        block_offsets1d = block_offsets[batch_ids, first_block_ids]

        token_ids_start = history_lengths % block_size
        first_seq_len = torch.minimum(seq_length, block_size - token_ids_start)

        state_start = start_loc[:batch_size]
        state_len = first_seq_len
        cache_start = token_ids_start

        # middle + last = remain
        remain_seq_len = torch.maximum(seq_length.new_zeros(1),
                                       seq_length - first_seq_len)
        last_seq_len = remain_seq_len % block_size
        middle_seq_len = remain_seq_len - last_seq_len
        middle_block_nums = middle_seq_len // block_size
        remain_block_nums = (remain_seq_len / block_size).ceil().long()

        remain_state_start = [
            ss + slen +
            torch.arange(0, rlen, block_size, device=k_states.device)
            for ss, slen, rlen in zip(state_start, first_seq_len,
                                      remain_seq_len)
        ]
        remain_seq_lens = [
            torch.full((mid, ), block_size, device=k_states.device)
            for mid in middle_block_nums
        ]
        remain_seq_lens = [
            (torch.cat([slen, last]) if last != 0 else slen)
            for slen, last in zip(remain_seq_lens, last_seq_len.unsqueeze(-1))
        ]
        remain_block_offsets1d = [
            block_offsets[bid, ids:ids + ids_len]
            for bid, ids, ids_len in zip(range(batch_size), first_block_ids +
                                         1, remain_block_nums)
        ]

        # state_start store the state index of the block
        # state_len store the length to write in the block
        # cache_start store the first index the write in block
        # block_offsets1d store the index of block in caches
        state_start = torch.cat([state_start] + remain_state_start)
        state_len = torch.cat([state_len] + remain_seq_lens)
        cache_start = torch.cat(
            [cache_start] +
            [state_start.new_zeros(state_start.size(0) - batch_size)])
        block_offsets1d = torch.cat([block_offsets1d] + remain_block_offsets1d)

        if context is not None:
            fill_cache_info = dict()
            fill_cache_info['state_start'] = state_start
            fill_cache_info['state_len'] = state_len
            fill_cache_info['cache_start'] = cache_start
            fill_cache_info['block_offsets1d'] = block_offsets1d
        context.fill_cache_info = fill_cache_info
    else:
        state_start = fill_cache_info['state_start']
        state_len = fill_cache_info['state_len']
        cache_start = fill_cache_info['cache_start']
        block_offsets1d = fill_cache_info['block_offsets1d']

    grid = (state_start.size(0), )
    BLOCK_M = k_caches.size(-3)
    BLOCK_N = min(128, k_caches.stride(-3), v_caches.stride(-3))

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
    )
