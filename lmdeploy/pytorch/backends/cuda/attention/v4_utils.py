# Copyright (c) OpenMMLab. All rights reserved.
"""Shared utilities for DeepSeek-V4 model and backend implementations."""

from dataclasses import dataclass

import torch


@dataclass
class V4PrefillTokenMeta:
    """Per-token sequence mapping for one prefill step."""

    token_pos: torch.Tensor
    total_tokens: torch.Tensor
    seq_id: torch.Tensor


@torch.compile(dynamic=True)
def build_prefix_positions(lengths: torch.Tensor, max_len: int):
    """Build ``[0, ..., len-1]`` positions padded with ``-1``."""
    device = lengths.device
    if max_len == 0:
        empty = torch.empty((lengths.numel(), 0), dtype=torch.int32, device=device)
        return empty, empty.bool()
    arange = torch.arange(max_len, dtype=torch.int32, device=device).unsqueeze(0)
    mask = arange < lengths.unsqueeze(1)
    positions = arange.masked_fill(~mask, -1)
    return positions, mask


@torch.compile(dynamic=True)
def build_window_positions(total_lens: torch.Tensor, window_size: int):
    """Build chronologically ordered trailing window positions in ring-buffer
    coordinates padded with ``-1``."""
    device = total_lens.device
    if window_size == 0:
        empty = torch.empty((total_lens.numel(), 0), dtype=torch.long, device=device)
        return empty, total_lens.new_zeros((total_lens.numel(), )), empty.bool()
    arange = torch.arange(window_size, device=device).unsqueeze(0)
    window_lens = total_lens.clamp(max=window_size)
    starts = total_lens - window_lens
    mask = arange < window_lens.unsqueeze(1)
    positions = torch.remainder(starts.unsqueeze(1) + arange, window_size)
    positions = positions.masked_fill(~mask, -1)
    return positions, window_lens, mask


def build_prefill_token_meta(q_seqlens: torch.Tensor,
                             cu_q_seqlens: torch.Tensor | None = None):
    """Build per-token prefill sequence mapping without CUDA sync.

    Given q_seqlens [bsz], returns token_pos [total_q] where token_pos[i] is the position of token i within its
    sequence, plus seq_id [total_q].
    """
    if cu_q_seqlens is None:
        cu_q_seqlens = torch.cat([
            q_seqlens.new_zeros(1, device=q_seqlens.device),
            q_seqlens.cumsum(0),
        ])
    total_tokens = cu_q_seqlens[-1]
    token_id = torch.arange(total_tokens, dtype=cu_q_seqlens.dtype,
                            device=q_seqlens.device)
    seq_id = torch.searchsorted(cu_q_seqlens[1:], token_id, right=True)
    token_pos = token_id - cu_q_seqlens[seq_id]
    return V4PrefillTokenMeta(
        token_pos=token_pos,
        total_tokens=total_tokens,
        seq_id=seq_id,
    )


def build_prefill_window_topk_indices(window_size: int,
                                      start_pos: torch.Tensor,
                                      token_meta: V4PrefillTokenMeta):
    """Build prefill local-window topk indices.

    The flat KV layout is ``[prev_window (ring buffer) | raw_kv |
    compressed]`` where ``prev_window`` starts at absolute position
    ``max(0, start_pos - window_size)``. The returned indices are positions
    in this flat KV tensor.
    """
    seq_id = token_meta.seq_id
    abs_pos = start_pos[seq_id] + token_meta.token_pos
    num_vis = (abs_pos + 1).clamp(max=window_size)

    window_start_abs = (start_pos - window_size).clamp(min=0)
    token_window_start_abs = window_start_abs[seq_id]
    first_vis_abs = (abs_pos - window_size + 1).clamp(min=0)
    first_flat_pos = first_vis_abs - token_window_start_abs

    col_idx = torch.arange(window_size, dtype=first_flat_pos.dtype,
                           device=start_pos.device).unsqueeze(0)
    valid = col_idx < num_vis.unsqueeze(1)
    flat_pos = torch.where(valid, first_flat_pos.unsqueeze(1) + col_idx,
                           col_idx.new_full((), -1))
    return flat_pos, num_vis


def build_prefill_compress_topk_indices(total_lens: torch.Tensor,
                                        compress_ratio: int,
                                        offset: torch.Tensor,
                                        start_pos: torch.Tensor,
                                        token_meta: V4PrefillTokenMeta,
                                        max_width: int):
    """Build prefill compressed-KV topk indices.

    Args:
        total_lens: [bsz] total KV length per sequence.
        compress_ratio: compression ratio (4 or 128).
        offset: [bsz] per-sequence offset into flattened KV.
        start_pos: [bsz] start position for each sequence.
        token_meta: precomputed per-token sequence mapping.
        max_width: maximum number of compressed entries.

    Returns:
        Tuple of topk [total_q_tokens, max_width] and per-token visible
        compressed count [total_q_tokens].
    """
    if not isinstance(max_width, int):
        raise TypeError('max_width must be a Python int to avoid CUDA sync.')

    num_compressed = torch.div(total_lens, compress_ratio, rounding_mode='floor').long()
    positions, mask = build_prefix_positions(num_compressed, max_width)
    positions = torch.where(mask, positions + offset.unsqueeze(1), positions)

    seq_id = token_meta.seq_id
    abs_pos = start_pos[seq_id] + token_meta.token_pos
    causal_limit = torch.div(abs_pos + 1, compress_ratio, rounding_mode='floor')
    return positions[seq_id], causal_limit
