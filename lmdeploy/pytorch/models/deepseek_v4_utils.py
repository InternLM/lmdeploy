# Copyright (c) OpenMMLab. All rights reserved.
"""Shared utilities for DeepSeek-V4 model and backend implementations."""

import torch
import torch.nn.functional as F


def gather_compressed_cache_entries(cache: torch.Tensor, block_offsets: torch.Tensor, positions: torch.Tensor,
                                    block_size: int, compress_ratio: int):
    """Gather entries from a compressed KV block cache.

    Scales logical positions by ``compress_ratio`` before computing the
    physical block index.
    """
    if positions.numel() == 0:
        return cache.new_empty((*positions.shape, cache.size(-1)))
    safe_positions = positions.clamp(min=0)
    token_positions = safe_positions * compress_ratio
    block_idx = torch.div(token_positions, block_size, rounding_mode='floor').long()
    max_block_idx = block_offsets.size(1)
    valid = (positions >= 0) & (block_idx < max_block_idx)
    safe_block_idx = block_idx.clamp(max=max_block_idx - 1)
    block_off = torch.remainder(safe_positions, cache.size(1)).long()
    phys_blocks = block_offsets.gather(1, safe_block_idx).long()
    gathered = cache[phys_blocks, block_off]
    return torch.where(valid.unsqueeze(-1), gathered, gathered.new_zeros(()))


def build_prefix_positions(lengths: torch.Tensor, max_len: int):
    """Build ``[0, ..., len-1]`` positions padded with ``-1``."""
    device = lengths.device
    if max_len == 0:
        empty = torch.empty((lengths.numel(), 0), dtype=torch.long, device=device)
        return empty, empty.bool()
    arange = torch.arange(max_len, device=device).unsqueeze(0)
    mask = arange < lengths.unsqueeze(1)
    positions = torch.where(mask, arange, arange.new_full((), -1))
    return positions, mask


def _build_token_positions(q_seqlens: torch.Tensor):
    """Build per-token position indices without CUDA synchronization.

    Given q_seqlens [bsz], returns token_pos [total_q] where
    token_pos[i] is the position of token i within its sequence.
    Also returns total_q as a tensor (no .item()).
    """
    cu_q = torch.cat([q_seqlens.new_zeros(1, device=q_seqlens.device), q_seqlens.cumsum(0)])
    total_q = cu_q[-1]
    # token_pos[i] = i - cu_q[seq_of_i]
    global_idx = torch.arange(total_q, device=q_seqlens.device)
    # Find which sequence each token belongs to: searchsorted in cu_q
    seq_id = torch.searchsorted(cu_q[1:], global_idx, right=True)
    token_pos = global_idx - cu_q[seq_id]
    return token_pos, total_q, cu_q, seq_id


def build_window_topk_indices(total_lens: torch.Tensor, window_size: int,
                              q_seqlens: torch.Tensor | None = None,
                              start_pos: torch.Tensor | None = None,
                              offset: int = 0, causal: bool = False):
    """Build window topk indices for both prefill and decode.

    Args:
        total_lens: [bsz] total KV length per sequence.
        window_size: sliding window size.
        q_seqlens: [bsz] query sequence lengths (required when causal=True).
        start_pos: [bsz] start position for each sequence (required when
            causal=True and start_pos varies across sequences).
        offset: offset to add to indices.
        causal: if True, apply per-token causal mask so that query position t
            can only attend to KV positions <= t.  Used for prefill.  When
            False (decode), all visible window KV entries are attended.

    Returns:
        topk: [bsz, seqlen, window_size] (causal) or [bsz, 1, window_size]
        (non-causal) topk indices with -1 padding.
    """
    device = total_lens.device

    if causal and q_seqlens is not None:
        # Prefill path: build per-token causal indices
        # Returns [total_q_tokens, window_size] (flattened across batch)
        token_pos_in_seq, total_q, cu_q, seq_id = _build_token_positions(q_seqlens)

        # Per-token absolute position: start_pos + position_in_seq
        abs_pos = start_pos[seq_id] + token_pos_in_seq

        # Per-token window_start and num_vis [total_q_tokens]
        window_starts = (total_lens - window_size).clamp(min=0)
        token_window_start = window_starts[seq_id]
        num_vis = (abs_pos - token_window_start + 1).clamp(min=0, max=window_size)

        # Build [total_q_tokens, window_size] mask: col j < num_vis[i]
        col_idx = torch.arange(window_size, device=device).unsqueeze(0)
        valid = col_idx < num_vis.unsqueeze(1)
        flat_pos = torch.where(valid, col_idx, col_idx.new_full((), -1))
        return flat_pos

    # Decode / non-causal path: simple topk range
    window_lens = total_lens.clamp(max=window_size).long()
    positions, mask = build_prefix_positions(window_lens, window_size)
    positions = torch.where(mask, positions + offset, positions)
    return positions.unsqueeze(1)  # [bsz, 1, window_size]


def build_compress_topk_indices(total_lens: torch.Tensor, compress_ratio: int, offset=0,
                                q_seqlens: torch.Tensor | None = None,
                                start_pos: torch.Tensor | None = None,
                                causal: bool = False, max_width: int | None = None):
    """Build compressed KV topk indices for both prefill and decode.

    Args:
        total_lens: [bsz] total KV length per sequence.
        compress_ratio: compression ratio (4 or 128).
        offset: offset to add to indices. Can be a scalar int or a [bsz]
            tensor of per-sequence offsets.
        q_seqlens: [bsz] query sequence lengths (required when causal=True).
        start_pos: [bsz] start position for each sequence (required when
            causal=True).
        causal: if True, apply per-token causal mask so that query position t
            can only attend to compressed KV with position <= t.
        max_width: maximum number of compressed entries (used for decode
            scratch allocation).  If None, uses max across the batch.

    Returns:
        topk: [bsz, seqlen, max_width] (causal) or [bsz, 1, max_width]
        (non-causal) topk indices with -1 padding.
    """
    device = total_lens.device
    bsz = total_lens.numel()
    per_seq_offset = isinstance(offset, torch.Tensor)

    if causal and q_seqlens is not None:
        # Prefill path: build per-token causal indices
        # Returns [total_q_tokens, max_width] (flattened across batch)
        if max_width is None:
            max_comp = torch.div(total_lens, compress_ratio, rounding_mode='floor').max()
            max_width = int(max_comp.item())
        elif isinstance(max_width, torch.Tensor):
            max_width = int(max_width.item())

        token_pos_in_seq, total_q, cu_q, seq_id = _build_token_positions(q_seqlens)
        abs_pos = start_pos[seq_id] + token_pos_in_seq

        # Per-token offset [total_q_tokens]
        token_offset = offset[seq_id] if per_seq_offset else offset

        # Per-token number of total compressed entries
        num_total = torch.div(total_lens, compress_ratio, rounding_mode='floor')[seq_id]

        # Causal limit: token at position t can see entries with index < (t+1) // compress_ratio
        # For sp > 0 this equals num_total (all compressed entries are in the past).
        # For sp == 0, only entries with compressed position < (t+1) // ratio are visible.
        causal_limit = torch.div(abs_pos + 1, compress_ratio, rounding_mode='floor')
        has_prefix = start_pos[seq_id] > 0
        limit = torch.where(has_prefix, num_total, causal_limit)

        # Build [total_q_tokens, max_width]
        col_idx = torch.arange(max_width, device=device).unsqueeze(0)
        valid = col_idx < limit.unsqueeze(1)

        if per_seq_offset:
            positions = torch.where(valid, col_idx + token_offset.unsqueeze(1), col_idx.new_full((), -1))
        else:
            positions = torch.where(valid, col_idx + offset, col_idx.new_full((), -1))
        return positions

    # Decode / non-causal path
    num_compressed = torch.div(total_lens, compress_ratio, rounding_mode='floor').long()
    if max_width is None:
        max_width = int(num_compressed.max().item())  # static per graph capture
    positions, mask = build_prefix_positions(num_compressed, max_width)
    if per_seq_offset:
        positions = torch.where(mask, positions + offset.unsqueeze(1), positions)
    else:
        positions = torch.where(mask, positions + offset, positions)
    return positions.unsqueeze(1)  # [bsz, 1, max_width]
