# Copyright (c) OpenMMLab. All rights reserved.
"""Shared utilities for DeepSeek-V4 model and backend implementations."""

import torch


def build_prefix_positions(lengths: torch.Tensor, max_len: int):
    """Build ``[0, ..., len-1]`` positions padded with ``-1``."""
    device = lengths.device
    if max_len == 0:
        empty = torch.empty((lengths.numel(), 0), dtype=torch.long, device=device)
        return empty, empty.bool()
    arange = torch.arange(max_len, device=device).unsqueeze(0)
    mask = arange < lengths.unsqueeze(1)
    positions = arange.masked_fill(~mask, -1)
    return positions, mask


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


def _build_token_positions(q_seqlens: torch.Tensor):
    """Build per-token position indices without CUDA synchronization.

    Given q_seqlens [bsz], returns token_pos [total_q] where token_pos[i] is the position of token i within its
    sequence. Also returns total_q as a tensor (no .item()).
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

    The flat KV layout is ``[prev_window (ring buffer) | raw_kv | compressed]``
    where ``prev_window`` starts at absolute position
    ``window_start_abs = max(0, start_pos - window_size)``.  The returned
    indices are positions in this flat KV tensor (not 0-based relative).

    Args:
        total_lens: [bsz] total KV length per sequence.
        window_size: sliding window size.
        q_seqlens: [bsz] query sequence lengths (required when causal=True).
        start_pos: [bsz] start position for each sequence (required when
            causal=True).
        offset: offset to add to decode-path indices.
        causal: if True, apply per-token causal mask so that query position t
            can only attend to KV positions <= t.  Used for prefill.  When
            False (decode), all visible window KV entries are attended.

    Returns:
        topk: [total_q_tokens, window_size] (causal) or
        [bsz, 1, window_size] (non-causal) topk indices with -1 padding.
    """
    device = total_lens.device

    if causal and q_seqlens is not None:
        # Prefill path: build per-token indices; causal visibility is
        # controlled by ``topk_length`` at the flash_mla call site.
        token_pos_in_seq, _, _, seq_id = _build_token_positions(q_seqlens)

        # Per-token absolute position: start_pos + position_in_seq
        abs_pos = start_pos[seq_id] + token_pos_in_seq

        # Per-token number of visible window entries (for topk_length).
        num_vis = (abs_pos + 1).clamp(max=window_size)

        # Flat KV starts at window_start_abs = max(0, start_pos - window_size)
        # per sequence.  The first visible flat KV position for a token at
        # abs_pos is max(0, abs_pos - window_size + 1) - window_start_abs.
        window_start_abs = (start_pos - window_size).clamp(min=0)
        token_window_start_abs = window_start_abs[seq_id]
        first_vis_abs = (abs_pos - window_size + 1).clamp(min=0)
        first_flat_pos = first_vis_abs - token_window_start_abs

        # Build [total_q_tokens, window_size]: fill all slots from
        # first_flat_pos, -1 padding for entries beyond num_vis.
        # topk_length at the call site uses window_size (not num_vis) so
        # that flash_mla scans all window slots, skipping -1 entries.
        col_idx = torch.arange(window_size, device=device).unsqueeze(0)
        valid = col_idx < num_vis.unsqueeze(1)
        flat_pos = torch.where(valid, first_flat_pos.unsqueeze(1) + col_idx,
                               col_idx.new_full((), -1))
        return flat_pos, num_vis

    # Decode / non-causal path: simple topk range
    window_lens = total_lens.clamp(max=window_size).long()
    positions, mask = build_prefix_positions(window_lens, window_size)
    positions = torch.where(mask, positions + offset, positions)
    return positions.unsqueeze(1), None  # [bsz, 1, window_size]


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
        causal: if True, expand per-sequence indices to per-token and return
            per-token causal visibility count.  Per-token causal masking is
            controlled by ``topk_length`` at the ``flash_mla_sparse_fwd``
            call site, not by -1 padding here.
        max_width: maximum number of compressed entries (used for decode
            scratch allocation).  If None, uses max across the batch.

    Returns:
        (topk, num_vis): tuple where topk is [total_q_tokens, max_width]
        (causal) or [bsz, 1, max_width] (non-causal), and num_vis is
        [total_q_tokens] per-token visible compress count (causal) or None.
    """
    per_seq_offset = isinstance(offset, torch.Tensor)

    num_compressed = torch.div(total_lens, compress_ratio, rounding_mode='floor').long()
    if max_width is None:
        # NOTE: This causes a GPU→CPU sync. Callers should always provide max_width
        # to avoid this. Currently all callers in the V4 path pass max_width explicitly.
        max_width = int(num_compressed.max().cpu().item())
    elif isinstance(max_width, torch.Tensor):
        max_width = int(max_width.cpu().item())

    # Per-sequence sequential indices with -1 padding beyond num_compressed
    positions, mask = build_prefix_positions(num_compressed, max_width)
    if per_seq_offset:
        positions = torch.where(mask, positions + offset.unsqueeze(1), positions)
    else:
        positions = torch.where(mask, positions + offset, positions)

    if causal and q_seqlens is not None:
        token_pos_in_seq, total_q, _, seq_id = _build_token_positions(q_seqlens)
        abs_pos = start_pos[seq_id] + token_pos_in_seq
        causal_limit = torch.div(abs_pos + 1, compress_ratio, rounding_mode='floor')
        return positions.repeat_interleave(q_seqlens.long(), dim=0, output_size=total_q), causal_limit

    return positions.unsqueeze(1), None  # [bsz, 1, max_width]
