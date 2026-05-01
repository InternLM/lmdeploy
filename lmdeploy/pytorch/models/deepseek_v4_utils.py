# Copyright (c) OpenMMLab. All rights reserved.
"""Shared utilities for DeepSeek-V4 model and backend implementations."""

import torch
import torch.nn.functional as F


def gather_cache_entries(cache: torch.Tensor, block_offsets: torch.Tensor, positions: torch.Tensor,
                         block_size: int):
    """Gather entries from a named block cache with `-1` padded positions."""
    if positions.numel() == 0:
        return cache.new_empty((*positions.shape, cache.size(-1)))
    safe_positions = positions.clamp(min=0)
    block_idx = torch.div(safe_positions, block_size, rounding_mode='floor').long()
    max_block_idx = block_offsets.size(1)
    valid = (positions >= 0) & (block_idx < max_block_idx)
    safe_block_idx = block_idx.clamp(max=max_block_idx - 1)
    block_off = torch.remainder(safe_positions, block_size).long()
    phys_blocks = block_offsets.gather(1, safe_block_idx).long()
    gathered = cache[phys_blocks, block_off]
    return torch.where(valid.unsqueeze(-1), gathered, gathered.new_zeros(()))


def gather_compressed_cache_entries(cache: torch.Tensor, block_offsets: torch.Tensor, positions: torch.Tensor,
                                    block_size: int, compress_ratio: int):
    """Gather entries from a compressed KV block cache.

    Unlike ``gather_cache_entries``, this scales logical positions by
    ``compress_ratio`` before computing the physical block index.
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


def write_cache_entries(cache: torch.Tensor, block_offsets: torch.Tensor, batch_idx: torch.Tensor,
                        positions: torch.Tensor, values: torch.Tensor, block_size: int,
                        write_mask: torch.Tensor | None = None):
    """Write one entry per batch item into a named block cache."""
    if positions.numel() == 0:
        return
    block_idx = torch.div(positions, block_size, rounding_mode='floor').long()
    valid = (positions >= 0) & (block_idx < block_offsets.size(1))
    safe_block_idx = block_idx.clamp(max=block_offsets.size(1) - 1)
    block_off = torch.remainder(positions, block_size).long()
    phys_blocks = block_offsets[batch_idx, safe_block_idx].long()
    if write_mask is None:
        write_mask = valid
    else:
        write_mask = write_mask & valid
    if write_mask is None:
        cache[phys_blocks, block_off] = values.to(cache.dtype)
        return
    target = cache[phys_blocks, block_off]
    values = values.to(target.dtype)
    blend_mask = write_mask.view(-1, *([1] * (values.dim() - 1)))
    cache[phys_blocks, block_off] = torch.where(blend_mask, values, target)


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
    bsz = total_lens.numel()

    if causal and q_seqlens is not None:
        # Prefill path: build per-token causal indices
        # Use the per-sequence scalar version for compatibility
        results = []
        for s in range(bsz):
            sl = q_seqlens[s].item()
            sp = start_pos[s].item() if start_pos is not None else 0
            if sp >= window_size - 1:
                ring_pos = sp % window_size
                row = torch.cat([
                    torch.arange(ring_pos + 1, window_size, device=device),
                    torch.arange(0, ring_pos + 1, device=device)
                ], dim=0)
                # Repeat for each query token (all tokens see full window)
                matrix = row.unsqueeze(0).expand(sl, -1)
            elif sp > 0:
                row = F.pad(torch.arange(sp + 1, device=device), (0, window_size - sp - 1), value=-1)
                # Repeat for each query token
                matrix = row.unsqueeze(0).expand(sl, -1)
            else:
                # sp == 0: per-token causal mask
                base = torch.arange(sl, device=device).unsqueeze(1)
                matrix = (base - window_size + 1).clamp(0) + torch.arange(min(sl, window_size), device=device)
                matrix = torch.where(matrix > base, -1, matrix)
            results.append(matrix.unsqueeze(0))
        return torch.cat(results, dim=0)

    # Decode / non-causal path: simple topk range
    window_lens = total_lens.clamp(max=window_size).long()
    positions, mask = build_prefix_positions(window_lens, window_size)
    positions = torch.where(mask, positions + offset, positions)
    return positions.unsqueeze(1)  # [bsz, 1, window_size]


def build_compress_topk_indices(total_lens: torch.Tensor, compress_ratio: int, offset: int = 0,
                                q_seqlens: torch.Tensor | None = None,
                                start_pos: torch.Tensor | None = None,
                                causal: bool = False, max_width: int | None = None):
    """Build compressed KV topk indices for both prefill and decode.

    Args:
        total_lens: [bsz] total KV length per sequence.
        compress_ratio: compression ratio (4 or 128).
        offset: offset to add to indices (usually window_size or
            window_size + sl for prefill sp==0).
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

    if causal and q_seqlens is not None:
        # Prefill path: build per-token causal indices
        results = []
        for s in range(bsz):
            sl = q_seqlens[s].item()
            sp = start_pos[s].item() if start_pos is not None else 0
            if sp > 0:
                num_compressed = (sp + sl) // compress_ratio
                row = torch.arange(num_compressed, device=device) + offset
                matrix = row.unsqueeze(0).expand(sl, -1)
            else:
                matrix = torch.arange(sl // compress_ratio, device=device).repeat(sl, 1)
                mask = matrix >= torch.arange(1, sl + 1, device=device).unsqueeze(1) // compress_ratio
                matrix = torch.where(mask, -1, matrix + offset)
            results.append(matrix.unsqueeze(0))
        return torch.cat(results, dim=0)

    # Decode / non-causal path
    num_compressed = torch.div(total_lens, compress_ratio, rounding_mode='floor').long()
    if max_width is None:
        max_width = int(num_compressed.max().item()) if num_compressed.numel() > 0 else 0
    positions, mask = build_prefix_positions(num_compressed, max_width)
    positions = torch.where(mask, positions + offset, positions)
    return positions.unsqueeze(1)  # [bsz, 1, max_width]
