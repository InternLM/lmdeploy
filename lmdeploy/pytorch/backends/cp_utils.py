# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass

import torch


def get_dcp_topk_workspace_size(num_rows: int, topk: int, dcp_size: int) -> int:
    """Bytes for local/global INT32 ids and packed/gathered FP32 pairs."""
    return num_rows * topk * (16 + 8 * dcp_size)


def get_dcp_prefill_workspace_size(*, batch_size: int, head_dim: int, block_size: int, dcp_size: int) -> int:
    """Bound BF16 local, gathered, and reordered prefix KV buffers.

    Allow at least one virtual block per request. The same bound is reserved during cache sizing and used to plan
    context chunks, independently of Q.
    """
    minimum = batch_size * block_size * (1 + 2 * dcp_size) * head_dim * 2
    return max(64 << 20, minimum)


@dataclass(frozen=True)
class DcpPrefillChunk:
    """Layer-independent lengths for one virtual-block-aligned prefix chunk."""

    start: int
    size: int
    kv_seqlens: torch.Tensor
    cu_seqlens: torch.Tensor
    local_kv_seqlens: torch.Tensor  # [dcp_size, requests]
    local_cu_seqlens: torch.Tensor  # current rank


def build_dcp_prefill_chunks(*, prefix_lens: torch.Tensor, prefix_limit: int, block_size: int, head_dim: int,
                             dcp_world_rank: tuple[int, int]) -> tuple[DcpPrefillChunk, ...]:
    """Plan identical collective shapes on all ranks without device sync."""
    if prefix_limit <= 0:
        return ()
    dcp_size, dcp_rank = dcp_world_rank
    batch_size = prefix_lens.numel()
    budget = get_dcp_prefill_workspace_size(batch_size=batch_size,
                                            head_dim=head_dim,
                                            block_size=block_size,
                                            dcp_size=dcp_size)
    virtual_block_size = block_size * dcp_size
    bytes_per_block = batch_size * block_size * (1 + 2 * dcp_size) * head_dim * 2
    chunk_size = (budget // bytes_per_block) * virtual_block_size
    ranks = torch.arange(dcp_size, device=prefix_lens.device, dtype=prefix_lens.dtype)[:, None]
    chunks = []
    for start in range(0, prefix_limit, chunk_size):
        # Bound the last allocation too, preserving the virtual-block alignment.
        size = min(chunk_size,
                   ((prefix_limit - start + virtual_block_size - 1) // virtual_block_size) * virtual_block_size)
        lengths = (prefix_lens - start).clamp(min=0, max=size)
        local_lengths = ((lengths[None, :] + dcp_size - 1 - ranks) // dcp_size).clamp_min(0)
        cu_lens = torch.nn.functional.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
        local_cu_lens = torch.nn.functional.pad(local_lengths[dcp_rank].cumsum(0, dtype=torch.int32), (1, 0))
        chunks.append(DcpPrefillChunk(start, size, lengths, cu_lens, local_lengths, local_cu_lens))
    return tuple(chunks)


def get_dcp_local_seq_lens(seq_lens: torch.Tensor,
                           dcp_world_rank: tuple[int, int]) -> torch.Tensor:
    """Return interleaved per-rank lengths for global sequence lengths."""
    dcp_world_size, dcp_rank = dcp_world_rank
    if dcp_world_size == 1:
        return seq_lens
    numer = seq_lens + dcp_world_size - 1 - dcp_rank
    return torch.clamp(torch.div(numer, dcp_world_size, rounding_mode='floor'),
                       min=0)


def get_dcp_local_causal_seq_lens(kv_seqlens: torch.Tensor, query_len: int,
                                  dcp_world_rank: tuple[int, int]) -> torch.Tensor:
    """Return rank-local causal lengths, flattened in request/query order."""
    if query_len > 1:
        offsets = torch.arange(1 - query_len, 1, device=kv_seqlens.device, dtype=kv_seqlens.dtype)
        kv_seqlens = (kv_seqlens[:, None] + offsets).flatten()
    return get_dcp_local_seq_lens(kv_seqlens, dcp_world_rank)


def get_dcp_local_cu_seqlens(
        seq_lens: torch.Tensor,
        dcp_world_rank: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return local lengths and their INT32 cumulative offsets."""
    local_lens = get_dcp_local_seq_lens(seq_lens, dcp_world_rank)
    cu_lens = torch.nn.functional.pad(
        torch.cumsum(local_lens, dim=0, dtype=torch.int32), (1, 0))
    return local_lens, cu_lens


def fill_dcp_local_seq_lens(seq_lens: torch.Tensor,
                            local_lens: torch.Tensor,
                            dcp_world_rank: tuple[int, int]) -> None:
    """Refresh a graph-stable local-length buffer without allocations."""
    dcp_world_size, dcp_rank = dcp_world_rank
    if local_lens.shape != seq_lens.shape:
        raise ValueError('local_lens and seq_lens must have identical shapes')
    torch.add(seq_lens, dcp_world_size - 1 - dcp_rank, out=local_lens)
    torch.div(local_lens,
              dcp_world_size,
              rounding_mode='floor',
              out=local_lens)
    local_lens.clamp_min_(0)


def get_dcp_local_indices(indices: torch.Tensor,
                          dcp_world_rank: tuple[int, int]) -> torch.Tensor:
    """Map global token positions to this rank's local positions."""
    dcp_world_size, dcp_rank = dcp_world_rank
    if dcp_world_size == 1:
        return indices
    valid = (indices >= 0) & (indices % dcp_world_size == dcp_rank)
    local_indices = torch.div(indices.clamp_min(0),
                              dcp_world_size,
                              rounding_mode='floor')
    return torch.where(valid, local_indices, -1)


def compact_dcp_local_indices(
        indices: torch.Tensor,
        dcp_world_rank: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Filter global winners to this rank and compact valid local ids."""
    local_indices = get_dcp_local_indices(indices, dcp_world_rank)
    return compact_valid_indices(local_indices)


def compact_valid_indices(
        indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Move valid indices before ``-1`` padding, preserving their order."""
    valid = indices >= 0
    valid_counts = valid.sum(dim=-1, dtype=torch.int32)
    order = torch.argsort((~valid).to(torch.int32), dim=-1, stable=True)
    return indices.gather(-1, order), valid_counts
