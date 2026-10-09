# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass

import torch


def get_dcp_topk_workspace_size(num_rows: int, topk: int, dcp_size: int) -> int:
    """Per-rank candidate bytes for cache sizing, not a top-k chunk limit."""
    # Two INT32 index buffers (8 bytes), local pairs (8), gathered pairs (8 * D).
    return num_rows * topk * (16 + 8 * dcp_size)


def get_dcp_prefill_workspace_size(*, batch_size: int, head_dim: int, block_size: int, dcp_size: int) -> int:
    """Per-rank prefix KV budget shared by chunk planning and cache sizing.

    Target 64 MiB, enlarged if one virtual block per request needs more. FP8 caches also gather dequantized BF16 KV.
    """
    # Simultaneous BF16 buffers: one local block, D gathered, D reordered.
    minimum = batch_size * block_size * (1 + 2 * dcp_size) * head_dim * 2
    return max(64 << 20, minimum)


def get_dcp_workspace_size(*, num_prefill_tokens: int, num_decode_tokens: int,
                           batch_size: int, num_heads: int, head_dim: int,
                           dtype: torch.dtype, block_size: int, dcp_size: int,
                           topk: int | None, score_workspace_bytes: int) -> int:
    """Estimate per-rank peak temporary bytes for enabled DCP.

    ``num_heads`` is the rank-local query-head count before DCP gathering. Reserve max(indexer, attention) plus final
    indices shared by both phases. Flattened indexer KV relies on memory headroom.
    """
    # Prefix KV: local, gathered, and reordered buffers.
    attention_workspace = get_dcp_prefill_workspace_size(
        batch_size=batch_size,
        head_dim=head_dim,
        block_size=block_size,
        dcp_size=dcp_size,
    )
    workspace_heads = num_heads
    if topk is not None:
        # Sparse FlashMLA output slices retain storage padded to 64-head multiples.
        workspace_heads = (workspace_heads + 63) // 64 * 64
    # Old, partial, and merged states: model-dtype output + FP32 LSE.
    # head_dim bounds the output width; standalone V-cache width is zero.
    attention_workspace += num_prefill_tokens * workspace_heads * 3 * (head_dim * dtype.itemsize + 4)
    if topk is None:
        return attention_workspace

    # Cover both prefill and MTP verification.
    max_rows = max(num_prefill_tokens, num_decode_tokens)
    # Indexer: scores, top-k ids, and local/gathered candidate pairs.
    indexer_workspace = score_workspace_bytes + get_dcp_topk_workspace_size(max_rows, topk, dcp_size)
    # Final INT32 ids survive into attention; other buffers are phase-local.
    index_bytes = max_rows * topk * 4
    return index_bytes + max(indexer_workspace, attention_workspace)


@dataclass(frozen=True)
class DCPPrefixChunk:
    """Metadata for one bounded chunk of cached prefix KV gathered across DCP
    ranks."""

    start: int
    size: int
    kv_seqlens: torch.Tensor
    cu_seqlens: torch.Tensor
    local_kv_seqlens: torch.Tensor  # [dcp_size, requests]
    local_cu_seqlens: torch.Tensor  # current rank


def build_dcp_prefix_chunks(*, prefix_lens: torch.Tensor, prefix_limit: int, block_size: int, head_dim: int,
                             dcp_world_rank: tuple[int, int]) -> tuple[DCPPrefixChunk, ...]:
    """Plan identical collective shapes on all ranks without device sync."""
    if prefix_limit <= 0:
        return ()
    dcp_size, dcp_rank = dcp_world_rank
    batch_size = prefix_lens.numel()
    workspace_bytes = get_dcp_prefill_workspace_size(batch_size=batch_size,
                                                     head_dim=head_dim,
                                                     block_size=block_size,
                                                     dcp_size=dcp_size)
    virtual_block_size = block_size * dcp_size
    bytes_per_block = batch_size * block_size * (1 + 2 * dcp_size) * head_dim * 2
    # Longer prefixes add chunks rather than enlarge the gather buffers.
    chunk_size = (workspace_bytes // bytes_per_block) * virtual_block_size
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
        chunks.append(DCPPrefixChunk(start, size, lengths, cu_lens, local_lengths, local_cu_lens))
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


def update_dcp_metadata(attn_metadata, step_context) -> None:
    """Populate DCP lengths and shared prefill metadata from common sequence
    fields.

    Without DCP, local lengths alias global lengths. Decode needs only local lengths; cached prefill additionally plans
    bounded KV chunks and, for sparse MLA, a request mapping shared across partitions and layers.
    """
    from lmdeploy.pytorch.distributed import get_dcp_world_rank

    dcp_world_rank = get_dcp_world_rank()
    attn_metadata.dcp_local_kv_seqlens = get_dcp_local_seq_lens(attn_metadata.kv_seqlens, dcp_world_rank)
    attn_metadata.dcp_prefix_chunks = ()
    attn_metadata.dcp_prefill_request_ids = None
    if dcp_world_rank[0] == 1 or attn_metadata.is_decoding:
        return

    num_tokens = step_context.input_ids.numel()
    prefix_total = attn_metadata.kv_flatten_size - num_tokens
    prefix_limit = min(prefix_total, max(0, attn_metadata.max_kv_seqlen - 1))
    attn_metadata.dcp_prefix_chunks = build_dcp_prefix_chunks(
        prefix_lens=attn_metadata.kv_seqlens - attn_metadata.q_seqlens,
        prefix_limit=prefix_limit,
        block_size=step_context.cache_config.block_size,
        head_dim=step_context.model_config.head_dim,
        dcp_world_rank=dcp_world_rank,
    )
    topk = step_context.model_config.mla_index_topk
    # Short-context sparse MLA uses dense attention and needs no index mapping.
    if attn_metadata.dcp_prefix_chunks and topk is not None and attn_metadata.max_kv_seqlen > topk:
        # Map query rows to requests once, reused across chunks and layers.
        # E.g. q_seqlens=[2, 3] -> request_ids=[0, 0, 1, 1, 1].
        attn_metadata.dcp_prefill_request_ids = torch.repeat_interleave(
            torch.arange(attn_metadata.q_seqlens.numel(), dtype=torch.int32, device=attn_metadata.q_seqlens.device),
            attn_metadata.q_seqlens,
            output_size=num_tokens)
