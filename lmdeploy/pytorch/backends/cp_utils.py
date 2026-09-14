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
    workspace_bytes = get_dcp_prefill_workspace_size(batch_size=batch_size,
                                                     head_dim=head_dim,
                                                     block_size=block_size,
                                                     dcp_size=dcp_size)
    virtual_block_size = block_size * dcp_size
    bytes_per_block = batch_size * block_size * (1 + 2 * dcp_size) * head_dim * 2
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


def update_dcp_metadata(attn_metadata, step_context) -> None:
    """Populate DCP lengths and shared prefill metadata from common sequence
    fields.

    Without DCP, local lengths alias global lengths. Decode needs only local lengths; cached prefill additionally plans
    bounded KV chunks and, for sparse MLA, a request mapping shared across partitions and layers.
    """
    from lmdeploy.pytorch.distributed import get_dcp_world_rank

    dcp_world_rank = get_dcp_world_rank()
    attn_metadata.dcp_local_kv_seqlens = get_dcp_local_seq_lens(attn_metadata.kv_seqlens, dcp_world_rank)
    attn_metadata.dcp_prefill_chunks = ()
    attn_metadata.dcp_prefill_request_ids = None
    if dcp_world_rank[0] == 1 or attn_metadata.is_decoding:
        return

    num_tokens = step_context.input_ids.numel()
    prefix_total = attn_metadata.kv_flatten_size - num_tokens
    prefix_limit = min(prefix_total, max(0, attn_metadata.max_kv_seqlen - 1))
    attn_metadata.dcp_prefill_chunks = build_dcp_prefill_chunks(
        prefix_lens=attn_metadata.kv_seqlens - attn_metadata.q_seqlens,
        prefix_limit=prefix_limit,
        block_size=step_context.cache_config.block_size,
        head_dim=step_context.model_config.head_dim,
        dcp_world_rank=dcp_world_rank,
    )
    topk = step_context.model_config.mla_index_topk
    # Short-context sparse MLA uses dense attention and needs no index mapping.
    if attn_metadata.dcp_prefill_chunks and topk is not None and attn_metadata.max_kv_seqlen > topk:
        attn_metadata.dcp_prefill_request_ids = torch.repeat_interleave(
            torch.arange(attn_metadata.q_seqlens.numel(), dtype=torch.int32, device=attn_metadata.q_seqlens.device),
            attn_metadata.q_seqlens,
            output_size=num_tokens)


def gather_dcp_query(query: torch.Tensor, *, dcp_world_size: int) -> torch.Tensor:
    """Gather [tokens, local_heads, dim] queries along the head axis."""
    if dcp_world_size == 1:
        return query
    from lmdeploy.pytorch.distributed import all_gather_into_tensor

    # Keep the existing packing path for strided queries. Direct gather
    # saves a copy only when the token-major input is already contiguous.
    if not query.is_contiguous():
        transposed = query.transpose(0, 1).contiguous()
        gathered = transposed.new_empty(
            dcp_world_size * transposed.size(0), *transposed.shape[1:])
        all_gather_into_tensor(gathered, transposed, group='dcp')
        return gathered.transpose(0, 1).contiguous()

    gathered = query.new_empty(dcp_world_size * query.size(0), *query.shape[1:])
    all_gather_into_tensor(gathered, query, group='dcp')
    # Gather token-major queries directly, then join rank-local heads.
    # Contiguous inputs need only this final layout conversion.
    gathered = gathered.view(dcp_world_size, *query.shape)
    return gathered.transpose(0, 1).reshape(query.size(0), -1, query.size(2)).contiguous()


def merge_dcp_attention(local_output: torch.Tensor,
                        local_lse: torch.Tensor,
                        valid_rows: torch.Tensor,
                        *,
                        dcp_world_rank: tuple[int, int]) -> torch.Tensor:
    """Merge normalized CUDA shard outputs and scatter heads back to each rank.

    Inputs are [tokens, gathered_heads, value_dim] outputs and natural-log [tokens, gathered_heads] LSE. Invalid rows
    contribute zero. Correction and reduction use FP32; the result retains the input output dtype.
    """
    dcp_world_size, dcp_rank = dcp_world_rank
    if dcp_world_size == 1:
        return local_output
    from lmdeploy.pytorch.distributed import all_gather_into_tensor, reduce_scatter_tensor
    from lmdeploy.pytorch.kernels.cuda.dcp import correct_dcp_attention_output, sanitize_dcp_lse

    local_lse = sanitize_dcp_lse(local_lse, valid_rows)
    gathered_lse = local_lse.new_empty(
        dcp_world_size * local_lse.size(0), local_lse.size(1))
    all_gather_into_tensor(gathered_lse, local_lse, group='dcp')
    gathered_lse = gathered_lse.view(dcp_world_size, *local_lse.shape)
    contribution = correct_dcp_attention_output(
        local_output, gathered_lse, dcp_rank=dcp_rank)

    num_heads = contribution.size(0)
    # The correction kernel writes [heads, tokens, dim] directly so the
    # head-sharded reduce-scatter needs no separate transpose/copy.
    assert num_heads % dcp_world_size == 0
    local_heads = num_heads // dcp_world_size
    scattered_output = contribution.new_empty(local_heads,
                                              contribution.size(1),
                                              contribution.size(2))
    reduce_scatter_tensor(scattered_output, contribution, group='dcp')
    return scattered_output.transpose(0, 1).to(local_output.dtype)
