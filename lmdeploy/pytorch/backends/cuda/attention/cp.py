# Copyright (c) OpenMMLab. All rights reserved.
"""CUDA DCP query gathering and attention-output merging."""

import torch


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
                        valid_counts: torch.Tensor,
                        *,
                        dcp_world_rank: tuple[int, int]) -> torch.Tensor:
    """Merge normalized CUDA shard outputs and scatter heads back to each rank.

    Inputs are [tokens, gathered_heads, value_dim] outputs and natural-log [tokens, gathered_heads] LSE. Empty rows
    contribute zero. LSE and correction arithmetic use FP32; reduce-scatter uses the input output dtype.
    """
    dcp_world_size, dcp_rank = dcp_world_rank
    if dcp_world_size == 1:
        return local_output
    from lmdeploy.pytorch.distributed import all_gather_into_tensor, reduce_scatter_tensor
    from lmdeploy.pytorch.kernels.cuda.dcp import correct_dcp_attention_output, sanitize_dcp_lse

    local_lse = sanitize_dcp_lse(local_lse, valid_counts)
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
    return scattered_output.transpose(0, 1)
