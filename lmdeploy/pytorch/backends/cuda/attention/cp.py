# Copyright (c) OpenMMLab. All rights reserved.
"""CUDA DCP query gathering and attention-output merging."""

import torch


def gather_dcp_query(query: torch.Tensor, *, dcp_world_size: int) -> torch.Tensor:
    """Gather [tokens, local_heads, dim] queries along the head axis.

    Consume the result on the same stream before the next query gather; the optimized path borrows a shared arena.
    """
    if dcp_world_size == 1:
        return query
    from lmdeploy.pytorch.distributed import get_dist_manager

    communicator = get_dist_manager().current_context().dcp_group.communicator
    return communicator.gather_query(query)


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
