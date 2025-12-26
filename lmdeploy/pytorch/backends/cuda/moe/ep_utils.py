# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
from torch import distributed as dist

from lmdeploy.pytorch.distributed import get_dist_manager


def split_inputs_by_attn_tp(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
):
    """Split input by attn tp."""
    dist_ctx = get_dist_manager().current_context()
    attn_tp = dist_ctx.dist_config.attn_tp
    attn_rank = dist_ctx.attn_tp_group.rank
    num_states = hidden_states.size(0)

    if attn_tp == 1 or attn_tp > num_states:
        return hidden_states, topk_weights, topk_ids, None

    # split size
    base = num_states // attn_tp
    remain = num_states % attn_tp
    split_size = [base + 1] * remain + [base] * (attn_tp - remain)

    # split inputs
    hidden_states = torch.split(hidden_states, split_size, dim=0)[attn_rank]
    topk_weights = torch.split(topk_weights, split_size, dim=0)[attn_rank]
    topk_ids = torch.split(topk_ids, split_size, dim=0)[attn_rank]

    return hidden_states, topk_weights, topk_ids, split_size


def gather_outputs_by_attn_tp(out_states: torch.Tensor, split_size: List[int]):
    """Gather output by attn tp."""
    if split_size is None:
        return out_states

    dist_ctx = get_dist_manager().current_context()
    gpu_group = dist_ctx.attn_tp_group.gpu_group
    new_out_states = out_states.new_empty((sum(split_size), out_states.shape[1]))
    new_out_states_list = list(new_out_states.split(split_size, dim=0))
    dist.all_gather(new_out_states_list, out_states, group=gpu_group)
    return new_out_states
