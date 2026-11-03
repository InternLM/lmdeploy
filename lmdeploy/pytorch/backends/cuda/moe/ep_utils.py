# Copyright (c) OpenMMLab. All rights reserved.

import torch
from torch import distributed as dist

from lmdeploy.pytorch.distributed import get_dist_manager


def mask_invalid_tokens(topk_idx: torch.Tensor) -> None:
    """Mask dummy/padded LL routes; TP callers must also mask before
    slicing."""
    from lmdeploy.pytorch.model_inputs import get_step_ctx_manager

    manager = get_step_ctx_manager()
    context = manager.current_context() if manager is not None else None
    if context is None:
        return
    if context.moe_valid_token_count is not None:
        from lmdeploy.pytorch.kernels.cuda.moe.route_mask import mask_invalid_routes

        mask_invalid_routes(topk_idx, context.moe_valid_token_count)
    elif context.is_dummy:
        # Eager dummy steps still join the collective, but send no routes.
        topk_idx.fill_(-1)


def split_inputs_by_attn_tp(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    mask_padding: bool = False,
):
    """Split input by attn tp."""
    dist_ctx = get_dist_manager().current_context()
    attn_tp = dist_ctx.dist_config.attn_tp
    attn_rank = dist_ctx.attn_tp_group.rank
    num_states = hidden_states.size(0)

    if attn_tp == 1 or attn_tp > num_states:
        return hidden_states, topk_weights, topk_ids, None

    if mask_padding:
        # The graph count is global to the local DP batch. Filter before TP
        # slicing so every slice retains the original token coordinates.
        # LL dispatch also filters unsplit callers; existing -1 IDs stay masked.
        mask_invalid_tokens(topk_ids)

    # split size
    base = num_states // attn_tp
    remain = num_states % attn_tp
    split_size = [base + 1] * remain + [base] * (attn_tp - remain)

    # split inputs
    hidden_states = torch.split(hidden_states, split_size, dim=0)[attn_rank]
    topk_weights = torch.split(topk_weights, split_size, dim=0)[attn_rank]
    topk_ids = torch.split(topk_ids, split_size, dim=0)[attn_rank]

    return hidden_states, topk_weights, topk_ids, split_size


def gather_outputs_by_attn_tp(out_states: torch.Tensor, split_size: list[int]):
    """Gather output by attn tp."""
    if split_size is None:
        return out_states

    dist_ctx = get_dist_manager().current_context()
    gpu_group = dist_ctx.attn_tp_group.gpu_group
    new_out_states = out_states.new_empty((sum(split_size), out_states.shape[1]))
    new_out_states_list = list(new_out_states.split(split_size, dim=0))
    dist.all_gather(new_out_states_list, out_states, group=gpu_group)
    return new_out_states
