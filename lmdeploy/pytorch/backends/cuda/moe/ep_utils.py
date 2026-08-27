# Copyright (c) OpenMMLab. All rights reserved.

import torch
from torch import distributed as dist

from lmdeploy.pytorch.distributed import get_dist_manager


class DeepEPMoEPaddedAdapter:
    """Bind a raw-token DeepEP MoE result into bucket-shaped stable storage.

    ``DeepEPTokenDispatcherNormal.combine`` returns ``hidden_states.view(self.hidden_shape)``,
    a view of the process-wide DeepEP combine buffer that the next layer's combine overwrites.
    Unlike :class:`PaddedTensorOutputAdapter`, this adapter accepts that view and copies it into
    a stable bridge so the following graph piece can consume a fixed address.
    """

    def __init__(self, token_axis: int = 0) -> None:
        self.token_axis = token_axis

    def allocate(self, output, boundary_input_storages, bridge_pool=None):
        from lmdeploy.pytorch.backends.cuda.graph_runner.piecewise import (
            UnsupportedBoundaryError,
            get_piecewise_graph_execution,
        )
        execution = get_piecewise_graph_execution()
        if execution is None:
            raise UnsupportedBoundaryError('DeepEP MoE adapter requires an active piecewise execution')
        if not isinstance(output, torch.Tensor):
            raise UnsupportedBoundaryError('DeepEP MoE adapter requires one tensor output')
        if output.layout is not torch.strided:
            raise UnsupportedBoundaryError(f'only strided tensor outputs are supported, got {output.layout}')
        if output.ndim == 0 or not -output.ndim <= self.token_axis < output.ndim:
            raise UnsupportedBoundaryError('DeepEP MoE adapter has an invalid token axis')
        token_axis = self.token_axis % output.ndim
        if output.size(token_axis) != execution.raw_tokens:
            raise UnsupportedBoundaryError('eager output does not match the active raw-token extent')
        shape = list(output.shape)
        shape[token_axis] = execution.token_bucket
        if bridge_pool is None:
            return output.new_empty(tuple(shape))
        return bridge_pool.allocate_padded_tensor(output, shape, token_axis)

    def copy(self, destination, source):
        from lmdeploy.pytorch.backends.cuda.graph_runner.piecewise import get_piecewise_graph_execution
        execution = get_piecewise_graph_execution()
        token_axis = self.token_axis % destination.ndim
        destination.narrow(token_axis, 0, execution.raw_tokens).copy_(source)
        if execution.raw_tokens < execution.token_bucket:
            destination.narrow(token_axis, execution.raw_tokens,
                               execution.token_bucket - execution.raw_tokens).zero_()


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
