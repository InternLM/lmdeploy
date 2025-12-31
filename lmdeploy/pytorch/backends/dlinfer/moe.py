# Copyright (c) OpenMMLab. All rights reserved.

from dataclasses import dataclass
from typing import Callable, List

import torch

from lmdeploy.pytorch.distributed import get_dist_manager
from lmdeploy.pytorch.kernels.dlinfer import DlinferDistContext, fused_moe, moe_gating_topk_softmax

from ..moe import FusedMoEBuilder, FusedMoEImpl, SoftmaxTopKBuilder, SoftmaxTopKImpl


def get_dist_ctx():
    dist_ctx = get_dist_manager().current_context()
    
    return DlinferDistContext(dp_size = dist_ctx.dist_config.dp,
                              tp_size = dist_ctx.dist_config.tp,
                              ep_size = dist_ctx.dist_config.ep,
                              dp_rank = dist_ctx.dp_rank,
                              tp_rank = dist_ctx.attn_tp_group.rank,
                              ep_rank = dist_ctx.ep_rank,
                              max_tokens_accros_dp = 1,
                              tp_group = dist_ctx.attn_tp_group.gpu_group,
                              ep_group = dist_ctx.ep_gpu_group)


class DlinferSoftmaxTopKImpl(SoftmaxTopKImpl):
    """Dlinfer softmax topk implementation."""

    def __init__(self, top_k: int, dim: int = -1, n_groups: int = -1):
        self.top_k = top_k
        self.dim = dim
        if n_groups != -1:
            raise NotImplementedError('Group router not supported')
        self.dist_ctx = get_dist_ctx()

    def forward(self, x: torch.Tensor):
        routing_weights, selected_experts = moe_gating_topk_softmax(x, self.top_k, self.dist_ctx)
        return routing_weights, selected_experts


class DlinferSoftmaxTopKBuilder(SoftmaxTopKBuilder):
    """Dlinfer softmax topk implementation builder."""

    @staticmethod
    def build(top_k: int, dim: int = -1, n_groups: int = -1):
        """build."""
        return DlinferSoftmaxTopKImpl(top_k, dim, n_groups)


class DlinferFusedMoEImpl(FusedMoEImpl):
    """Dlinfer fused moe implementation."""

    def __init__(self,
                 top_k: int,
                 num_experts: int,
                 renormalize: bool = False,
                 ep_size: int = 1,
                 ep_group: torch.distributed.ProcessGroup = None):
        self.top_k = top_k
        self.num_experts = num_experts
        self.renormalize = renormalize
        self.ep_size = ep_size
        self.ep_group = ep_group
        self.dist_ctx = get_dist_ctx()

    def update_weights(self, gate_up_weights: torch.Tensor, down_weights: torch.Tensor):
        """Update weights."""
        device_type = gate_up_weights.device.type
        if device_type in ['npu']:
            return gate_up_weights.transpose(-1, -2).contiguous(), down_weights.transpose(-1, -2).contiguous()
        return gate_up_weights, down_weights

    def ep_expert_list(self, world_size: int, rank: int):
        """Experts list of current rank."""
        num_experts = self.num_experts
        expert_per_rank = (num_experts + world_size - 1) // world_size
        first_expert = rank * expert_per_rank
        last_expert = min(first_expert + expert_per_rank, num_experts)
        return list(range(first_expert, last_expert))

    def forward(self,
                hidden_states: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor,
                gate_up_weights: torch.Tensor,
                down_weights: torch.Tensor,
                gate_up_bias: torch.Tensor = None,
                down_bias: torch.Tensor = None,
                expert_list: List[int] = None,
                act_func: Callable = None):
        """forward."""
        assert gate_up_bias is None
        assert down_bias is None
        return fused_moe(hidden_states, gate_up_weights, down_weights, topk_weights, topk_ids, self.top_k,
                         self.renormalize, self.dist_ctx)


class DlinferFusedMoEBuilder(FusedMoEBuilder):
    """Dlinfer fused moe builder."""

    @staticmethod
    def build(top_k: int,
              num_experts: int,
              renormalize: bool = False,
              hidden_dim: int = 1,
              ep_size: int = 1,
              ep_group: torch.distributed.ProcessGroup = None,
              layer_idx: int = 0,
              out_dtype: torch.dtype = torch.bfloat16):
        """Build from mlp."""
        return DlinferFusedMoEImpl(top_k=top_k,
                                   num_experts=num_experts,
                                   renormalize=renormalize,
                                   ep_size=ep_size,
                                   ep_group=ep_group)
