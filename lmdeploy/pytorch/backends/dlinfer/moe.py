# Copyright (c) OpenMMLab. All rights reserved.

from typing import List

import torch

from lmdeploy.pytorch.kernels.dlinfer import fused_moe, moe_gating_topk_softmax

from ..moe import FusedMoEBuilder, FusedMoEImpl, SoftmaxTopKBuilder, SoftmaxTopKImpl


class DlinferSoftmaxTopKImpl(SoftmaxTopKImpl):
    """dlinfer softmax topk implementation."""

    def __init__(self, top_k: int, dim: int = -1):
        self.top_k = top_k
        self.dim = dim

    def forward(self, x: torch.Tensor):
        routing_weights, selected_experts = moe_gating_topk_softmax(x.to(torch.float32), self.top_k)
        return routing_weights, selected_experts


class DlinferSoftmaxTopKBuilder(SoftmaxTopKBuilder):
    """dlinfer softmax topk implementation builder."""

    @staticmethod
    def build(top_k: int, dim: int = -1):
        """build."""
        return DlinferSoftmaxTopKImpl(top_k, dim)


class DlinferFusedMoEImpl(FusedMoEImpl):
    """dlinfer fused moe implementation."""

    def __init__(self, top_k: int, num_experts: int, renormalize: bool = False):
        self.top_k = top_k
        self.num_experts = num_experts
        self.renormalize = renormalize

    def support_ep(self):
        """support expert parallelism."""
        return True

    def ep_expert_list(self, world_size: int, rank: int):
        """experts list of current rank."""
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
                expert_list: List[int] = None):
        """forward."""
        if expert_list is None:
            return fused_moe(hidden_states, gate_up_weights, down_weights, topk_weights, topk_ids, self.top_k,
                             self.renormalize)
        # support ep.
        expert_offset = 0
        num_experts = None
        if expert_list is not None and len(expert_list) != self.num_experts:
            expert_offset = expert_list[0]
            num_experts = self.num_experts
        from lmdeploy.pytorch.backends.cuda.moe import fused_moe
        out = fused_moe(hidden_states,
                        gate_up_weights,
                        down_weights,
                        topk_weights=topk_weights,
                        topk_ids=topk_ids,
                        topk=self.top_k,
                        expert_offset=expert_offset,
                        num_experts=num_experts,
                        renormalize=self.renormalize)
        return out


class DlinferFusedMoEBuilder(FusedMoEBuilder):
    """dlinfer fused moe builder."""

    @staticmethod
    def build(top_k: int, num_experts: int, renormalize: bool = False):
        """build from mlp."""
        return DlinferFusedMoEImpl(top_k=top_k, num_experts=num_experts, renormalize=renormalize)
