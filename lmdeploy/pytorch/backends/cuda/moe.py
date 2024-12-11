# Copyright (c) OpenMMLab. All rights reserved.

from typing import List

import torch

from lmdeploy.pytorch.kernels.cuda import fused_moe

from ..moe import FusedMoEBuilder, FusedMoEImpl


class TritonFusedMoEImpl(FusedMoEImpl):
    """triton fused moe implementation."""

    def __init__(self,
                 top_k: int,
                 num_experts: int,
                 renormalize: bool = False):
        self.num_experts = num_experts
        self.top_k = top_k
        self.renormalize = renormalize

    def update_weights(self, gate_up_weights: torch.Tensor,
                       down_weights: torch.Tensor):
        gate_up_weights = gate_up_weights.transpose(1,
                                                    2).contiguous().transpose(
                                                        1, 2)
        down_weights = down_weights.transpose(1,
                                              2).contiguous().transpose(1, 2)
        return gate_up_weights, down_weights

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
        expert_offset = 0
        num_experts = None
        if expert_list is not None and len(expert_list) != self.num_experts:
            expert_offset = expert_list[0]
            num_experts = self.num_experts
        return fused_moe(hidden_states,
                         gate_up_weights,
                         down_weights,
                         topk_weights=topk_weights,
                         topk_ids=topk_ids,
                         topk=self.top_k,
                         expert_offset=expert_offset,
                         num_experts=num_experts,
                         renormalize=self.renormalize)


class TritonFusedMoEBuilder(FusedMoEBuilder):
    """triton fused moe builder."""

    @staticmethod
    def build(top_k: int, num_experts: int, renormalize: bool = False):
        """build from mlp."""
        return TritonFusedMoEImpl(top_k=top_k,
                                  num_experts=num_experts,
                                  renormalize=renormalize)
