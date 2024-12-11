# Copyright (c) OpenMMLab. All rights reserved.

from typing import List

import torch

from lmdeploy.pytorch.kernels.dlinfer import fused_moe, moe_gating_topk_softmax

from ..moe import (FusedMoEBuilder, FusedMoEImpl, SoftmaxTopKBuilder,
                   SoftmaxTopKImpl)


class DlinferSoftmaxTopKImpl(SoftmaxTopKImpl):
    """dlinfer softmax topk implementation."""

    def __init__(self, top_k: int, dim: int = -1):
        self.top_k = top_k
        self.dim = dim

    def forward(self, x: torch.Tensor):
        routing_weights, selected_experts = moe_gating_topk_softmax(
            x.to(torch.float32), self.top_k)
        return routing_weights, selected_experts


class DlinferSoftmaxTopKBuilder(SoftmaxTopKBuilder):
    """dlinfer softmax topk implementation builder."""

    @staticmethod
    def build(top_k: int, dim: int = -1):
        """build."""
        return DlinferSoftmaxTopKImpl(top_k, dim)


class DlinferFusedMoEImpl(FusedMoEImpl):
    """dlinfer fused moe implementation."""

    def __init__(self, top_k: int, renormalize: bool = False):
        self.top_k = top_k
        self.renormalize = renormalize

    def forward(self,
                hidden_states: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor,
                gate_up_weights: torch.Tensor,
                down_weights: torch.Tensor,
                expert_list: List[int] = None):
        """forward."""
        return fused_moe(hidden_states, self.top_k, topk_ids, topk_weights,
                         gate_up_weights, down_weights)


class DlinferFusedMoEBuilder(FusedMoEBuilder):
    """dlinfer fused moe builder."""

    @staticmethod
    def build(top_k: int, num_experts: int, renormalize: bool = False):
        """build from mlp."""
        return DlinferFusedMoEImpl(top_k=top_k, renormalize=renormalize)
