# Copyright (c) OpenMMLab. All rights reserved.

import torch

from lmdeploy.pytorch.kernels.ascend import fused_moe, moe_gating_topk_softmax

from ..moe import (FusedMoEBuilder, FusedMoEImpl, SoftmaxTopKBuilder,
                   SoftmaxTopKImpl)


class AscendSoftmaxTopKImpl(SoftmaxTopKImpl):
    """ascend softmax topk implementation."""

    def __init__(self, top_k: int, dim: int = -1):
        self.top_k = top_k
        self.dim = dim

    def forward(self, x: torch.Tensor):
        routing_weights, selected_experts = moe_gating_topk_softmax(
            x, self.top_k)
        return routing_weights.to(torch.float32), selected_experts.to(
            torch.int64)


class AscendSoftmaxTopKBuilder(SoftmaxTopKBuilder):
    """ascend softmax topk implementation builder."""

    @staticmethod
    def build(top_k: int, dim: int = -1):
        """build."""
        return AscendSoftmaxTopKImpl(top_k, dim)


class AscendFusedMoEImpl(FusedMoEImpl):
    """ascend fused moe implementation."""

    def __init__(self, top_k: int, renormalize: bool = False):
        self.top_k = top_k
        self.renormalize = renormalize

    def forward(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor, gate_up_weights: torch.Tensor,
                down_weights: torch.Tensor):
        """forward."""
        return fused_moe(hidden_states, self.top_k, topk_ids, topk_weights,
                         gate_up_weights, down_weights)


class AscendFusedMoEBuilder(FusedMoEBuilder):
    """ascend fused moe builder."""

    @staticmethod
    def build(top_k: int, renormalize: bool = False):
        """build from mlp."""
        return AscendFusedMoEImpl(top_k=top_k, renormalize=renormalize)
