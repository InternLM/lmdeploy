# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from lmdeploy.pytorch.kernels.maca import moe_topk_softmax
from lmdeploy.pytorch.kernels.maca import silu_and_mul

from ..moe import (FusedMoEBuilder, FusedMoEImpl, SoftmaxTopKBuilder,
                   SoftmaxTopKImpl)


class MacaSoftmaxTopKImpl(SoftmaxTopKImpl):
    """maca softmax topk implementation."""

    def __init__(self, top_k: int, dim: int = -1):
        self.top_k = top_k
        self.dim = dim

    def forward(self, x: torch.Tensor):
        routing_weights, selected_experts = moe_topk_softmax(
            x, self.top_k)
        return routing_weights.to(torch.float32), selected_experts.to(
            torch.int64)


class MacaSoftmaxTopKBuilder(SoftmaxTopKBuilder):
    """maca softmax topk implementation builder."""

    @staticmethod
    def build(top_k: int, dim: int = -1):
        """build."""
        return MacaSoftmaxTopKImpl(top_k, dim)


class MacaFusedMoEImpl(FusedMoEImpl):
    """maca fused moe implementation."""

    def __init__(self, top_k: int, renormalize: bool = False):
        self.top_k = top_k
        self.renormalize = renormalize

    def forward(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor, gate_up_weights: torch.Tensor,
                down_weights: torch.Tensor):
        """forward."""
        N, D= hidden_states.shape
        hidden_states = hidden_states.view(N, -1, D).repeat(1, self.top_k, 1).reshape(-1, D)
        out = torch.zeros(N * self.top_k, down_weights.shape[1], dtype=hidden_states.dtype, device=hidden_states.device)
        for i in range(gate_up_weights.shape[0]):
            mask = topk_ids == i
            if mask.sum():
                out[mask] = silu_and_mul(hidden_states[mask] @ gate_up_weights[i].transpose(0, 1)) @ down_weights[i].transpose(0, 1)
        return (out.view(N, -1, down_weights.shape[1]) * topk_weights.view(N, -1, 1).to(out.dtype)).sum(dim=1)

class MacaFusedMoEBuilder(FusedMoEBuilder):
    """maca fused moe builder."""

    @staticmethod
    def build(top_k: int, renormalize: bool = False):
        """build from mlp."""
        return MacaFusedMoEImpl(top_k=top_k, renormalize=renormalize)
