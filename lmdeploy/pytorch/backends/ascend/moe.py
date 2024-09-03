# Copyright (c) OpenMMLab. All rights reserved.

import torch

from lmdeploy.pytorch.kernels.ascend import moe_gating_topk_softmax

from ..moe import SoftmaxTopKBuilder, SoftmaxTopKImpl


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
