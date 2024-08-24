# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..moe import SoftmaxTopKBuilder, SoftmaxTopKImpl


class DefaultSoftmaxTopKImpl(SoftmaxTopKImpl):
    """RMS norm implementation api."""

    def __init__(self, top_k: int, dim: int = -1):
        self.top_k = top_k
        self.dim = dim

    def forward(self, x: torch.Tensor):
        """forward."""
        routing_weights = torch.softmax(x, dim=self.dim, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(routing_weights,
                                            self.top_k,
                                            dim=self.dim)
        return topk_weights, topk_ids


class DefaultSoftmaxTopKBuilder(SoftmaxTopKBuilder):
    """RMS norm implementation builder."""

    @staticmethod
    def build(top_k: int, dim: int = -1):
        """build."""
        return DefaultSoftmaxTopKImpl(top_k, dim)
