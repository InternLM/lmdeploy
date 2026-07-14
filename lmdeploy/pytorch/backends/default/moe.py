# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..moe import SoftmaxTopKBuilder, SoftmaxTopKImpl


class DefaultSoftmaxTopKImpl(SoftmaxTopKImpl):
    """RMS norm implementation api."""

    def __init__(self, top_k: int, dim: int = -1, n_groups: int = -1):
        self.top_k = top_k
        self.dim = dim
        self.n_groups = n_groups
        assert self.top_k % self.n_groups == 0, f'{self.top_k} cannot be divided by {self.n_groups}'

    def forward(self, x: torch.Tensor):
        """forward."""
        routing_weights = torch.softmax(x, dim=self.dim, dtype=torch.float32)
        if self.n_groups > 0:
            assert routing_weights.shape[
                self.
                dim] % self.n_groups == 0, f'{routing_weights.shape[self.dim]} cannot be divided by {self.n_groups}'
            per_group_top_k = self.top_k // self.n_groups
            group_size = routing_weights.shape[self.dim] // self.n_groups
            group_offsets = self.get_group_offsets(self.n_groups, group_size, routing_weights.device)
            routing_weights = routing_weights.unflatten(self.dim, (self.n_groups, group_size))
            topk_weights, topk_ids = torch.topk(routing_weights, per_group_top_k, dim=-1)
            topk_ids = (topk_ids + group_offsets).flatten(-2, -1)
            topk_weights = topk_weights.flatten(-2, -1)
        else:
            topk_weights, topk_ids = torch.topk(routing_weights, self.top_k, dim=self.dim)
        return topk_weights, topk_ids


class DefaultSoftmaxTopKBuilder(SoftmaxTopKBuilder):
    """RMS norm implementation builder."""

    @staticmethod
    def build(top_k: int, dim: int = -1, n_groups: int = -1):
        """build."""
        return DefaultSoftmaxTopKImpl(top_k, dim, n_groups=n_groups)
