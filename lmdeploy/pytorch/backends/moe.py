# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod

import torch


class SoftmaxTopKImpl(ABC):
    """Softmax topk implementation api."""

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """forward."""
        raise NotImplementedError


class SoftmaxTopKBuilder(ABC):
    """Softmax topk implementation builder."""

    @staticmethod
    @abstractmethod
    def build(top_k: int, dim: int = -1):
        """build."""
        raise NotImplementedError


class FusedMoEImpl(ABC):
    """fused moe implementation."""

    def update_weights(self, gate_up_weights: torch.Tensor,
                       down_weights: torch.Tensor):
        """update weights."""
        return gate_up_weights, down_weights

    @abstractmethod
    def forward(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor, gate_up_weights: torch.Tensor,
                down_weights: torch.Tensor):
        """forward."""
        raise NotImplementedError


class FusedMoEBuilder(ABC):
    """fused moe builder."""

    @staticmethod
    @abstractmethod
    def build(top_k: int, renormalize: bool = False):
        """build from mlp."""
        raise NotImplementedError
