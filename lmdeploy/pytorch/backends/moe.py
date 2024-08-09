# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import List

import torch
from torch import nn


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


class FusedMoEImpl(ABC, nn.Module):
    """fused moe implementation."""

    @abstractmethod
    def forward(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor):
        """forward."""
        raise NotImplementedError


class FusedMoEBuilder(ABC):
    """fused moe builder."""

    @staticmethod
    @abstractmethod
    def build_from_mlp(gates: List[torch.Tensor],
                       ups: List[torch.Tensor],
                       downs: List[torch.Tensor],
                       top_k: int,
                       renormalize: bool = False):
        """build from mlp."""
        raise NotImplementedError
