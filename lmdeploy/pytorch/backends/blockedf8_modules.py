# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import List, Optional

import torch


class LinearBlockedF8Impl(ABC):
    """Linear BlockedF8 implementation api."""

    def update_weights(self, weight: torch.Tensor, scale: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """Update weights."""
        return weight, scale, bias

    @abstractmethod
    def forward(self,
                x,
                weight: torch.Tensor,
                scale: torch.Tensor,
                bias: Optional[torch.Tensor] = None,
                all_reduce: bool = False,
                rank: int = 0,
                scatter_size: List[int] = None):
        """forward."""
        raise NotImplementedError


class LinearBlockedF8Builder(ABC):
    """Linear BlockedF8 implementation builder."""

    @staticmethod
    @abstractmethod
    def build(in_features: int, out_features: int, bias: bool = True, dtype: torch.dtype = None):
        """build."""
        raise NotImplementedError
