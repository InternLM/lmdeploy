# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import List, Optional

import torch
import torch.distributed as dist


class LinearBlockedF8Impl(ABC):
    """Linear BlockedF8 implementation api."""

    def __init__(self):
        self.scale_fmt: Optional[str] = None

    def update_weights(self, weight: torch.Tensor, scale: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """Update weights."""
        return weight, scale, bias

    def set_scale_fmt(self, scale_fmt: Optional[str]):
        """Set scale fmt."""
        self.scale_fmt = scale_fmt

    @abstractmethod
    def forward(self,
                x,
                weight: torch.Tensor,
                scale: torch.Tensor,
                bias: Optional[torch.Tensor] = None,
                all_reduce: bool = False,
                group: Optional[dist.ProcessGroup] = None,
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
