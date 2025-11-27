# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import List, Optional

import torch
import torch.distributed as dist


class LinearImpl(ABC):
    """Linear implementation api."""

    def update_weights(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """Update weights."""
        return weight, bias

    @abstractmethod
    def forward(self,
                x,
                weight: torch.Tensor,
                bias: Optional[torch.Tensor] = None,
                all_reduce: bool = False,
                group: dist.ProcessGroup = None,
                rank: int = 0,
                scatter_size: List[int] = None):
        """forward."""
        raise NotImplementedError


class LinearBuilder(ABC):
    """Linear implementation builder."""

    @staticmethod
    @abstractmethod
    def build(in_features: int, out_features: int, bias: bool = True, dtype: torch.dtype = None):
        """build."""
        raise NotImplementedError
