# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod

import torch


class LinearW4A16Impl(ABC):
    """W4a16 linear implementation."""

    def update_weights(self,
                       qweight: torch.Tensor,
                       scales: torch.Tensor,
                       qzeros: torch.Tensor,
                       bias: torch.Tensor | None = None):
        """Update weights."""
        return qweight, scales, qzeros, bias

    @abstractmethod
    def forward(self,
                x,
                weight: torch.Tensor,
                bias: torch.Tensor | None = None,
                all_reduce: bool = False,
                group: torch.distributed.ProcessGroup | None = None):
        """forward."""
        raise NotImplementedError


class LinearW4A16Builder(ABC):
    """W4a16 linear implementation builder."""

    @staticmethod
    @abstractmethod
    def build(in_features: int,
              out_features: int,
              w_bit: int,
              group_size: int,
              bias: bool = False,
              dtype: torch.dtype = None):
        """build."""
        raise NotImplementedError
