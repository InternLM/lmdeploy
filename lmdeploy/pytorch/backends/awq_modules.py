# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import Optional

import torch


class LinearW4A16Impl(ABC):
    """w4a16 linear implementation."""

    def update_weights(self,
                       qweight: torch.Tensor,
                       scales: torch.Tensor,
                       qzeros: torch.Tensor,
                       bias: Optional[torch.Tensor] = None):
        """update weights."""
        return qweight, scales, qzeros, bias

    @abstractmethod
    def forward(self,
                x,
                weight: torch.Tensor,
                bias: Optional[torch.Tensor] = None,
                all_reduce: bool = False):
        """forward."""
        raise NotImplementedError


class LinearW4A16Builder(ABC):
    """w4a16 linear implementation builder."""

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
