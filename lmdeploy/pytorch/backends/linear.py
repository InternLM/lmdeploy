# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import Optional

import torch


class LinearImpl(ABC):
    """Linear implementation api."""

    def update_weights(self,
                       weight: torch.Tensor,
                       bias: Optional[torch.Tensor] = None):
        """update weights."""
        return weight, bias

    @abstractmethod
    def forward(self,
                x,
                weight: torch.Tensor,
                bias: Optional[torch.Tensor] = None,
                all_reduce: bool = False):
        """forward."""
        raise NotImplementedError


class LinearBuilder(ABC):
    """linear implementation builder."""

    @staticmethod
    @abstractmethod
    def build(in_features: int,
              out_features: int,
              bias: bool = True,
              dtype: torch.dtype = None):
        """build."""
        raise NotImplementedError
