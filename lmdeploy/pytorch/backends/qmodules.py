# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import Optional

import torch


class RMSNormW8A8Impl(ABC):
    """RMS norm w8a8 implementation api."""

    @staticmethod
    def create_weight(hidden_size: int,
                      dtype: torch.dtype = None,
                      device: torch.device = None):
        """create weight."""
        if dtype is None:
            dtype = torch.float16
        if device is None:
            device = 'cuda'
        weight = torch.nn.Parameter(torch.ones(hidden_size,
                                               dtype=dtype,
                                               device=device),
                                    requires_grad=False)
        return weight

    @abstractmethod
    def forward(self,
                x: torch.Tensor,
                weight: torch.Tensor,
                residual: torch.Tensor = None):
        """forward."""
        raise NotImplementedError


class RMSNormW8A8Builder(ABC):
    """RMS norm w8a8 implementation builder."""

    @staticmethod
    @abstractmethod
    def build(hidden_size: int, eps: float = 1e-6):
        """build."""
        raise NotImplementedError


class LinearW8A8Impl(ABC):
    """linear w8a8 implementation api."""

    def update_weights(self,
                       weight: torch.Tensor,
                       scale: torch.Tensor,
                       bias: Optional[torch.Tensor] = None):
        """update weights."""
        return weight, scale, bias

    @abstractmethod
    def forward(self,
                x,
                weight: torch.Tensor,
                scale: torch.Tensor,
                bias: Optional[torch.Tensor] = None,
                all_reduce: bool = False):
        """forward."""
        raise NotImplementedError


class LinearW8A8Builder(ABC):
    """linear w8a8 implementation builder."""

    @staticmethod
    @abstractmethod
    def build(in_features: int,
              out_features: int,
              bias: bool = True,
              dtype: torch.dtype = None):
        """build."""
        raise NotImplementedError
