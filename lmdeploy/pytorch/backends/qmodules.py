# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from .base import BuildSpec


class RMSNormW8A8Impl(ABC):
    """RMS norm w8a8 implementation api."""

    @staticmethod
    def create_weight(hidden_size: int, dtype: torch.dtype = None, device: torch.device = None):
        """Create weight."""
        if dtype is None:
            dtype = torch.float16
        if device is None:
            device = 'cuda'
        weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device), requires_grad=False)
        return weight

    @abstractmethod
    def forward(self, x: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor = None):
        """forward."""
        raise NotImplementedError


class RMSNormW8A8Builder(ABC):
    """RMS norm w8a8 implementation builder."""

    @staticmethod
    @abstractmethod
    def build(hidden_size: int, eps: float = 1e-6, quant_dtype: torch.dtype = torch.int8):
        """build."""
        raise NotImplementedError


class LinearW8A8Impl(ABC):
    """Linear w8a8 implementation api."""

    def update_weights(self, weight: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None = None):
        """Update weights."""
        return weight, scale, bias

    @abstractmethod
    def forward(self,
                x,
                weight: torch.Tensor,
                scale: torch.Tensor,
                bias: torch.Tensor | None = None,
                all_reduce: bool = False,
                group: torch.distributed.ProcessGroup | None = None):
        """forward."""
        raise NotImplementedError


@dataclass(frozen=True)
class LinearW8A8BuildSpec(BuildSpec[LinearW8A8Impl]):
    """Immutable requirements for constructing a W8A8 linear operator."""

    in_features: int
    out_features: int
    bias: bool
    dtype: torch.dtype | None
    quant_dtype: torch.dtype | None
