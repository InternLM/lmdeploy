# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from .base import BuildSpec


class RMSNormImpl(ABC):
    """RMS norm implementation api."""

    @abstractmethod
    def forward(self, x: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor = None):
        """forward."""
        raise NotImplementedError


@dataclass(frozen=True)
class RMSNormBuildSpec(BuildSpec[RMSNormImpl]):
    """Immutable requirements for constructing an RMS norm operator."""

    hidden_size: int
    eps: float = 1e-6


class LayerNormImpl(ABC):
    """Layer norm implementation api."""

    @abstractmethod
    def forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, residual: torch.Tensor = None):
        """forward."""
        raise NotImplementedError


@dataclass(frozen=True)
class LayerNormBuildSpec(BuildSpec[LayerNormImpl]):
    """Immutable requirements for constructing a layer norm operator."""

    normalized_shape: int
    eps: float = 1e-6
