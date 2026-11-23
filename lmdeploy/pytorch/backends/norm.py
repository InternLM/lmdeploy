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


class RMSNormGatedImpl(ABC):
    """Gated RMS normalization using weights owned by the public layer."""

    @abstractmethod
    def forward(self,
                x: torch.Tensor,
                weight: torch.Tensor | None,
                z: torch.Tensor,
                residual: torch.Tensor | None = None,
                prenorm: bool = False,
                residual_in_fp32: bool = False):
        """Normalize, apply the scale, then apply the gate activation."""
        raise NotImplementedError


@dataclass(frozen=True)
class RMSNormGatedBuildSpec(BuildSpec[RMSNormGatedImpl]):
    """Requirements for a gated RMS normalization implementation."""

    hidden_size: int
    eps: float = 1e-6
    activation: str = 'silu'

    def __post_init__(self):
        if self.hidden_size <= 0:
            raise ValueError('hidden_size must be positive.')
        if self.activation not in ('silu', 'swish', 'sigmoid'):
            raise ValueError(f'Unsupported gated RMSNorm activation: {self.activation}')
