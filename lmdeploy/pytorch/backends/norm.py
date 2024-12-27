# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod

import torch


class RMSNormImpl(ABC):
    """RMS norm implementation api."""

    @abstractmethod
    def forward(self,
                x: torch.Tensor,
                weight: torch.Tensor,
                residual: torch.Tensor = None):
        """forward."""
        raise NotImplementedError


class RMSNormBuilder(ABC):
    """RMS norm implementation builder."""

    @staticmethod
    @abstractmethod
    def build(hidden_size: int, eps: float = 1e-6):
        """build."""
        raise NotImplementedError


class LayerNormImpl(ABC):
    """Layer norm implementation api."""

    @abstractmethod
    def forward(self,
                x: torch.Tensor,
                weight: torch.Tensor,
                bias: torch.Tensor = None,
                residual: torch.Tensor = None):
        """forward."""
        raise NotImplementedError


class LayerNormBuilder(ABC):
    """layer norm implementation builder."""

    @staticmethod
    @abstractmethod
    def build(normalized_shape: int, eps: float = 1e-6):
        """build."""
        raise NotImplementedError
