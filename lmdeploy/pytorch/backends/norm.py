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
    def build(hidden_size: int, eps: float = 1e-6, inplace: bool = False):
        """build."""
        raise NotImplementedError
