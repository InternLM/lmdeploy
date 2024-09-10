# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod

from torch import Tensor


class ApplyRotaryEmbImpl(ABC):
    """Apply rotary embedding implementation."""

    @abstractmethod
    def forward(self,
                query: Tensor,
                key: Tensor,
                cos: Tensor,
                sin: Tensor,
                inplace: bool = True):
        """forward."""
        raise NotImplementedError


class ApplyRotaryEmbBuilder(ABC):
    """Apply rotary embedding implementation builder."""

    @staticmethod
    @abstractmethod
    def build():
        """build implementation."""
        raise NotImplementedError
