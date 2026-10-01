# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor

from .base import BuildSpec


class ApplyRotaryEmbImpl(ABC):
    """Apply rotary embedding implementation."""

    @abstractmethod
    def forward(self,
                query: Tensor,
                key: Tensor,
                cos: Tensor,
                sin: Tensor,
                inplace: bool = True,
                complex_mode: bool = False):
        """forward."""
        raise NotImplementedError


@dataclass(frozen=True)
class ApplyRotaryEmbBuildSpec(BuildSpec[ApplyRotaryEmbImpl]):
    """Request construction of an apply-RoPE operator."""
