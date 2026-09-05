# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from .base import BuildSpec


class LinearW4A16Impl(ABC):
    """W4a16 linear implementation."""

    def update_weights(self,
                       qweight: torch.Tensor,
                       scales: torch.Tensor,
                       qzeros: torch.Tensor,
                       bias: torch.Tensor | None = None):
        """Update weights."""
        return qweight, scales, qzeros, bias

    @abstractmethod
    def forward(self,
                x,
                weight: torch.Tensor,
                bias: torch.Tensor | None = None,
                all_reduce: bool = False,
                group: torch.distributed.ProcessGroup | None = None):
        """forward."""
        raise NotImplementedError


@dataclass(frozen=True)
class LinearW4A16BuildSpec(BuildSpec[LinearW4A16Impl]):
    """Immutable requirements for constructing a W4A16 linear operator."""

    in_features: int
    out_features: int
    w_bit: int
    group_size: int
    bias: bool
    output_dtype: torch.dtype | None
