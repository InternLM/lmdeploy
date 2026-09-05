# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .base import BuildSpec


class LinearImpl(ABC):
    """Linear implementation api."""

    def update_weights(self, weight: torch.Tensor, bias: torch.Tensor | None = None):
        """Update weights."""
        return weight, bias

    @abstractmethod
    def forward(self,
                x,
                weight: torch.Tensor,
                bias: torch.Tensor | None = None,
                all_reduce: bool = False,
                group: dist.ProcessGroup = None,
                rank: int = 0,
                scatter_size: list[int] = None):
        """forward."""
        raise NotImplementedError


@dataclass(frozen=True)
class LinearBuildSpec(BuildSpec[LinearImpl]):
    """Immutable requirements for constructing an unquantized linear op."""

    in_features: int
    out_features: int
    bias: bool
    dtype: torch.dtype | None
