# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .base import BuildSpec


class LinearBlockedF8Impl(ABC):
    """Linear BlockedF8 implementation api."""

    def __init__(self):
        self.scale_fmt: str | None = None

    def update_weights(self, weight: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None = None):
        """Update weights."""
        return weight, scale, bias

    def set_scale_fmt(self, scale_fmt: str | None):
        """Set scale fmt."""
        self.scale_fmt = scale_fmt

    @abstractmethod
    def forward(self,
                x,
                weight: torch.Tensor,
                scale: torch.Tensor,
                bias: torch.Tensor | None = None,
                all_reduce: bool = False,
                group: dist.ProcessGroup | None = None,
                rank: int = 0,
                scatter_size: list[int] = None):
        """forward."""
        raise NotImplementedError


@dataclass(frozen=True)
class LinearBlockedF8BuildSpec(BuildSpec[LinearBlockedF8Impl]):
    """Immutable requirements for constructing a blocked-FP8 linear
    operator."""

    in_features: int
    out_features: int
    block_size: int
    bias: bool
    output_dtype: torch.dtype | None
    fp8_dtype: torch.dtype
    scale_fmt: str | None
