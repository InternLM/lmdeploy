# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .base import BuildSpec


class LinearStaticF8Impl(ABC):
    """Static per-tensor FP8 linear implementation API."""

    def update_weights(
        self,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
    ):
        """Update weights."""
        return weight, input_scale, weight_scale, bias

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
        all_reduce: bool = False,
        group: dist.ProcessGroup | None = None,
        rank: int = 0,
        scatter_size: list[int] | None = None,
    ):
        """Run static FP8 linear."""
        raise NotImplementedError


@dataclass(frozen=True)
class LinearStaticF8BuildSpec(BuildSpec[LinearStaticF8Impl]):
    """Immutable requirements for constructing a static-FP8 linear operator."""

    in_features: int
    out_features: int
    bias: bool
    dtype: torch.dtype | None
    fp8_dtype: torch.dtype
