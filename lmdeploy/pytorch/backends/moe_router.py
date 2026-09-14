# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from .base import BuildSpec


class RouterGemmImpl(ABC):
    """Router GEMM implementation api."""

    def __init__(self, out_dtype: torch.dtype | None = None):
        self.out_dtype = out_dtype

    @abstractmethod
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """forward."""
        raise NotImplementedError


@dataclass(frozen=True)
class RouterGemmBuildSpec(BuildSpec[RouterGemmImpl]):
    """Requirements for constructing router GEMM."""

    output_dtype: torch.dtype | None = None


class RouterNoauxTCImpl(ABC):
    """Noaux tc implementation api."""

    @abstractmethod
    def forward(self, logits: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """forward."""
        raise NotImplementedError


@dataclass(frozen=True)
class RouterNoauxTCBuildSpec(BuildSpec[RouterNoauxTCImpl]):
    """Immutable requirements for constructing no-aux-loss routing."""

    scoring_func: str
    top_k: int
    n_group: int
    top_k_group: int
    n_routed_experts: int
    routed_scaling_factor: float
    renormalize: bool = True
    router_n_groups: int = -1
