# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod

import torch


class RouterGemmImpl(ABC):
    """Router GEMM implementation api."""

    def __init__(self, out_dtype: torch.dtype | None = None):
        self.out_dtype = out_dtype

    @abstractmethod
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """forward."""
        raise NotImplementedError


class RouterGemmBuilder(ABC):
    """Router GEMM implementation builder."""

    @staticmethod
    @abstractmethod
    def build(out_dtype: torch.dtype | None = None):
        """build."""
        raise NotImplementedError


class RouterNoauxTCImpl(ABC):
    """Noaux tc implementation api."""

    @abstractmethod
    def forward(self, logits: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """forward."""
        raise NotImplementedError


class RouterNoauxTCBuilder(ABC):
    """Noaux tc implementation builder."""

    @staticmethod
    @abstractmethod
    def build(
        scoring_func: str,
        top_k: int,
        n_group: int,
        topk_group: int,
        n_routed_experts: int,
        routed_scaling_factor: float,
        renormalize: bool = True,
        router_n_groups: int = -1,
    ):
        """build."""
        raise NotImplementedError
