# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import Tuple

import torch


class RouterNoauxTCImpl(ABC):
    """Noaux tc implementation api."""

    @abstractmethod
    def forward(self, logits: torch.Tensor, bias: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
