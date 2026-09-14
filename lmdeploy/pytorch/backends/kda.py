# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from .base import BuildSpec


class KdaImpl(ABC):
    """Backend interface for Kimi Delta Attention inference."""

    @abstractmethod
    def forward(
        self,
        mixed_qkv: torch.Tensor,
        raw_gate: torch.Tensor,
        raw_beta: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor | None,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        metadata: Any,
        num_heads: int,
        head_dim: int,
        lower_bound: float,
    ) -> torch.Tensor:
        """Run short convolution followed by the KDA recurrence."""
        raise NotImplementedError


@dataclass(frozen=True)
class KdaBuildSpec(BuildSpec[KdaImpl]):
    """Request a device-specific KDA implementation."""
