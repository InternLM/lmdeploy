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
        *,
        checkpoint_state: torch.Tensor | None = None,
        checkpoint_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run convolution and recurrence, optionally exporting a prefill
        recurrent state.

        Checkpoint arguments must be provided together. The caller supplies a contiguous [N, H, V, K] FP32 output buffer
        and [N] sequence-relative token counts aligned to 16 (matching cu_seqlens dtype). The caller owns convolution
        snapshots, checkpoint publication and any MTP ring-slot selection. Decode does not export here.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class KdaBuildSpec(BuildSpec[KdaImpl]):
    """Request a device-specific KDA implementation."""

    model_type: str | None = None
