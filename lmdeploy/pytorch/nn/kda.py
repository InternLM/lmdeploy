# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any

import torch
from torch import nn

from lmdeploy.pytorch.backends import get_backend
from lmdeploy.pytorch.backends.kda import KdaBuildSpec
from lmdeploy.pytorch.models.patch import get_build_model_context


class Kda(nn.Module):
    """Backend-dispatched Kimi Delta Attention recurrence."""

    def __init__(self, model_type: str | None = None):
        super().__init__()
        self.impl = get_backend().build_op(
            KdaBuildSpec(model_type=model_type),
            enable_deterministic=get_build_model_context().enable_deterministic,
        )

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
        return self.impl.forward(
            mixed_qkv=mixed_qkv,
            raw_gate=raw_gate,
            raw_beta=raw_beta,
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            a_log=a_log,
            dt_bias=dt_bias,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            metadata=metadata,
            num_heads=num_heads,
            head_dim=head_dim,
            lower_bound=lower_bound,
            checkpoint_state=checkpoint_state,
            checkpoint_offsets=checkpoint_offsets,
        )
