# Copyright (c) Shanghai AI Laboratory. All rights reserved.
"""Build request for the TurboMind grouped W4A16 MoE route.

The grouped route packs every expert into one TurboMind weight handle
(``LinkLinearExperts``): gate/up optionally interleaved for the fused SiLU
families, down separate. The spec type stays separate from the packed-tensor
MoE specs because the produced operator owns native packed weights and takes
only activations plus routing on ``forward``.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from .base import BuildSpec


class FusedMoETmW4A16Impl(ABC):
    """TurboMind grouped W4A16 routed-expert implementation API."""

    @abstractmethod
    def update_weights(
        self,
        gate_up_qweight: torch.Tensor,
        gate_up_scales: torch.Tensor,
        gate_up_qzeros: torch.Tensor,
        down_qweight: torch.Tensor,
        down_scales: torch.Tensor,
        down_qzeros: torch.Tensor,
    ):
        """Pack the canonical AWQ expert tensors into the TurboMind channel.

        Each argument stacks the whole rank-local expert set: ``qweight``
        ``[E, K, N * 8]`` words for gate/up and ``[E, K, N]`` words for down,
        with matching ``scales`` and ``qzeros``. The gate half of gate_up sits
        in the first ``N`` output channels, the up half in the second.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.LongTensor,
    ):
        """Run the routed experts and return the combined token outputs."""
        raise NotImplementedError


@dataclass(frozen=True)
class FusedMoETmW4A16BuildSpec(BuildSpec[FusedMoETmW4A16Impl]):
    """Immutable requirements for a grouped W4A16 MoE on the TurboMind channel.

    ``ffn_dim`` is the rank-local expert width after moe-TP slicing; the
    kernel sees one expert list whose projections are ``hidden_dim ->
    ffn_dim`` (gate/up) and ``ffn_dim -> hidden_dim`` (down). The probe
    applies the registry alignment rules to both shapes, so callers must
    keep the sliced width on the family alignment grid.
    """

    top_k: int
    num_experts: int
    hidden_dim: int
    ffn_dim: int
    renormalize: bool
    group_size: int
    output_dtype: torch.dtype
    # Module path carried for routing logs only.
    prefix: str = ''
