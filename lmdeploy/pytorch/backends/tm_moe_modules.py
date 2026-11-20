# Copyright (c) Shanghai AI Laboratory. All rights reserved.
"""Build request for the TurboMind grouped W4A16 MoE route.

The grouped route packs every expert into one TurboMind weight handle
(``LinkLinearExperts``): gate/up optionally interleaved for the fused SiLU
families, down separate. The spec type stays separate from the packed-tensor
MoE specs because the produced operator owns native packed weights and takes
only activations plus routing on ``forward``.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
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

    def sleep_release(self):
        """Free packed native handles before ``ModelAgent.sleep`` moves the
        model.

        Backends whose packed weights are torch tensors need no work — the
        registered parameters already travel with ``Module.to``. Native
        packed-weight backends override this to close their handles.
        """


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


class FusedMoETmMXFP4Impl(ABC):
    """TurboMind grouped MXFP4 routed-expert implementation API.

    Same native-handle contract as the W4A16 route, over the MXFP4 weight
    format (e2m1 blocks + e8m0 scales, group 32). Experts may carry bias,
    which has no kernel epilogue on this route: ``update_weights`` takes the
    per-projection bias tensors and ``forward`` applies them python-side at
    the blocked-slot granularity.
    """

    @abstractmethod
    def update_weights(
        self,
        gate_up_blocks: torch.Tensor,
        gate_up_scales: torch.Tensor,
        gate_up_bias: torch.Tensor | None,
        down_blocks: torch.Tensor,
        down_scales: torch.Tensor,
        down_bias: torch.Tensor | None,
    ):
        """Pack the checkpoint-native MXFP4 expert tensors into the channel.

        Each argument stacks the whole rank-local expert set in checkpoint
        orientation: ``blocks`` ``[E, N, K // 2]`` uint8 (two e2m1 codes per
        byte along K), ``scales`` ``[E, N, K // 32]`` e8m0 uint8, ``bias``
        ``[E, N]`` float. The gate half of gate_up sits in the first ``N``
        output channels, the up half in the second (already deinterleaved
        at load time).
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

    def sleep_release(self):
        """Free packed native handles and device bias copies before
        ``ModelAgent.sleep`` moves the model."""


@dataclass(frozen=True)
class FusedMoETmMXFP4BuildSpec(BuildSpec[FusedMoETmMXFP4Impl]):
    """Immutable requirements for a grouped MXFP4 MoE on the TurboMind channel.

    Shape semantics match ``FusedMoETmW4A16BuildSpec``. ``act_func`` is the
    checkpoint's gated activation over the concatenated ``(gate, up)``
    intermediate; ``None`` means the built-in SiLU-mul. A custom activation
    rules out the fused-SiLU kernel lane, whose epilogue bakes plain SiLU
    in.
    """

    top_k: int
    num_experts: int
    hidden_dim: int
    ffn_dim: int
    renormalize: bool
    group_size: int
    output_dtype: torch.dtype
    act_func: Callable | None
    # Module path carried for routing logs only.
    prefix: str = ''
