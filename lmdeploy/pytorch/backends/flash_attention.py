# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC
from dataclasses import dataclass

from torch import Tensor

from .base import BuildSpec


class FlashAttentionImpl(ABC):
    """FlashAttention implementation."""

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                q_start_loc: Tensor,
                q_seqlens: Tensor,
                kv_start_loc: Tensor,
                kv_seqlens: Tensor,
                max_q_seqlen: int = None):
        """forward."""
        raise NotImplementedError


@dataclass(frozen=True)
class FlashAttentionBuildSpec(BuildSpec[FlashAttentionImpl]):
    """Immutable requirements for constructing non-paged attention."""

    num_heads: int
    head_dim: int
    scale: float | None
    num_kv_heads: int
    v_head_dim: int
    causal: bool
    sliding_window: int | tuple[int, int] | None
    logit_softcapping: float
