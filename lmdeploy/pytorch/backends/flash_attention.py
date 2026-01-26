# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod

from torch import Tensor


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


class FlashAttentionBuilder(ABC):
    """FlashAttention implementation builder."""

    @staticmethod
    @abstractmethod
    def build(
        num_heads: int,
        head_dim: int,
        scale: float = None,
        num_kv_heads: int = None,
        v_head_dim: int = None,
        causal: bool = True,
        sliding_window: int = None,
        logit_softcapping: float = None,
        **kwargs,
    ) -> FlashAttentionImpl:
        """build."""
        raise NotImplementedError
