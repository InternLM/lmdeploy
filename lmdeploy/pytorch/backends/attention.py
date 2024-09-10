# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch


@dataclass
class AttentionMetadata:
    """Base Attention metadata."""
    is_decoding: bool
    block_offsets: torch.Tensor
    q_start_loc: torch.Tensor = None
    q_seqlens: torch.Tensor = None
    kv_seqlens: torch.Tensor = None


T = TypeVar('T', bound=AttentionMetadata)


class AttentionImpl(ABC, Generic[T]):
    """Attention implementation."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float = None,
        num_kv_heads: int = None,
        v_head_size: int = None,
        alibi: bool = None,
        sliding_window: int = None,
        logit_softcapping: float = None,
        **kwargs,
    ) -> None:
        if scale is None:
            scale = 1.0 / (head_size**0.5)

        if num_kv_heads is None:
            num_kv_heads = num_heads

        if v_head_size is None:
            v_head_size = head_size

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.v_head_size = v_head_size
        self.alibi = alibi
        self.sliding_window = sliding_window
        self.logit_softcapping = logit_softcapping

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata: T,
    ) -> torch.Tensor:
        """forward."""
        raise NotImplementedError


class AttentionBuilder(ABC, Generic[T]):
    """Attention implementation builder."""

    @staticmethod
    @abstractmethod
    def build(
        num_heads: int,
        head_size: int,
        scale: float = None,
        num_kv_heads: int = None,
        v_head_size: int = None,
        alibi: bool = False,
        sliding_window: int = None,
        logical_softcapping: float = None,
        **kwargs,
    ) -> AttentionImpl[T]:
        """build."""
        raise NotImplementedError
