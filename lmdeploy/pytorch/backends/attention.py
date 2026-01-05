# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Generic, Literal, TypeVar

import torch


@dataclass
class AttentionMetadata:
    """Base Attention metadata."""
    is_decoding: bool
    block_offsets: torch.Tensor
    q_start_loc: torch.Tensor = None
    q_seqlens: torch.Tensor = None
    kv_seqlens: torch.Tensor = None
    fill_seqlens: torch.Tensor = None
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None
    quant_policy: Literal[0, 4, 8] = 0


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
        logit_softcapping: float = 0.0,
        causal: bool = True,
        use_flash_mla: bool = False,
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
        self.causal = causal
        self.use_flash_mla = use_flash_mla
        self.alibi_slopes = None

    @staticmethod
    @lru_cache(maxsize=4)
    def make_alibi_slopes(head_start: int, head_end: int, num_heads: int, alibi_scale: float, dtype: torch.dtype,
                          device: torch.device):
        """Make alibi slopes."""
        head_ids = torch.arange(head_start, head_end, dtype=dtype, device=device)
        num_heads_tensor = head_ids.new_full([1], num_heads)
        num_heads_p2 = num_heads_tensor.log2().to(torch.int64).exp2()

        # update head_ids and closest_power_of_2
        mask = head_ids < num_heads_p2
        head_ids = torch.where(mask, head_ids, (head_ids - num_heads_p2) * 2)
        closest_power_of_2 = torch.where(mask, num_heads_p2, num_heads_p2 * 2)

        # get slope
        start = torch.sub(3, closest_power_of_2.log2()).exp2().neg()
        start = start.exp2()
        ratio = start
        return start * torch.pow(ratio, head_ids) * alibi_scale

    def set_alibi_slopes(self, slopes: torch.Tensor):
        self.alibi_slopes = slopes

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata: T,
        k_scales_zeros: torch.Tensor = None,
        v_scales_zeros: torch.Tensor = None,
        learnable_sink: torch.Tensor = None,
        nsa_indices: torch.Tensor = None,
        inplace: bool = False,
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
        logit_softcapping: float = 0.0,
        causal: bool = True,
        use_flash_mla: bool = False,
        learnable_sink: bool = False,
        block_sparse_size: int = 1,
        **kwargs,
    ) -> AttentionImpl[T]:
        """build."""
        raise NotImplementedError
