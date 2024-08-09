# Copyright (c) OpenMMLab. All rights reserved.
# modify from:
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/backends/abstract.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Tuple, Type, TypeVar

import torch


class AttentionBackend(ABC):

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_impl_cls() -> Type['AttentionImpl']:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_metadata_cls() -> Type['AttentionMetadata']:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_k_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_v_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        raise NotImplementedError


@dataclass
class AttentionMetadata:
    is_decoding: bool
    block_offsets: torch.Tensor
    q_start_loc: torch.Tensor = None
    q_seqlens: torch.Tensor = None
    kv_seqlens: torch.Tensor = None
    max_q_seqlen: int = 0
    max_kv_seqlen: int = 0


T = TypeVar('T', bound=AttentionMetadata)


class AttentionImpl(ABC, Generic[T]):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float = None,
        num_kv_heads: int = None,
        v_head_size: int = None,
        alibi_scale: float = None,
        sliding_window: int = None,
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
        self.alibi_scale = alibi_scale
        self.sliding_window = sliding_window

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
        raise NotImplementedError
