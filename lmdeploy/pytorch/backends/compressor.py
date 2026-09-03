# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..engine.cache_engine.schema import BlockCacheGeometry, BlockCacheRequest


@dataclass
class V4CompressorMetadata:
    """DeepSeek V4 compressor metadata."""

    cu_q_seqlens: torch.Tensor
    kv_seqlens: torch.Tensor
    block_offsets: torch.Tensor
    block_size: int
    max_q_seqlen: int


class BaseV4Compressor(ABC):

    @abstractmethod
    def get_block_cache_requests(self, geometry: 'BlockCacheGeometry') -> tuple['BlockCacheRequest', ...]:
        """Describe block caches required by this compressor implementation."""
        raise NotImplementedError

    @abstractmethod
    def score_and_fill_state(
        self,
        kv: torch.Tensor,
        score: torch.Tensor,
        ape: torch.Tensor,
        kv_state: torch.Tensor,
        score_state: torch.Tensor,
        state_ids: torch.Tensor,
        meta: V4CompressorMetadata,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def write_compressed_kv(
        self,
        compressed_kv: torch.Tensor,
        block_caches: Mapping[str, torch.Tensor],
        meta: V4CompressorMetadata,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def rotate_activation(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BaseV4CompressorBuilder:

    @staticmethod
    @abstractmethod
    def build(compress_ratio: int,
              overlap: bool,
              head_dim: int,
              is_indexer: bool = False) -> BaseV4Compressor:
        raise NotImplementedError
