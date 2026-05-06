# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class V4IndexerMetadata:
    """DeepSeek V4 indexer metadata."""

    block_offsets: torch.Tensor
    compress_ratio: int
    is_decoding: bool
    cu_q_seqlens: torch.Tensor = None
    kv_seqlens: torch.Tensor = None
    max_kv_seqlen: int = None
    max_q_seqlen: int = None


@dataclass
class V4IndexerOutput:
    """DeepSeek V4 indexer output."""

    indices_in_kvcache: torch.Tensor
    topk_length: torch.Tensor


class BaseV4Indexer(ABC):

    @abstractmethod
    def forward(self,
                query: torch.Tensor,
                weights: torch.Tensor,
                index_kv_cache: torch.Tensor,
                index_kv_scale_cache: torch.Tensor,
                meta: V4IndexerMetadata,
                block_size: int,
                offset: int) -> V4IndexerOutput:
        raise NotImplementedError


class BaseV4IndexerBuilder:

    @staticmethod
    @abstractmethod
    def build(index_topk: int, compress_ratio: int) -> BaseV4Indexer:
        """Build layer implementation."""
        raise NotImplementedError
