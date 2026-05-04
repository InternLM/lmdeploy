# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class V4IndexerMetadata:
    """DeepSeek V4 indexer metadata."""

    block_offsets: torch.Tensor
    start_pos: torch.Tensor
    state_ids: torch.Tensor
    compress_ratio: int
    cu_q_seqlens: torch.Tensor = None
    kv_seqlens: torch.Tensor = None


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
                meta: V4IndexerMetadata,
                block_size: int,
                layer_id: int,
                index_scratch: torch.Tensor,
                offset: int,
                is_decoding: bool) -> V4IndexerOutput:
        raise NotImplementedError


class BaseV4IndexerBuilder:

    @staticmethod
    @abstractmethod
    def build(index_topk: int, compress_ratio: int, world_size: int = 1) -> BaseV4Indexer:
        """Build layer implementation."""
        raise NotImplementedError
