# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class V4IndexerMetadata:
    """DeepSeek V4 indexer decode metadata."""

    block_offsets: torch.Tensor
    start_pos: torch.Tensor
    valid_mask: torch.Tensor
    state_ids: torch.Tensor
    compress_ratio: int


class BaseV4Indexer(ABC):

    @abstractmethod
    def forward_decode(self,
                       query: torch.Tensor,
                       weights: torch.Tensor,
                       new_kv: torch.Tensor,
                       emit_mask: torch.Tensor,
                       index_kv_cache: torch.Tensor,
                       meta: V4IndexerMetadata,
                       block_size: int,
                       layer_id: int,
                       index_scratch: torch.Tensor) -> torch.Tensor:
        """forward_decode."""
        raise NotImplementedError


class BaseV4IndexerBuilder:

    @staticmethod
    @abstractmethod
    def build(index_topk: int, compress_ratio: int, world_size: int = 1) -> BaseV4Indexer:
        """Build layer implementation."""
        raise NotImplementedError
