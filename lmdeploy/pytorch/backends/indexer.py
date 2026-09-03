# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from .base import BuildSpec


@dataclass
class V4IndexerMetadata:
    """DeepSeek V4 indexer metadata."""

    block_offsets: torch.Tensor
    is_decoding: bool
    cu_q_seqlens: torch.Tensor = None
    kv_seqlens: torch.Tensor = None
    q_seqlens: torch.Tensor = None
    max_kv_seqlen: int = None
    max_q_seqlen: int = None
    block_size: int = None
    num_index: torch.Tensor = None
    num_index_r4: torch.Tensor = None
    num_index_r128: torch.Tensor = None


@dataclass
class V4IndexerOutput:
    """DeepSeek V4 indexer output."""

    indices_in_kvcache: torch.Tensor
    topk_length: torch.Tensor


class V4IndexerImpl(ABC):

    @abstractmethod
    def forward(self,
                query: torch.Tensor,
                weights: torch.Tensor,
                block_caches: Mapping[str, torch.Tensor],
                meta: V4IndexerMetadata) -> V4IndexerOutput:
        raise NotImplementedError


@dataclass(frozen=True)
class V4IndexerBuildSpec(BuildSpec[V4IndexerImpl]):
    """Immutable requirements for constructing a DeepSeek-V4 indexer."""

    index_top_k: int
    compress_ratio: int
    num_heads: int
    head_dim: int
