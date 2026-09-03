# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from ..engine.cache_engine.schema import BlockCacheGeometry, BlockCacheRequest


@dataclass
class NSAIndexMeta:
    """Meta info of NSAIndex layer."""
    cu_seqlen_q: Tensor
    q_seqlens: Tensor
    k_seqlens: Tensor
    cu_seqlen_k: Tensor
    block_offset: Tensor
    indexer_kv_seqlens: Tensor = None
    max_q_seqlen: int = None
    max_kv_seqlen: int = None
    kv_flatten_size: int = None
    block_size: int = None
    is_decoding: bool = False
    score_meta: object = None


def _build_indexer_kv_seqlens(num_tokens: int, q_seqlens: Tensor,
                               kv_seqlens: Tensor,
                               cu_seqlens_q: Tensor) -> Tensor:
    """Build the causal KV length visible to every query row."""
    if num_tokens == kv_seqlens.size(0):
        indexer_kv_seqlens = kv_seqlens
    else:
        q_start = torch.repeat_interleave(
            cu_seqlens_q[:-1], q_seqlens, output_size=num_tokens)
        history_lengths = torch.repeat_interleave(
            kv_seqlens - q_seqlens, q_seqlens, output_size=num_tokens)
        query_offsets = torch.arange(
            num_tokens, device=q_seqlens.device, dtype=q_start.dtype) - q_start
        indexer_kv_seqlens = history_lengths + query_offsets + 1
    return indexer_kv_seqlens.to(torch.int32)


def build_nsa_index_meta(*, num_tokens: int, is_decoding: bool,
                         block_size: int, num_gpu_blocks: int,
                         sequence_metadata,
                         indexer_kv_seqlens: Tensor | None = None) -> NSAIndexMeta:
    """Build layer-invariant DSA metadata from a sequence layout.

    Derive causal KV lengths with device-agnostic Torch operations unless the caller supplies them.
    """
    q_seqlens = sequence_metadata.q_seqlens
    batch_size = q_seqlens.size(0)
    is_decoding = is_decoding or num_tokens == batch_size
    max_q_seqlen = num_tokens // batch_size if is_decoding else num_tokens
    max_kv_seqlen = (block_size * num_gpu_blocks
                     if is_decoding else sequence_metadata.max_kv_seqlen)
    kv_flatten_size = (None if is_decoding else
                       sequence_metadata.kv_flatten_size)
    if indexer_kv_seqlens is None:
        indexer_kv_seqlens = _build_indexer_kv_seqlens(
            num_tokens, q_seqlens, sequence_metadata.kv_seqlens,
            sequence_metadata.cu_seqlens_q)
    return NSAIndexMeta(
        cu_seqlen_q=sequence_metadata.cu_seqlens_q,
        q_seqlens=q_seqlens,
        k_seqlens=sequence_metadata.kv_seqlens,
        cu_seqlen_k=sequence_metadata.cu_seqlens_k,
        block_offset=sequence_metadata.block_offsets,
        indexer_kv_seqlens=indexer_kv_seqlens,
        max_q_seqlen=max_q_seqlen,
        max_kv_seqlen=max_kv_seqlen,
        kv_flatten_size=kv_flatten_size,
        block_size=block_size,
        is_decoding=is_decoding,
    )


def should_skip_nsa_indexer(model_metas) -> bool:
    """Whether an MTP step reuses previously computed top-k indices."""
    return bool(model_metas) and all(
        meta is not None and meta.get('skip_topk', False)
        for meta in model_metas)


class BaseNSAIndexFP8(ABC):

    @abstractmethod
    def get_block_cache_requests(self, geometry: BlockCacheGeometry,
                                 head_dim: int) -> tuple[BlockCacheRequest, ...]:
        """Describe the selected implementation's indexer-K caches."""
        raise NotImplementedError('Not implemented.')

    @abstractmethod
    def get_step_metadata(self, attn_metadata) -> NSAIndexMeta:
        """Return metadata prepared for the current inference step."""
        raise NotImplementedError('Not implemented.')

    @abstractmethod
    def forward(self, q: Tensor, k: Tensor, weights: Tensor,
                indexer_k_cache: Tensor, meta: NSAIndexMeta) -> Tensor | None:
        """forward."""
        raise NotImplementedError('Not implemented.')

    @abstractmethod
    def forward_fused(self, q: Tensor, k: Tensor, weights: Tensor, norm_weight: Tensor, norm_bias: Tensor, cos: Tensor,
                      sin: Tensor, indexer_k_cache: Tensor, norm_eps: float, head_gate_scale: float,
                      rope_interleaved: bool, meta: NSAIndexMeta) -> Tensor | None:
        """Forward with fused DSA indexer preparation."""
        raise NotImplementedError('Not implemented.')

class BaseNSAIndexFP8Builder:

    @staticmethod
    @abstractmethod
    def build(topk: int, softmax_scale: float, block_size: int = 128, fill: int = -1,
              allow_short_prefill_scoring_skip: bool = False) -> BaseNSAIndexFP8:
        """Build layer implementation."""
        raise NotImplementedError('Not implemented.')
