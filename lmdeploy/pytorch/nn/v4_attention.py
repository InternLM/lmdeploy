# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Mapping

import torch
from torch import nn

from lmdeploy.pytorch.backends import get_backend
from lmdeploy.pytorch.backends.attention import V4AttentionBuildSpec, V4AttentionMetadata
from lmdeploy.pytorch.backends.indexer import V4IndexerOutput
from lmdeploy.pytorch.models.patch import get_build_model_context


class V4Attention(nn.Module):
    """DeepSeek V4 cache-aware attention wrapper."""

    def __init__(self, head_size: int, scale: float, window_size: int,
                 compress_ratio: int,
                 ring_storage_capacity: int | None = None, **kwargs):
        super().__init__()
        if ring_storage_capacity is None:
            ring_storage_capacity = window_size
        self.impl = get_backend().build_op(
            V4AttentionBuildSpec(
                head_dim=head_size,
                scale=scale,
                window_size=window_size,
                ring_storage_capacity=ring_storage_capacity,
                compress_ratio=compress_ratio,
            ),
            enable_deterministic=get_build_model_context().enable_deterministic,
        )

    def build_cache_write_metadata(self, attn_metadata, position_ids: torch.Tensor,
                                   state_ids: torch.Tensor, num_tokens: int):
        return self.impl.build_cache_write_metadata(
            attn_metadata, position_ids, state_ids, num_tokens)

    def write_cache(self, kv: torch.Tensor, window_state: torch.Tensor, metadata) -> None:
        self.impl.write_cache(kv, window_state, metadata)

    def forward(self,
                query: torch.Tensor,
                kv: torch.Tensor,
                attn_sink: torch.Tensor,
                attn_metadata: V4AttentionMetadata,
                window_state_fp8: torch.Tensor,
                block_caches: Mapping[str, torch.Tensor],
                slot: torch.Tensor,
                index_out: V4IndexerOutput | None = None):
        """Unified forward — dispatches to decoding or prefilling
        internally."""
        return self.impl.forward(query, kv, attn_sink, attn_metadata,
                                 window_state_fp8, block_caches, slot,
                                 index_out=index_out)
