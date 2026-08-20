# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from lmdeploy.pytorch.backends import get_backend
from lmdeploy.pytorch.backends.attention import V4AttentionBuildSpec, V4AttentionMetadata
from lmdeploy.pytorch.backends.indexer import V4IndexerOutput
from lmdeploy.pytorch.models.patch import get_build_model_context


class V4Attention(nn.Module):
    """DeepSeek V4 cache-aware attention wrapper."""

    def __init__(self, head_size: int, scale: float, window_size: int, compress_ratio: int, **kwargs):
        super().__init__()
        self.impl = get_backend().build_op(
            V4AttentionBuildSpec(
                head_size=head_size,
                scale=scale,
                window_size=window_size,
                compress_ratio=compress_ratio,
            ),
            enable_deterministic=get_build_model_context().enable_deterministic,
        )

    def forward(self,
                query: torch.Tensor,
                kv: torch.Tensor,
                attn_sink: torch.Tensor,
                attn_metadata: V4AttentionMetadata,
                caches: dict,
                slot: torch.Tensor,
                index_out: V4IndexerOutput | None = None):
        """Unified forward — dispatches to decoding or prefilling
        internally."""
        return self.impl.forward(query, kv, attn_sink, attn_metadata, caches, slot,
                                 index_out=index_out)
