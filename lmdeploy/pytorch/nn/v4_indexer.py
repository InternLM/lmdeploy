# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Mapping

import torch
from torch import nn

from lmdeploy.pytorch.backends import get_backend
from lmdeploy.pytorch.backends.indexer import V4IndexerBuildSpec, V4IndexerMetadata, V4IndexerOutput
from lmdeploy.pytorch.models.patch import get_build_model_context


class V4Indexer(nn.Module):
    """DeepSeek V4 indexer wrapper."""

    def __init__(self, index_topk: int, compress_ratio: int, num_heads: int,
                 head_dim: int):
        super().__init__()
        self.impl = get_backend().build_op(
            V4IndexerBuildSpec(
                index_top_k=index_topk,
                compress_ratio=compress_ratio,
                num_heads=num_heads,
                head_dim=head_dim,
            ),
            enable_deterministic=get_build_model_context().enable_deterministic,
        )

    def forward(self,
                query,
                weights,
                block_caches: Mapping[str, torch.Tensor],
                meta: V4IndexerMetadata) -> V4IndexerOutput:
        return self.impl.forward(query, weights, block_caches, meta)
