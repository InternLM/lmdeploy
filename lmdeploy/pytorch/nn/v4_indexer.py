# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Mapping

import torch
from torch import nn

from lmdeploy.pytorch.backends import OpType, get_backend
from lmdeploy.pytorch.backends.indexer import V4IndexerMetadata, V4IndexerOutput


class V4Indexer(nn.Module):
    """DeepSeek V4 indexer wrapper."""

    def __init__(self, index_topk: int, compress_ratio: int, num_heads: int,
                 head_dim: int):
        super().__init__()
        backend = get_backend()
        impl_builder = backend.get_layer_impl_builder(OpType.V4Indexer)
        self.impl = impl_builder.build(
            index_topk=index_topk,
            compress_ratio=compress_ratio,
            num_heads=num_heads,
            head_dim=head_dim)

    def forward(self,
                query,
                weights,
                block_caches: Mapping[str, torch.Tensor],
                meta: V4IndexerMetadata) -> V4IndexerOutput:
        return self.impl.forward(query, weights, block_caches, meta)
