# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn

from lmdeploy.pytorch.backends import OpType, get_backend
from lmdeploy.pytorch.backends.indexer import V4IndexerMetadata, V4IndexerOutput


class V4Indexer(nn.Module):
    """DeepSeek V4 indexer wrapper."""

    def __init__(self, index_topk: int, compress_ratio: int):
        super().__init__()
        backend = get_backend()
        impl_builder = backend.get_layer_impl_builder(OpType.V4Indexer)
        self.impl = impl_builder.build(index_topk=index_topk, compress_ratio=compress_ratio)

    def forward(self,
                query,
                weights,
                index_kv_cache,
                index_kv_scale_cache,
                meta: V4IndexerMetadata,
                block_size: int,
                offset: int) -> V4IndexerOutput:
        return self.impl.forward(query, weights, index_kv_cache, index_kv_scale_cache, meta, block_size,
                                 offset)
