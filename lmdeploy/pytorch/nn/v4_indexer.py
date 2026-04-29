# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn

from lmdeploy.pytorch.backends import OpType, get_backend
from lmdeploy.pytorch.backends.indexer import V4IndexerMetadata, V4IndexerOutput


class V4Indexer(nn.Module):
    """DeepSeek V4 decode indexer wrapper."""

    def __init__(self, index_topk: int, compress_ratio: int, world_size: int = 1):
        super().__init__()
        backend = get_backend()
        impl_builder = backend.get_layer_impl_builder(OpType.V4Indexer)
        self.impl = impl_builder.build(index_topk=index_topk, compress_ratio=compress_ratio, world_size=world_size)

    def forward_decode(self,
                       query,
                       weights,
                       index_kv_cache,
                       meta: V4IndexerMetadata,
                       block_size: int,
                       layer_id: int,
                       index_scratch) -> V4IndexerOutput:
        return self.impl.forward_decode(query, weights, index_kv_cache, meta, block_size, layer_id, index_scratch)
