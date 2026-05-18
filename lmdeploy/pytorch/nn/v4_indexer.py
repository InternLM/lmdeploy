# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn

from lmdeploy.pytorch.backends import OpType, get_backend
from lmdeploy.pytorch.backends.indexer import V4IndexerMetadata, V4IndexerOutput
from lmdeploy.pytorch.distributed import get_dist_manager


class V4Indexer(nn.Module):
    """DeepSeek V4 indexer wrapper."""

    def __init__(self, index_topk: int, compress_ratio: int):
        super().__init__()
        backend = get_backend()
        impl_builder = backend.get_layer_impl_builder(OpType.V4Indexer)
        self.impl = impl_builder.build(index_topk=index_topk, compress_ratio=compress_ratio)
        dist_ctx = get_dist_manager().current_context()
        attn_tp = dist_ctx.dist_config.attn_tp
        self.tp_group = dist_ctx.attn_tp_group.gpu_group if attn_tp > 1 else None

    def forward(self,
                query,
                weights,
                index_kv_cache,
                index_kv_scale_cache,
                meta: V4IndexerMetadata) -> V4IndexerOutput:
        # Lazy-init block_size on impl from metadata (global constant, same every step)
        if self.impl._block_size is None:
            self.impl.block_size = meta.block_size
        return self.impl.forward(query, weights, index_kv_cache, index_kv_scale_cache, meta,
                                 tp_group=self.tp_group)
