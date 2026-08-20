# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn

from lmdeploy.pytorch.backends import get_backend
from lmdeploy.pytorch.backends.indexer import V4IndexerBuildSpec, V4IndexerMetadata, V4IndexerOutput
from lmdeploy.pytorch.models.patch import get_build_model_context


class V4Indexer(nn.Module):
    """DeepSeek V4 indexer wrapper."""

    def __init__(self, index_topk: int, compress_ratio: int):
        super().__init__()
        self.impl = get_backend().build_op(
            V4IndexerBuildSpec(index_topk=index_topk, compress_ratio=compress_ratio),
            enable_deterministic=get_build_model_context().enable_deterministic,
        )

    def forward(self,
                query,
                weights,
                index_kv_cache,
                index_kv_scale_cache,
                meta: V4IndexerMetadata) -> V4IndexerOutput:
        return self.impl.forward(query, weights, index_kv_cache, index_kv_scale_cache, meta)
