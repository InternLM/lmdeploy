# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn

from lmdeploy.pytorch.backends import OpType, get_backend
from lmdeploy.pytorch.backends.attention import AttentionMetadata
from lmdeploy.pytorch.backends.indexer import V4IndexerMetadata, V4IndexerOutput
from lmdeploy.pytorch.model_inputs import get_step_ctx_manager


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
                attn_metadata: AttentionMetadata) -> V4IndexerOutput:
        step_ctx = get_step_ctx_manager().current_context()
        cache_config = step_ctx.cache_config
        is_decoding = attn_metadata.is_decoding
        cu_q_seqlens = attn_metadata.cu_seqlens_q
        kv_seqlens = attn_metadata.kv_seqlens
        # CUDAGraph compat: fixed upper bound for decode max_kv_seqlen
        max_kv_seqlen = cache_config.block_size * cache_config.num_gpu_blocks if is_decoding else step_ctx.max_kv_seqlen
        max_q_seqlen = step_ctx.max_q_seqlen
        block_size = cache_config.block_size
        meta = V4IndexerMetadata(
            block_offsets=attn_metadata.block_offsets,
            compress_ratio=self.impl.compress_ratio,
            is_decoding=is_decoding,
            cu_q_seqlens=cu_q_seqlens,
            kv_seqlens=kv_seqlens,
            q_seqlens=attn_metadata.q_seqlens,
            max_kv_seqlen=max_kv_seqlen,
            max_q_seqlen=max_q_seqlen,
        )
        return self.impl.forward(query, weights, index_kv_cache, index_kv_scale_cache, meta, block_size)
