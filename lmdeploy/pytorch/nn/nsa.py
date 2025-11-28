# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor, nn

from lmdeploy.pytorch.backends import OpType, get_backend
from lmdeploy.pytorch.backends.attention import AttentionMetadata
from lmdeploy.pytorch.backends.nsa import NSAIndexMeta
from lmdeploy.pytorch.model_inputs import get_step_ctx_manager


class IndexerTopKFP8(nn.Module):

    def __init__(self, topk: int, softmax_scale: float, block_size: int = 128, fill: int = -1):
        super().__init__()
        backend = get_backend()
        index_builder = backend.get_layer_impl_builder(OpType.NSAIndexFP8)
        self.index_impl = index_builder.build(topk, softmax_scale, block_size, fill)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        weights: Tensor,
        k_cache: Tensor,
        k_s_cache: Tensor,
        attn_metadata: AttentionMetadata = None,
    ):
        """forward."""
        step_ctx = get_step_ctx_manager().current_context()
        cache_config = step_ctx.cache_config
        max_tokens = cache_config.block_size * cache_config.num_gpu_blocks
        is_decoding = attn_metadata.is_decoding
        if q.size(0) == attn_metadata.kv_seqlens.size(0):
            is_decoding = True
        max_q_seqlen = 1 if is_decoding else q.size(0)
        # we need to make max_kv_seqlen=max_allocated_cache_len to enable cudagraph
        max_kv_seqlen = max_tokens if is_decoding else attn_metadata.kv_flatten_size
        meta = NSAIndexMeta(cu_seqlen_q=attn_metadata.cu_seqlens_q,
                            q_seqlens=attn_metadata.q_seqlens,
                            k_seqlens=attn_metadata.kv_seqlens,
                            block_offset=attn_metadata.block_offsets,
                            max_q_seqlen=max_q_seqlen,
                            max_kv_seqlen=max_kv_seqlen)
        ret = self.index_impl.forward(q, k, weights, k_cache, k_s_cache, meta=meta)
        return ret
