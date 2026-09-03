# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor, nn

from lmdeploy.pytorch.backends import OpType, get_backend
from lmdeploy.pytorch.backends.attention import AttentionMetadata
from lmdeploy.pytorch.consts import DSA_INDEXER_K_CACHE_NAME
from lmdeploy.pytorch.engine.cache_engine.schema import BlockCacheBinding, BlockCacheRequestContext
from lmdeploy.pytorch.model_inputs import get_step_ctx_manager


class IndexerTopKFP8(nn.Module):

    def __init__(self, topk: int, softmax_scale: float, head_dim: int, block_size: int = 128,
                 fill: int = -1,
                 allow_short_prefill_scoring_skip: bool = False):
        super().__init__()
        backend = get_backend()
        index_builder = backend.get_layer_impl_builder(OpType.NSAIndexFP8)
        self.index_impl = index_builder.build(
            topk,
            softmax_scale,
            block_size,
            fill,
            allow_short_prefill_scoring_skip=allow_short_prefill_scoring_skip,
        )
        self.head_dim = head_dim
        self._block_cache_binding: BlockCacheBinding | None = None

    def get_block_cache_requests(self, context: BlockCacheRequestContext):
        """Return the selected implementation's cache requirements."""
        return self.index_impl.get_block_cache_requests(context.geometry,
                                                        self.head_dim)

    def bind_block_cache(self, binding: BlockCacheBinding):
        """Retain the logical cache binding assigned to this indexer."""
        if binding.cache_name != DSA_INDEXER_K_CACHE_NAME:
            raise ValueError(f'Unexpected DSA indexer cache name: {binding.cache_name}.')
        self._block_cache_binding = binding

    def _get_block_cache(self) -> Tensor:
        binding = self._block_cache_binding
        if binding is None:
            raise RuntimeError('The DSA indexer block cache has not been bound.')
        context = get_step_ctx_manager().current_context()
        block_caches = context.block_caches
        if hasattr(block_caches, 'row'):
            return block_caches.row(binding.cache_name, binding.consumer_row)
        return block_caches[binding.cache_name][binding.consumer_row]

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        weights: Tensor,
        attn_metadata: AttentionMetadata = None,
    ):
        """forward."""
        indexer_k_cache = self._get_block_cache()
        meta = self.index_impl.get_step_metadata(attn_metadata)
        ret = self.index_impl.forward(q,
                                      k,
                                      weights,
                                      indexer_k_cache,
                                      meta=meta)
        return ret

    def forward_fused(self,
                      q: Tensor,
                      k: Tensor,
                      weights: Tensor,
                      norm_weight: Tensor,
                      norm_bias: Tensor,
                      cos: Tensor,
                      sin: Tensor,
                      norm_eps: float,
                      head_gate_scale: float,
                      rope_interleaved: bool,
                      attn_metadata: AttentionMetadata = None):
        """Forward with fused DSA indexer preparation."""
        indexer_k_cache = self._get_block_cache()
        meta = self.index_impl.get_step_metadata(attn_metadata)
        return self.index_impl.forward_fused(q,
                                             k,
                                             weights,
                                             norm_weight,
                                             norm_bias,
                                             cos,
                                             sin,
                                             indexer_k_cache,
                                             norm_eps=norm_eps,
                                             head_gate_scale=head_gate_scale,
                                             rope_interleaved=rope_interleaved,
                                             meta=meta)
