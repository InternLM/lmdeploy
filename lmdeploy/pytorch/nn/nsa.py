# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor, nn

from lmdeploy.pytorch.backends import OpType, get_backend
from lmdeploy.pytorch.backends.attention import AttentionMetadata
from lmdeploy.pytorch.backends.nsa import NSAIndexMeta
from lmdeploy.pytorch.consts import DSA_INDEXER_K_CACHE_NAME
from lmdeploy.pytorch.engine.cache_engine.schema import BlockCacheBinding, BlockCacheRequestContext
from lmdeploy.pytorch.model_inputs import get_step_ctx_manager


def update_nsa_indexer_kv_seqlens(num_tokens: int, attn_metadata: AttentionMetadata) -> None:
    """Prepare per-query causal KV lengths once for all indexer layers."""
    q_seqlens = attn_metadata.q_seqlens
    kv_seqlens = attn_metadata.kv_seqlens
    if num_tokens == kv_seqlens.size(0):
        indexer_kv_seqlens = kv_seqlens
    else:
        cu_seqlens_q = attn_metadata.cu_seqlens_q
        q_start = torch.repeat_interleave(cu_seqlens_q[:-1], q_seqlens, output_size=num_tokens)
        history_lengths = torch.repeat_interleave(kv_seqlens - q_seqlens, q_seqlens, output_size=num_tokens)
        query_offsets = torch.arange(num_tokens, device=q_seqlens.device, dtype=q_start.dtype) - q_start
        indexer_kv_seqlens = history_lengths + query_offsets + 1
    if indexer_kv_seqlens.dtype != torch.int32:
        indexer_kv_seqlens = indexer_kv_seqlens.to(torch.int32)
    attn_metadata.indexer_kv_seqlens = indexer_kv_seqlens


class IndexerTopKFP8(nn.Module):

    def __init__(self, topk: int, softmax_scale: float, head_dim: int, block_size: int = 128, fill: int = -1):
        super().__init__()
        backend = get_backend()
        index_builder = backend.get_layer_impl_builder(OpType.NSAIndexFP8)
        self.index_impl = index_builder.build(topk, softmax_scale, block_size, fill)
        self.head_dim = head_dim
        self._block_cache_binding: BlockCacheBinding | None = None

    def get_block_cache_requests(self, context: BlockCacheRequestContext):
        """Return the selected implementation's cache requirements."""
        request = self.index_impl.get_block_cache_request(context.geometry, self.head_dim)
        return (request, )

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

    @staticmethod
    def _get_max_q_seqlen(q: Tensor,
                          attn_metadata: AttentionMetadata) -> int:
        """Get the query width used by the index and cache-fill kernels.

        Speculative target verification remains a decoding step, but its
        flattened Q contains ``num_spec_tokens + 1`` rows per request. The
        kernels need that real width to process every verification row.
        """
        batch_size = attn_metadata.kv_seqlens.size(0)
        # fp8_index also identifies one row per request as decode layout, so
        # keep its metadata consistent when the phase flag has not changed yet.
        is_decoding = attn_metadata.is_decoding or q.size(0) == batch_size
        # Prefer a width prepared by the attention backend; otherwise derive it
        # from the flattened query rows.
        max_q_seqlen = attn_metadata.max_q_seqlen
        if max_q_seqlen is None:
            max_q_seqlen = q.size(0)
            if is_decoding:
                max_q_seqlen //= batch_size
        return max_q_seqlen

    @staticmethod
    def _build_meta(q: Tensor, attn_metadata: AttentionMetadata) -> NSAIndexMeta:
        step_ctx = get_step_ctx_manager().current_context()
        cache_config = step_ctx.cache_config
        max_tokens = cache_config.block_size * cache_config.num_gpu_blocks
        is_decoding = attn_metadata.is_decoding
        if q.size(0) == attn_metadata.kv_seqlens.size(0):
            is_decoding = True
        max_q_seqlen = IndexerTopKFP8._get_max_q_seqlen(q, attn_metadata)
        # Decode uses the full cache capacity to keep CUDA graph shapes stable.
        max_kv_seqlen = max_tokens if is_decoding else attn_metadata.kv_flatten_size
        return NSAIndexMeta(cu_seqlen_q=attn_metadata.cu_seqlens_q,
                            q_seqlens=attn_metadata.q_seqlens,
                            k_seqlens=attn_metadata.kv_seqlens,
                            block_offset=attn_metadata.block_offsets,
                            indexer_kv_seqlens=attn_metadata.indexer_kv_seqlens,
                            max_q_seqlen=max_q_seqlen,
                            max_kv_seqlen=max_kv_seqlen)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        weights: Tensor,
        attn_metadata: AttentionMetadata = None,
    ):
        """forward."""
        indexer_k_cache = self._get_block_cache()
        meta = self._build_meta(q, attn_metadata)
        ret = self.index_impl.forward(q, k, weights, indexer_k_cache, meta=meta)
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
        meta = self._build_meta(q, attn_metadata)
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
