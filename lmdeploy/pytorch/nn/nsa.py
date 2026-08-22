# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor, nn

from lmdeploy.pytorch.backends import get_backend
from lmdeploy.pytorch.backends.attention import AttentionMetadata
from lmdeploy.pytorch.backends.nsa import NSAIndexFP8BuildSpec
from lmdeploy.pytorch.consts import DSA_INDEXER_K_CACHE_NAME
from lmdeploy.pytorch.model_inputs import get_step_ctx_manager
from lmdeploy.pytorch.models.patch import get_build_model_context


def get_dsa_indexer_k_cache(layer_idx: int) -> Tensor:
    """Return the packed indexer-K cache owned by one DSA layer."""
    context = get_step_ctx_manager().current_context()
    return context.block_caches.layer(DSA_INDEXER_K_CACHE_NAME, layer_idx)


class IndexerTopKFP8(nn.Module):

    def __init__(self, topk: int, softmax_scale: float, block_size: int = 128, fill: int = -1):
        super().__init__()
        self.index_impl = get_backend().build_op(
            NSAIndexFP8BuildSpec(top_k=topk, softmax_scale=softmax_scale, block_size=block_size, fill=fill),
            enable_deterministic=get_build_model_context().enable_deterministic,
        )

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        weights: Tensor,
        indexer_k_cache: Tensor,
        attn_metadata: AttentionMetadata = None,
    ):
        """forward."""
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
                      indexer_k_cache: Tensor,
                      norm_eps: float,
                      head_gate_scale: float,
                      rope_interleaved: bool,
                      attn_metadata: AttentionMetadata = None):
        """Forward with fused DSA indexer preparation."""
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
