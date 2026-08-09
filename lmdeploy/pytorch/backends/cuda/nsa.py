# Copyright (c) OpenMMLab. All rights reserved.
import functools

import torch
from torch import Tensor

from lmdeploy.pytorch.consts import (
    DSA_INDEX_SCALE_BYTES,
    DSA_INDEXER_K_CACHE_NAME,
    dsa_packed_indexer_k_cache_shape,
)
from lmdeploy.pytorch.engine.cache_engine.schema import BlockCacheGeometry, BlockCacheRequest
from lmdeploy.pytorch.kernels.cuda.bitonic_topk import bitonic_topk
from lmdeploy.pytorch.kernels.cuda.blocked_gemm_fp8 import quant_fp8
from lmdeploy.pytorch.kernels.cuda.ds_index import fp8_index
from lmdeploy.pytorch.kernels.cuda.dsa_indexer_preprocess import prepare_dsa_indexer_k_cache, prepare_dsa_indexer_q
from lmdeploy.pytorch.kernels.cuda.fill_kv_cache import fill_kv_cache_blocked_fp8

from ..nsa import BaseNSAIndexFP8, BaseNSAIndexFP8Builder, NSAIndexMeta


def _get_dsa_indexer_k_cache_views(indexer_k_cache: Tensor, head_dim: int) -> tuple[Tensor, Tensor]:
    """Return FP8 K and FP32 scale views of a packed DSA indexer-K cache."""
    if indexer_k_cache.dtype != torch.uint8:
        raise TypeError(f'Packed DSA indexer-K cache must be uint8, got {indexer_k_cache.dtype}.')
    if indexer_k_cache.dim() != 4 or indexer_k_cache.size(2) != 1:
        raise ValueError('Packed DSA indexer-K cache must have shape [num_blocks, entries, 1, head_dim + 4].')
    if indexer_k_cache.size(-1) != head_dim + DSA_INDEX_SCALE_BYTES:
        raise ValueError(f'Packed DSA indexer-K cache last dim must be {head_dim + DSA_INDEX_SCALE_BYTES}, '
                         f'got {indexer_k_cache.size(-1)}.')

    num_blocks, entries_per_block = indexer_k_cache.shape[:2]
    flat = indexer_k_cache.view(num_blocks, -1)
    value_bytes = entries_per_block * head_dim
    scale_bytes = entries_per_block * DSA_INDEX_SCALE_BYTES
    values = flat[:, :value_bytes].view(torch.float8_e4m3fn).view(num_blocks, entries_per_block, head_dim)
    scales = flat[:, value_bytes:value_bytes + scale_bytes].view(torch.float32).view(num_blocks, entries_per_block, 1)
    return values, scales


@functools.lru_cache
def _get_sparse_index_topk(topk: int):
    try:
        from lmdeploy.pytorch.kernels.cuda.sparse_index_topk import (
            is_sparse_index_topk_supported,
            sparse_index_topk,
        )
    except ImportError:
        return None
    if is_sparse_index_topk_supported(topk):
        return sparse_index_topk
    return None


class TritonNSAIndexFP8(BaseNSAIndexFP8):

    def __init__(self, topk: int, softmax_scale: float, block_size: int, fill: int) -> None:
        super().__init__()
        self.topk = topk
        self.softmax_scale = softmax_scale
        self.block_size = block_size
        self.fill = fill
        # TODO: configable scale fmt
        self.scale_fmt = 'ue8m0'
        self._sparse_index_topk = _get_sparse_index_topk(topk)

    def get_block_cache_request(self, geometry: BlockCacheGeometry, head_dim: int) -> BlockCacheRequest:
        """Request one DeepGEMM-compatible packed cache row per indexer."""
        return BlockCacheRequest(
            name=DSA_INDEXER_K_CACHE_NAME,
            shape=dsa_packed_indexer_k_cache_shape(geometry.kernel_block_size, head_dim),
            dtype=torch.uint8,
            per_row_contiguous=True,
        )

    def _forward_index(self, q: Tensor, q_s: Tensor, k_cache: Tensor, k_s_cache: Tensor, meta: NSAIndexMeta) -> Tensor:
        cu_seqlen_q = meta.cu_seqlen_q
        q_seqlens = meta.q_seqlens
        k_seqlens = meta.k_seqlens
        block_offset = meta.block_offset
        max_q_seqlen = meta.max_q_seqlen
        max_kv_seqlen = meta.max_kv_seqlen

        scores = fp8_index(q,
                           q_s,
                           k_cache,
                           k_s_cache[..., 0],
                           cu_seqlen_q,
                           k_seqlens,
                           block_offset,
                           max_q_seqlen=max_q_seqlen,
                           max_k_seqlen=max_kv_seqlen,
                           causal=True)
        indexer_kv_seqlens = meta.indexer_kv_seqlens
        if self._sparse_index_topk is not None:
            return self._sparse_index_topk(scores,
                                           q_seqlens,
                                           indexer_kv_seqlens,
                                           self.topk,
                                           fill=self.fill,
                                           descending=True,
                                           sorted=False)
        return bitonic_topk(scores, q_seqlens, indexer_kv_seqlens, self.topk, fill=self.fill, descending=True)

    def forward(self, q: Tensor, k: Tensor, weights: Tensor, indexer_k_cache: Tensor,
                meta: NSAIndexMeta) -> Tensor:
        assert q.dim() == 3
        assert k.dim() == 2
        k_cache, k_s_cache = _get_dsa_indexer_k_cache_views(indexer_k_cache, k.size(-1))
        q_shape = q.shape
        q = q.reshape(-1, q_shape[-1])
        q, q_s = quant_fp8(q, self.block_size, dtype=k_cache.dtype, trans_scale=True, scale_fmt=self.scale_fmt)
        q = q.reshape(*q_shape)
        q_s = q_s.reshape(weights.shape)
        q_s = q_s * self.softmax_scale * weights

        fill_kv_cache_blocked_fp8(k[:, None],
                                  None,
                                  k_cache[..., None, :],
                                  None,
                                  k_s_cache[..., None, :],
                                  None,
                                  cu_seqlen_q=meta.cu_seqlen_q,
                                  kv_seqlens=meta.k_seqlens,
                                  max_q_seqlen=meta.max_q_seqlen,
                                  block_offsets=meta.block_offset,
                                  group_size=self.block_size,
                                  scale_fmt=self.scale_fmt)
        return self._forward_index(q, q_s, k_cache, k_s_cache, meta)

    def forward_fused(self, q: Tensor, k: Tensor, weights: Tensor, norm_weight: Tensor, norm_bias: Tensor, cos: Tensor,
                      sin: Tensor, indexer_k_cache: Tensor, norm_eps: float, head_gate_scale: float,
                      rope_interleaved: bool, meta: NSAIndexMeta) -> Tensor:
        """Prepare FP8 Q and write K cache without allocating rotated BF16
        Q/K."""
        k_cache, k_s_cache = _get_dsa_indexer_k_cache_views(indexer_k_cache, k.size(-1))
        q, q_s = prepare_dsa_indexer_q(q,
                                       weights,
                                       cos,
                                       sin,
                                       score_scale=self.softmax_scale * head_gate_scale,
                                       out_dtype=k_cache.dtype,
                                       rope_interleaved=rope_interleaved)
        prepare_dsa_indexer_k_cache(k,
                                    norm_weight,
                                    norm_bias,
                                    cos,
                                    sin,
                                    k_cache,
                                    k_s_cache[..., 0],
                                    cu_seqlen_q=meta.cu_seqlen_q,
                                    kv_seqlens=meta.k_seqlens,
                                    block_offsets=meta.block_offset,
                                    max_q_seqlen=meta.max_q_seqlen,
                                    eps=norm_eps,
                                    rope_interleaved=rope_interleaved)
        return self._forward_index(q, q_s, k_cache, k_s_cache, meta)

class TritonNSAIndexFP8Builder(BaseNSAIndexFP8Builder):

    @staticmethod
    def build(topk: int, softmax_scale: float, block_size: int = 128, fill: int = -1) -> BaseNSAIndexFP8:
        return TritonNSAIndexFP8(topk, softmax_scale=softmax_scale, block_size=block_size, fill=fill)
