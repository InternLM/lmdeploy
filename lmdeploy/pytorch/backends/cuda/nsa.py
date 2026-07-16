# Copyright (c) OpenMMLab. All rights reserved.
import functools

from torch import Tensor

from lmdeploy.pytorch.kernels.cuda.bitonic_topk import bitonic_topk
from lmdeploy.pytorch.kernels.cuda.blocked_gemm_fp8 import quant_fp8
from lmdeploy.pytorch.kernels.cuda.ds_index import fp8_index
from lmdeploy.pytorch.kernels.cuda.dsa_indexer import prepare_dsa_indexer_k_cache, prepare_dsa_indexer_q
from lmdeploy.pytorch.kernels.cuda.fill_kv_cache import fill_kv_cache_blocked_fp8

from ..nsa import BaseNSAIndexFP8, BaseNSAIndexFP8Builder, NSAIndexMeta


@functools.lru_cache
def _get_sparse_index_topk(topk: int):
    from lmdeploy.pytorch.kernels.cuda.sparse_index_topk import (
        is_sparse_index_topk_supported,
        sparse_index_topk,
    )
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
        sparse_index_topk = _get_sparse_index_topk(self.topk)
        if sparse_index_topk is not None:
            return sparse_index_topk(scores,
                                     q_seqlens,
                                     k_seqlens,
                                     self.topk,
                                     fill=self.fill,
                                     descending=True,
                                     sorted=False)
        return bitonic_topk(scores, q_seqlens, k_seqlens, self.topk, fill=self.fill, descending=True)

    def forward(self, q: Tensor, k: Tensor, weights: Tensor, k_cache: Tensor, k_s_cache: Tensor,
                meta: NSAIndexMeta) -> Tensor:
        assert q.dim() == 3
        assert k.dim() == 2
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
                      sin: Tensor, k_cache: Tensor, k_s_cache: Tensor, norm_eps: float, head_gate_scale: float,
                      rope_interleaved: bool, meta: NSAIndexMeta) -> Tensor:
        """Prepare FP8 Q and write K cache without allocating rotated BF16
        Q/K."""
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
