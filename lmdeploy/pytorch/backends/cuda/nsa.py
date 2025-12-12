# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor

from lmdeploy.pytorch.kernels.cuda.bitonic_topk import bitonic_topk
from lmdeploy.pytorch.kernels.cuda.blocked_gemm_fp8 import quant_fp8
from lmdeploy.pytorch.kernels.cuda.ds_index import fp8_index
from lmdeploy.pytorch.kernels.cuda.fill_kv_cache import fill_kv_cache_blocked_fp8

from ..nsa import BaseNSAIndexFP8, BaseNSAIndexFP8Builder, NSAIndexMeta


class TritonNSAIndexFP8(BaseNSAIndexFP8):

    def __init__(self, topk: int, softmax_scale: float, block_size: int, fill: int) -> None:
        super().__init__()
        self.topk = topk
        self.softmax_scale = softmax_scale
        self.block_size = block_size
        self.fill = fill

    def forward(self, q: Tensor, k: Tensor, weights: Tensor, k_cache: Tensor, k_s_cache: Tensor,
                meta: NSAIndexMeta) -> Tensor:

        assert q.dim() == 3
        assert k.dim() == 2
        cu_seqlen_q = meta.cu_seqlen_q
        q_seqlens = meta.q_seqlens
        k_seqlens = meta.k_seqlens
        block_offset = meta.block_offset
        max_q_seqlen = meta.max_q_seqlen
        max_kv_seqlen = meta.max_kv_seqlen

        q_shape = q.shape
        q = q.reshape(-1, q_shape[-1])
        q, q_s = quant_fp8(q, self.block_size, dtype=k_cache.dtype, trans_scale=True)
        q = q.reshape(*q_shape)
        q_s = q_s.reshape(weights.shape)
        q_s = q_s * self.softmax_scale * weights

        fill_kv_cache_blocked_fp8(k[:, None],
                                  None,
                                  k_cache[..., None, :],
                                  None,
                                  k_s_cache[..., None, :],
                                  None,
                                  cu_seqlen_q=cu_seqlen_q,
                                  kv_seqlens=k_seqlens,
                                  max_q_seqlen=max_q_seqlen,
                                  block_offsets=block_offset,
                                  group_size=self.block_size)

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
        return bitonic_topk(scores, q_seqlens, k_seqlens, self.topk, fill=self.fill, descending=True)


class TritonNSAIndexFP8Builder(BaseNSAIndexFP8Builder):

    @staticmethod
    def build(topk: int, softmax_scale: float, block_size: int = 128, fill: int = -1) -> BaseNSAIndexFP8:
        return TritonNSAIndexFP8(topk, softmax_scale=softmax_scale, block_size=block_size, fill=fill)
