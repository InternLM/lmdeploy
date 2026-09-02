# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor

from lmdeploy.utils import get_logger
from lmdeploy.pytorch.kernels.dlinfer import fill_kv_cache, lightning_indexer

from ..nsa import BaseNSAIndexFP8, BaseNSAIndexFP8Builder, NSAIndexMeta

logger = get_logger('lmdeploy')


class DlinferNSAIndexBF16(BaseNSAIndexFP8):
    """Ascend BF16 implementation of the legacy NSAIndexFP8 interface."""

    supports_fused_preprocess = False
    requires_unfused_hadamard = False

    def __init__(self, topk: int, softmax_scale: float, block_size: int,
                 fill: int):
        self.topk = topk
        self.softmax_scale = softmax_scale
        self.block_size = block_size

    def get_step_metadata(self, attn_metadata) -> NSAIndexMeta:
        """Adapt dlinfer attention metadata to the backend-neutral NSA form."""
        if isinstance(attn_metadata, NSAIndexMeta):
            return attn_metadata

        q_seqlens = attn_metadata.q_seqlens
        kv_seqlens = attn_metadata.kv_seqlens
        cu_seqlen_q = attn_metadata.cu_seqlens_q
        cu_seqlen_k = getattr(attn_metadata, 'cu_seqlens_k', None)
        if cu_seqlen_k is None:
            cu_seqlen_k = torch.cat((kv_seqlens.new_zeros(1), kv_seqlens.cumsum(0)))
        block_size = getattr(attn_metadata, 'block_size', self.block_size)
        max_q_seqlen = getattr(attn_metadata, 'max_q_seqlen', None)
        if max_q_seqlen is None:
            max_q_seqlen = int(q_seqlens.max().item())
        max_kv_seqlen = getattr(attn_metadata, 'max_kv_seq_len', None)
        if max_kv_seqlen is None:
            max_kv_seqlen = int(kv_seqlens.max().item())
        return NSAIndexMeta(
            cu_seqlen_q=cu_seqlen_q,
            q_seqlens=q_seqlens,
            k_seqlens=kv_seqlens,
            cu_seqlen_k=cu_seqlen_k,
            block_offset=attn_metadata.block_offsets,
            max_q_seqlen=max_q_seqlen,
            max_kv_seqlen=max_kv_seqlen,
            kv_flatten_size=getattr(attn_metadata, 'kv_flatten_size', None),
            block_size=block_size,
            is_decoding=attn_metadata.is_decoding,
            kv_start_indices=getattr(attn_metadata, 'kv_start_indices', None),
        )

    def _forward_index(
        self,
        q: Tensor,
        k: Tensor,
        weights: Tensor,
        k_cache: Tensor,
        meta: NSAIndexMeta,
    ) -> Tensor:
        if meta.kv_start_indices is None:
            raise RuntimeError(
                "Ascend NSA metadata is missing kv_start_indices")
        if meta.cu_seqlen_q is None:
            raise RuntimeError("Ascend NSA metadata is missing cu_seqlen_q")

        k_cache = k_cache.unsqueeze(-2)
        k = k.unsqueeze(-2)
        fill_kv_cache(
            k,
            k,
            k_cache,
            k_cache,
            meta.kv_start_indices,
            k_scales_zeros=(),
            v_scales_zeros=(),
            quant_bits=0,
        )

        actual_q = meta.cu_seqlen_q[1:]
        actual_k = meta.k_seqlens
        block_offset = meta.block_offset
        indices = lightning_indexer(
            q.contiguous(),
            k_cache,
            weights.to(dtype=q.dtype).contiguous(),
            actual_seq_lengths_query=actual_q,
            actual_seq_lengths_key=actual_k,
            block_table=block_offset,
            sparse_count=self.topk,
        )
        if indices.dim() != 3 or indices.size(1) != 1:
            raise RuntimeError(
                "Ascend Lightning Indexer returned an unexpected shape: "
                f"{tuple(indices.shape)}")
        return indices.squeeze(1)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        weights: Tensor,
        k_cache: Tensor,
        meta: NSAIndexMeta,
    ) -> Tensor:
        return self._forward_index(q, k, weights, k_cache, meta)

    def forward_fused(
        self,
        q: Tensor,
        k: Tensor,
        weights: Tensor,
        norm_weight: Tensor,
        norm_bias: Tensor,
        cos: Tensor,
        sin: Tensor,
        k_cache: Tensor,
        norm_eps: float,
        head_gate_scale: float,
        rope_interleaved: bool,
        meta: NSAIndexMeta,
    ) -> Tensor:
        raise NotImplementedError(
            'DSA indexer fused preprocessing is not supported on Ascend.')


class DlinferNSAIndexBF16Builder(BaseNSAIndexFP8Builder):

    @staticmethod
    def build(
        topk: int,
        softmax_scale: float,
        block_size: int = 128,
        fill: int = -1,
        allow_short_prefill_scoring_skip: bool = False,
    ) -> BaseNSAIndexFP8:
        logger.warning('Ascend backend does not support FP8 indexer; '
                       'falling back to BF16 indexer.')
        return DlinferNSAIndexBF16(topk, softmax_scale, block_size, fill)
