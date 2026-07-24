# Copyright (c) OpenMMLab. All rights reserved.
import random

import torch

from lmdeploy.pytorch.kernels.cuda.bitonic_topk import bitonic_topk
from lmdeploy.pytorch.kernels.cuda.blocked_gemm_fp8 import quant_fp8
from lmdeploy.pytorch.kernels.cuda.ds_index import fp8_index
from lmdeploy.pytorch.kernels.cuda.sparse_index_topk import (
    is_sparse_index_topk_supported,
    sparse_index_topk,
)
from lmdeploy.utils import get_logger

from ..indexer import BaseV4Indexer, BaseV4IndexerBuilder, V4IndexerMetadata, V4IndexerOutput
from .warmup_manager import get_warmup_manager

logger = get_logger('lmdeploy')


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


class _V4PagedMQALogitsWarmup:
    """Warm DeepGEMM paged-MQA score-generation kernels used by V4 indexer."""

    _METADATA_ROW_ALIGNMENT = 32

    @staticmethod
    def _get_shape(model_config) -> tuple[int, int, int] | None:
        hf_config = getattr(model_config, 'hf_config', None)
        if getattr(hf_config, 'model_type', None) != 'deepseek_v4':
            return None
        if 4 not in getattr(hf_config, 'compress_ratios', ()):
            return None

        block_size = getattr(model_config, 'block_size', 256)
        entries_per_block = block_size // 4
        num_heads = getattr(hf_config, 'index_n_heads', None)
        head_dim = getattr(hf_config, 'index_head_dim', 128)
        if num_heads not in (32, 64):
            logger.warning(
                'Skip DeepGEMM V4 paged-MQA warmup: unsupported index_n_heads=%s.',
                num_heads)
            return None
        if entries_per_block not in (32, 64):
            logger.warning(
                'Skip DeepGEMM V4 paged-MQA warmup: unsupported entries_per_block=%s.',
                entries_per_block)
            return None
        return entries_per_block, num_heads, head_dim

    def _warm_metadata(self, deep_gemm, entries_per_block: int, max_rows: int):
        alignment = self._METADATA_ROW_ALIGNMENT
        max_aligned_rows = _align(max_rows, alignment)
        rows = list(range(alignment, max_aligned_rows + 1, alignment))
        random.shuffle(rows)
        for row_count in rows:
            context_lens = torch.zeros(row_count, 1, dtype=torch.int32, device='cuda')
            deep_gemm.get_paged_mqa_logits_metadata(
                context_lens, entries_per_block, deep_gemm.get_num_sms())

    def _warm_logits(self, deep_gemm, entries_per_block: int, num_heads: int,
                     head_dim: int, max_context_len: int):
        row_count = self._METADATA_ROW_ALIGNMENT
        max_context_len = max(max_context_len, 1)
        max_blocks = (max_context_len + entries_per_block - 1) // entries_per_block
        context_lens = torch.full((row_count, 1), max_context_len, dtype=torch.int32, device='cuda')
        block_table = torch.arange(max_blocks, dtype=torch.int32, device='cuda').repeat(row_count, 1)
        schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
            context_lens, entries_per_block, deep_gemm.get_num_sms())
        q = torch.empty(
            row_count, 1, num_heads, head_dim,
            dtype=torch.float8_e4m3fn, device='cuda')
        weights = torch.empty(row_count, num_heads, dtype=torch.float32, device='cuda')
        fused_kv_cache = torch.empty(
            max_blocks, entries_per_block, 1, head_dim + 4,
            dtype=torch.uint8, device='cuda')
        deep_gemm.fp8_paged_mqa_logits(
            q, fused_kv_cache, weights, context_lens, block_table,
            schedule_metadata, max_context_len, False)

    def warmup(self, warmup_meta):
        import deep_gemm

        shape = self._get_shape(warmup_meta.model_config)
        if shape is None:
            return
        entries_per_block, num_heads, head_dim = shape

        max_rows = max(warmup_meta.max_num_tokens, warmup_meta.max_batch_size)
        max_context_len = max(warmup_meta.max_num_tokens // 4, 1)
        logger.info(
            'Warming up DeepGEMM V4 paged-MQA logits: rows<=%s stride=%s, '
            'heads=%s, head_dim=%s, entries_per_block=%s.',
            _align(max_rows, self._METADATA_ROW_ALIGNMENT),
            self._METADATA_ROW_ALIGNMENT,
            num_heads,
            head_dim,
            entries_per_block)
        self._warm_metadata(deep_gemm, entries_per_block, max_rows)
        self._warm_logits(
            deep_gemm, entries_per_block, num_heads, head_dim, max_context_len)


class TritonV4IndexerImpl(BaseV4Indexer):

    def __init__(self, index_topk: int, compress_ratio: int) -> None:
        super().__init__()
        self.index_topk = index_topk
        self.compress_ratio = compress_ratio
        if compress_ratio == 4:
            warmup_mgr = get_warmup_manager()
            key = 'deepgemm_v4_paged_mqa_logits'
            if key not in warmup_mgr:
                warmup_mgr[key] = _V4PagedMQALogitsWarmup().warmup

    def forward(self,
                query: torch.Tensor,
                weights: torch.Tensor,
                index_kv_cache: torch.Tensor,
                index_kv_scale_cache: torch.Tensor | None,
                meta: V4IndexerMetadata) -> V4IndexerOutput:
        block_offsets = meta.block_offsets
        cu_q_seqlens = meta.cu_q_seqlens
        kv_seqlens = meta.kv_seqlens
        q_seqlens = meta.q_seqlens
        block_size = meta.block_size
        if block_size is None:
            raise RuntimeError('V4IndexerMetadata.block_size is required.')

        # Reshape to fp8_index expected layout upfront.
        # query: [bsz, seqlen, n_heads, head_dim] -> [cum_seqlen, n_heads, head_dim]
        # weights: [bsz, seqlen, n_heads] -> [cum_seqlen, n_heads]
        q_3d = query.flatten(0, 1)
        weights_2d = weights.flatten(0, -2)

        # FP8 quantize Indexer Q directly on 3D (replaces fp4_act_quant for better precision)
        q_2d = q_3d.reshape(-1, q_3d.size(-1) * q_3d.size(-2))
        q_fp8, q_scale_2d = quant_fp8(q_2d, group_size=128,
                                       dtype=torch.float8_e4m3fn, scale_fmt='ue8m0')
        q_3d = q_fp8.view_as(q_3d)
        q_scale = q_scale_2d.view(q_3d.shape[:-1])  # [cum_seqlen, n_heads]
        q_scale_weighted = q_scale * weights_2d

        total_lens = kv_seqlens
        # Use pre-computed num_index from metadata (avoid per-layer torch.div).
        # Prefer ratio-specific field, then generic num_index, then compute on-the-fly.
        if self.compress_ratio == 4 and meta.num_index_r4 is not None:
            num_index = meta.num_index_r4
        elif self.compress_ratio == 128 and meta.num_index_r128 is not None:
            num_index = meta.num_index_r128
        elif meta.num_index is not None:
            num_index = meta.num_index
        else:
            num_index = torch.div(total_lens, self.compress_ratio, rounding_mode='floor')
        if meta.is_decoding:
            # Keep CUDA graph shapes static without using the whole GPU cache pool
            # as the score width. The block table width is fixed for the captured
            # graph bucket and is the real per-sequence addressable upper bound.
            max_kv_seqlen = block_offsets.size(1) * block_size
        else:
            max_kv_seqlen = meta.max_kv_seqlen
            if max_kv_seqlen is None:
                max_kv_seqlen = block_offsets.size(1) * block_size
        max_index = max(max_kv_seqlen // self.compress_ratio, 1)

        if index_kv_cache.dtype == torch.uint8:
            import deep_gemm

            score_meta = getattr(meta, 'index_score_meta', None)
            if (score_meta is None or score_meta.context_lens is None or score_meta.block_offsets is None
                    or score_meta.schedule_metadata is None or score_meta.max_k_seqlen is None):
                raise RuntimeError('Packed V4 index scoring requires CUDA index-score metadata.')
            q_mqa = q_3d.view(q_3d.size(0), 1, q_3d.size(1), q_3d.size(2))
            scores = deep_gemm.fp8_paged_mqa_logits(
                q_mqa,
                index_kv_cache,
                q_scale_weighted,
                score_meta.context_lens,
                score_meta.block_offsets,
                score_meta.schedule_metadata,
                score_meta.max_k_seqlen,
                False)
            topk_seqlens = score_meta.context_lens.squeeze(-1)
        else:
            # fp8_index fallback for the legacy split value/scale cache layout.
            if index_kv_scale_cache is None:
                raise RuntimeError('Legacy V4 index cache scoring requires a scale cache.')
            k_cache = index_kv_cache
            k_s_cache = index_kv_scale_cache.squeeze(-1)
            if meta.is_decoding:
                scores = fp8_index(q_3d, q_scale_weighted,
                                   k_cache, k_s_cache,
                                   cu_q_seqlens, num_index, block_offsets,
                                   max_q_seqlen=meta.max_q_seqlen, max_k_seqlen=max_index, causal=True)
                topk_seqlens = num_index
            else:
                scores, topk_seqlens = fp8_index(q_3d, q_scale_weighted,
                                                 k_cache, k_s_cache,
                                                 cu_q_seqlens, num_index, block_offsets,
                                                 max_q_seqlen=meta.max_q_seqlen,
                                                 max_k_seqlen=max_index,
                                                 causal=True,
                                                 raw_k_seqlens=total_lens,
                                                 compress_ratio=self.compress_ratio,
                                                 return_row_k_seqlens=True,
                                                 trim_causal_tail=True)

        topk_width = self.index_topk
        topk_length = topk_seqlens.clamp(max=topk_width)
        if is_sparse_index_topk_supported(topk_width) and scores.dtype == torch.float32:
            topk = sparse_index_topk(scores, q_seqlens, topk_seqlens,
                                     k=topk_width, fill=-1, descending=True)
        else:
            topk = bitonic_topk(scores, q_seqlens, topk_seqlens,
                                k=topk_width, fill=-1, descending=True)

        # Always return [total_q, topk_width] — caller handles decode/prefill dimension adaptation
        return V4IndexerOutput(indices_in_kvcache=topk, topk_length=topk_length)


class TritonV4IndexerBuilder(BaseV4IndexerBuilder):

    @staticmethod
    def build(index_topk: int, compress_ratio: int) -> BaseV4Indexer:
        return TritonV4IndexerImpl(index_topk=index_topk, compress_ratio=compress_ratio)
