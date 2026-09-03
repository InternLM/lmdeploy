# Copyright (c) OpenMMLab. All rights reserved.
import functools
from collections.abc import Hashable
from dataclasses import dataclass

import torch
from torch import Tensor

from lmdeploy.pytorch import envs as _envs
from lmdeploy.pytorch.backends.cuda.step_metadata import (
    CudaAttentionMetaBuilder,
    CudaSequenceMetadata,
    register_step_metadata_impl,
)
from lmdeploy.pytorch.consts import (
    DSA_INDEX_SCALE_BYTES,
    DSA_INDEXER_K_CACHE_NAME,
    dsa_packed_indexer_k_cache_shape,
)
from lmdeploy.pytorch.engine.cache_engine.schema import BlockCacheGeometry, BlockCacheRequest
from lmdeploy.pytorch.kernels.cuda.bitonic_topk import bitonic_topk
from lmdeploy.pytorch.kernels.cuda.blocked_gemm_fp8 import quant_fp8
from lmdeploy.pytorch.kernels.cuda.ds_index import fp8_index
from lmdeploy.pytorch.kernels.cuda.dsa_indexer_preprocess import (
    flatten_dsa_indexer_k_cache,
    prepare_dsa_indexer_k_cache,
    prepare_dsa_indexer_q,
)
from lmdeploy.pytorch.kernels.cuda.fill_kv_cache import fill_kv_cache_blocked_fp8
from lmdeploy.pytorch.kernels.cuda.step_metadata.fill_dsa_indexer_metadata import (
    fill_dsa_indexer_metadata,
)
from lmdeploy.utils import get_logger

from ..nsa import (
    BaseNSAIndexFP8,
    BaseNSAIndexFP8Builder,
    NSAIndexMeta,
    build_nsa_index_meta,
    should_skip_nsa_indexer,
)

logger = get_logger('lmdeploy')


def _get_max_score_rows(max_kv_seqlen: int, max_logits_bytes: int) -> int:
    """Return the query rows fitting in a bounded FP32 score tensor."""
    if max_kv_seqlen <= 0:
        return 1
    # DeepGEMM materializes an aligned [query_rows, max_kv_seqlen] output
    # before top-k selection:
    # https://github.com/deepseek-ai/DeepGEMM/blob/88965b078186ee7510ab9fc4f1d5ebc19adfa8d1/csrc/apis/attention.hpp#L155-L171
    # Bounding flattened KV alone therefore does not bound the M * N logits
    # allocation; limit M so its FP32 payload stays within the runtime budget.
    _fp32_bytes = 4
    return max(1, max_logits_bytes // (max_kv_seqlen * _fp32_bytes))


def _get_dsa_indexer_k_cache_views(indexer_k_cache: Tensor,
                                   head_dim: int) -> tuple[Tensor, Tensor]:
    """Return FP8 K and FP32 scale views of a packed DSA indexer-K cache.

    Raw block layout: ``[all FP8 K][all FP32 scales]``.
    """
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
    values = flat[:, :value_bytes].view(torch.float8_e4m3fn).view(
        num_blocks, entries_per_block, head_dim)
    scales = flat[:, value_bytes:value_bytes + scale_bytes].view(
        torch.float32).view(num_blocks, entries_per_block, 1)
    return values, scales


@dataclass
class _DeepGemmPagedScoreMeta:
    """Layer-invariant metadata consumed by paged MQA scoring."""
    context_lens: Tensor
    block_offsets: Tensor
    schedule: Tensor
    max_kv_seqlen: int


@dataclass
class _DeepGemmContiguousScoreMeta:
    """Contiguous-MQA metadata shared by prefill indexer layers."""
    k_starts: Tensor
    k_ends: Tensor
    max_kv_seqlen: int


_DeepGemmScoreMeta = _DeepGemmPagedScoreMeta | _DeepGemmContiguousScoreMeta


@dataclass
class _DSAIndexerGraphBuffer:
    """Stable graph tensors referenced by DSA indexer metadata.

    Single-token graphs alias the common ``kv_seqlens`` input buffer. Multi-
    token graphs use dedicated indexer lengths and, when required, an expanded
    block table.
    """
    indexer_kv_seqlens: Tensor
    expanded_block_offsets: Tensor | None
    schedule: Tensor | None


@functools.lru_cache
def _get_deep_gemm():
    try:
        import deep_gemm
    except ImportError:
        return None
    required = ('fp8_fp4_mqa_logits', 'fp8_fp4_paged_mqa_logits',
                'get_paged_mqa_logits_metadata', 'get_num_sms')
    if not all(hasattr(deep_gemm, name) for name in required):
        return None
    return deep_gemm


@functools.lru_cache
def _warn_triton_index_scoring():
    logger.warning(
        'DSA index scoring is using the Triton FP8 index kernel instead of '
        'DeepGEMM MQA logits.')


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


def _build_deep_gemm_score_meta(
        meta: NSAIndexMeta,
        expanded_block_offsets: Tensor | None = None,
        schedule_buffer: Tensor | None = None
) -> _DeepGemmScoreMeta | None:
    """Build layer-invariant DeepGEMM index-scoring metadata."""
    deep_gemm = _get_deep_gemm()
    if deep_gemm is None or not meta.block_offset.is_cuda:
        return None

    if not meta.is_decoding:
        k_starts = torch.repeat_interleave(
            meta.cu_seqlen_k[:-1],
            meta.q_seqlens,
            output_size=meta.indexer_kv_seqlens.numel(),
        ).to(torch.int32)
        return _DeepGemmContiguousScoreMeta(
            k_starts=k_starts,
            k_ends=k_starts + meta.indexer_kv_seqlens,
            max_kv_seqlen=meta.max_kv_seqlen,
        )

    # DeepGEMM expects context lengths in [batch, next_n] layout.
    context_lens = meta.indexer_kv_seqlens.unsqueeze(-1)
    block_offsets = meta.block_offset
    if context_lens.size(0) != block_offsets.size(0):
        if expanded_block_offsets is None:
            expanded_block_offsets = torch.repeat_interleave(
                block_offsets,
                meta.q_seqlens,
                dim=0,
                output_size=context_lens.size(0),
            )
        block_offsets = expanded_block_offsets
    if block_offsets.dtype != torch.int32:
        block_offsets = block_offsets.to(torch.int32)

    schedule = deep_gemm.get_paged_mqa_logits_metadata(
        context_lens, meta.block_size, deep_gemm.get_num_sms())
    if schedule_buffer is not None:
        schedule_buffer.copy_(schedule)
        schedule = schedule_buffer
    return _DeepGemmPagedScoreMeta(
        context_lens=context_lens,
        block_offsets=block_offsets,
        schedule=schedule,
        max_kv_seqlen=meta.block_offset.size(1) * meta.block_size,
    )


@dataclass(frozen=True)
class DSAIndexerMetaBuilder(
        CudaAttentionMetaBuilder[_DSAIndexerGraphBuffer, NSAIndexMeta | None]):
    """Own per-step and CUDA graph metadata for the selected DSA indexer."""

    @property
    def key(self) -> Hashable:
        return type(self)

    def build(self, step_context,
              sequence_metadata: CudaSequenceMetadata) -> NSAIndexMeta | None:
        if should_skip_nsa_indexer(step_context.model_metas):
            return None
        cache_config = step_context.cache_config
        num_tokens = step_context.input_ids.size(1)
        is_multi_token_decode = (step_context.is_decoding
                                 and step_context.max_q_seqlen > 1)
        indexer_kv_seqlens = None
        expanded_block_offsets = None
        if is_multi_token_decode:
            indexer_kv_seqlens = torch.empty(
                num_tokens,
                dtype=torch.int32,
                device=sequence_metadata.q_seqlens.device,
            )
            if _get_deep_gemm() is not None:
                expanded_block_offsets = torch.empty(
                    num_tokens,
                    sequence_metadata.block_offsets.size(1),
                    dtype=torch.int32,
                    device=sequence_metadata.block_offsets.device,
                )
            fill_dsa_indexer_metadata(
                sequence_metadata.q_seqlens,
                sequence_metadata.kv_seqlens,
                sequence_metadata.cu_seqlens_q,
                sequence_metadata.block_offsets,
                indexer_kv_seqlens,
                expanded_block_offsets,
                num_tokens,
                step_context.max_q_seqlen,
            )
        meta = build_nsa_index_meta(
            num_tokens=num_tokens,
            is_decoding=step_context.is_decoding,
            block_size=cache_config.block_size,
            num_gpu_blocks=cache_config.num_gpu_blocks,
            sequence_metadata=sequence_metadata,
            indexer_kv_seqlens=indexer_kv_seqlens,
        )
        meta.score_meta = _build_deep_gemm_score_meta(
            meta,
            expanded_block_offsets=expanded_block_offsets,
        )
        return meta

    def apply_legacy_metadata(self, attn_metadata,
                              metadata: NSAIndexMeta | None) -> None:
        pass

    def make_cudagraph_buffer(self, graph_meta, input_buffers,
                              step_context) -> _DSAIndexerGraphBuffer:
        deep_gemm = _get_deep_gemm()
        expanded_block_offsets = None
        schedule = None
        if deep_gemm is not None:
            if graph_meta.decode_query_len > 1:
                expanded_block_offsets = torch.empty(
                    graph_meta.max_tokens,
                    graph_meta.num_blocks,
                    dtype=torch.int32,
                    device=graph_meta.device,
                )
            schedule = torch.empty(
                deep_gemm.get_num_sms() + 1,
                2,
                dtype=torch.int32,
                device=graph_meta.device,
            )
        if graph_meta.decode_query_len == 1:
            indexer_kv_seqlens = input_buffers['kv_seqlens']
        else:
            indexer_kv_seqlens = torch.empty(
                graph_meta.max_tokens,
                dtype=torch.int32,
                device=graph_meta.device,
            )
        return _DSAIndexerGraphBuffer(
            indexer_kv_seqlens=indexer_kv_seqlens,
            expanded_block_offsets=expanded_block_offsets,
            schedule=schedule,
        )

    def fill_cudagraph_buffer(self, graph_meta, input_buffers, step_context,
                              buffer: _DSAIndexerGraphBuffer) -> NSAIndexMeta | None:
        if should_skip_nsa_indexer(step_context.model_metas):
            return None
        sequence_metadata = CudaSequenceMetadata(
            block_offsets=input_buffers['block_offsets'],
            q_start_loc=input_buffers['q_start_loc'],
            q_seqlens=input_buffers['q_seqlens'],
            kv_start_loc=None,
            kv_seqlens=input_buffers['kv_seqlens'],
            kv_flatten_size=None,
            cu_seqlens_q=input_buffers['cu_seqlens_q'],
            cu_seqlens_k=input_buffers['cu_seqlens_k'],
            max_kv_seqlen=graph_meta.num_blocks * graph_meta.block_size,
        )
        if graph_meta.decode_query_len > 1:
            fill_dsa_indexer_metadata(
                sequence_metadata.q_seqlens,
                sequence_metadata.kv_seqlens,
                sequence_metadata.cu_seqlens_q,
                sequence_metadata.block_offsets,
                buffer.indexer_kv_seqlens,
                buffer.expanded_block_offsets,
                graph_meta.max_tokens,
                graph_meta.decode_query_len,
            )
        meta = build_nsa_index_meta(
            num_tokens=graph_meta.max_tokens,
            is_decoding=True,
            block_size=step_context.cache_config.block_size,
            num_gpu_blocks=step_context.cache_config.num_gpu_blocks,
            sequence_metadata=sequence_metadata,
            indexer_kv_seqlens=buffer.indexer_kv_seqlens,
        )
        meta.score_meta = _build_deep_gemm_score_meta(
            meta,
            expanded_block_offsets=buffer.expanded_block_offsets,
            schedule_buffer=buffer.schedule,
        )
        return meta


class TritonNSAIndexFP8(BaseNSAIndexFP8):

    def __init__(self, topk: int, softmax_scale: float, block_size: int,
                 fill: int,
                 allow_short_prefill_scoring_skip: bool = False) -> None:
        super().__init__()
        self.topk = topk
        self.softmax_scale = softmax_scale
        self.block_size = block_size
        self.fill = fill
        self._allow_short_prefill_scoring_skip = allow_short_prefill_scoring_skip
        # TODO: configable scale fmt
        self.scale_fmt = 'ue8m0'
        self.max_logits_bytes = _envs.dsa_indexer_max_logits_mb * (1 << 20)
        self._sparse_index_topk = _get_sparse_index_topk(topk)
        self._step_meta_group: int | None = None
        register_step_metadata_impl(self)

    def get_block_cache_requests(self, geometry: BlockCacheGeometry,
                                 head_dim: int) -> tuple[BlockCacheRequest, ...]:
        """Request one DeepGEMM-compatible packed cache row per indexer."""
        if geometry.logical_block_size != geometry.kernel_block_size:
            raise ValueError(
                'DSA indexer cache requires equal logical and kernel block sizes, '
                f'got {geometry.logical_block_size} and {geometry.kernel_block_size}.')
        request = BlockCacheRequest(
            name=DSA_INDEXER_K_CACHE_NAME,
            shape=dsa_packed_indexer_k_cache_shape(geometry.kernel_block_size, head_dim),
            dtype=torch.uint8,
            per_row_contiguous=True,
        )
        return (request, )

    def _should_skip_scoring(self, meta: NSAIndexMeta) -> bool:
        """Whether dense prefill makes index scoring unnecessary."""
        return (self._allow_short_prefill_scoring_skip and not meta.is_decoding
                and meta.max_kv_seqlen <= self.topk)

    def _maybe_score_and_select(self, q: Tensor, q_s: Tensor,
                                indexer_k_cache: Tensor,
                                meta: NSAIndexMeta) -> Tensor | None:
        """Score after the caller has preserved K for future decode."""
        if self._should_skip_scoring(meta):
            return None
        return self._score_and_select(q, q_s, indexer_k_cache, meta)

    def get_step_metadata_provider(self):
        """Describe metadata required by the selected DSA indexer."""
        return DSAIndexerMetaBuilder()

    def bind_step_meta_group(self, group_id: int) -> None:
        """Bind this implementation to its deduplicated metadata group."""
        self._step_meta_group = group_id

    def get_step_metadata(self, attn_metadata) -> NSAIndexMeta:
        """Return the DSA metadata prepared for this operator group."""
        assert self._step_meta_group is not None
        meta = attn_metadata.kernel_metadata[self._step_meta_group]
        assert isinstance(meta, NSAIndexMeta)
        return meta

    def _flatten_prefill_k(self, indexer_k_cache: Tensor, head_dim: int,
                           meta: NSAIndexMeta) -> tuple[Tensor, Tensor]:
        """Flatten the paged indexer K cache once for prefill scoring."""
        k_cache, k_s_cache = _get_dsa_indexer_k_cache_views(
            indexer_k_cache, head_dim)
        return flatten_dsa_indexer_k_cache(
            k_cache,
            k_s_cache[..., 0],
            meta.cu_seqlen_k,
            meta.k_seqlens,
            meta.block_offset,
            out_size=meta.kv_flatten_size,
        )

    def _compute_prefill_scores(
            self,
            q: Tensor,
            q_s: Tensor,
            flat_k: Tensor,
            flat_k_s: Tensor,
            score_meta: _DeepGemmContiguousScoreMeta,
            row_slice: slice = slice(None)) -> Tensor:
        """Compute one DeepGEMM prefill score-row slice."""
        return _get_deep_gemm().fp8_fp4_mqa_logits(
            q=(q, None),
            kv=(flat_k, flat_k_s),
            weights=q_s,
            cu_seq_len_k_start=score_meta.k_starts[row_slice],
            cu_seq_len_k_end=score_meta.k_ends[row_slice],
            clean_logits=False,
            max_seqlen_k=score_meta.max_kv_seqlen,
            logits_dtype=torch.float32,
        )

    def _select_topk(self, scores: Tensor, meta: NSAIndexMeta,
                     row_slice: slice = slice(None)) -> Tensor:
        """Select sparse-attention positions from dense index scores."""
        kv_seqlens = meta.indexer_kv_seqlens[row_slice]
        # Both selectors consume q_seqlens only when kv_seqlens still has one
        # entry per request. DSA metadata already expands it to one entry per
        # score row, including for a query-row chunk.
        if self._sparse_index_topk is not None:
            return self._sparse_index_topk(scores,
                                           meta.q_seqlens,
                                           kv_seqlens,
                                           self.topk,
                                           fill=self.fill,
                                           descending=True,
                                           sorted=False)
        return bitonic_topk(scores,
                            meta.q_seqlens,
                            kv_seqlens,
                            self.topk,
                            fill=self.fill,
                            descending=True)

    def _score_and_select_prefill(
            self, q: Tensor, q_s: Tensor, indexer_k_cache: Tensor,
            meta: NSAIndexMeta,
            score_meta: _DeepGemmContiguousScoreMeta) -> Tensor:
        """Bound prefill score memory by chunking only the query rows."""
        # Keep KV contiguous for DeepGEMM's TMA descriptors; slicing only Q
        # avoids the alignment failures caused by per-request KV views.
        flat_k, flat_k_s = self._flatten_prefill_k(
            indexer_k_cache, q.size(-1), meta)
        max_rows = _get_max_score_rows(score_meta.max_kv_seqlen,
                                       self.max_logits_bytes)
        num_rows = q.size(0)
        if num_rows <= max_rows:
            scores = self._compute_prefill_scores(
                q, q_s, flat_k, flat_k_s, score_meta)
            return self._select_topk(scores, meta)

        logger.debug('Split DSA prefill scores into %d chunks with at most %d query rows.',
                     (num_rows + max_rows - 1) // max_rows, max_rows)
        out = torch.empty((num_rows, self.topk),
                          dtype=torch.int32,
                          device=q.device)
        for start in range(0, num_rows, max_rows):
            end = min(start + max_rows, num_rows)
            row_slice = slice(start, end)
            scores = self._compute_prefill_scores(
                q[row_slice], q_s[row_slice], flat_k, flat_k_s,
                score_meta, row_slice)
            selected = self._select_topk(scores, meta, row_slice)
            out[row_slice].copy_(selected)
            del scores, selected
        return out

    def _score_and_select(self, q: Tensor, q_s: Tensor,
                          indexer_k_cache: Tensor, meta: NSAIndexMeta) -> Tensor:
        score_meta = meta.score_meta
        if isinstance(score_meta, _DeepGemmContiguousScoreMeta):
            return self._score_and_select_prefill(
                q, q_s, indexer_k_cache, meta, score_meta)

        if isinstance(score_meta, _DeepGemmPagedScoreMeta):
            # Paged MQA reads the packed cache directly and requires its compact
            # ``entries * (D + 4)`` byte block stride.
            scores = _get_deep_gemm().fp8_fp4_paged_mqa_logits(
                q=(q[:, None], None),
                kv_cache=indexer_k_cache,
                weights=q_s,
                context_lens=score_meta.context_lens,
                block_table=score_meta.block_offsets,
                schedule_meta=score_meta.schedule,
                max_context_len=score_meta.max_kv_seqlen,
                clean_logits=False,
                logits_dtype=torch.float32,
            )
        else:
            _warn_triton_index_scoring()
            score_bytes = q.size(0) * meta.max_kv_seqlen * 4
            if score_bytes > self.max_logits_bytes:
                raise RuntimeError(
                    'DSA index scoring exceeds the configured logits memory budget; '
                    'a compatible DeepGEMM installation is required.')
            k_cache, k_s_cache = _get_dsa_indexer_k_cache_views(
                indexer_k_cache, q.size(-1))
            scores = fp8_index(q,
                               q_s,
                               k_cache,
                               k_s_cache[..., 0],
                               meta.cu_seqlen_q,
                               meta.k_seqlens,
                               meta.block_offset,
                               max_q_seqlen=meta.max_q_seqlen,
                               max_k_seqlen=meta.max_kv_seqlen,
                               causal=True)
        return self._select_topk(scores, meta)

    def forward(self, q: Tensor, k: Tensor, weights: Tensor,
                indexer_k_cache: Tensor, meta: NSAIndexMeta) -> Tensor | None:
        assert q.dim() == 3
        assert k.dim() == 2
        k_cache, k_s_cache = _get_dsa_indexer_k_cache_views(
            indexer_k_cache, k.size(-1))
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
        return self._maybe_score_and_select(q, q_s, indexer_k_cache, meta)

    def forward_fused(self, q: Tensor, k: Tensor, weights: Tensor, norm_weight: Tensor, norm_bias: Tensor, cos: Tensor,
                      sin: Tensor, indexer_k_cache: Tensor, norm_eps: float, head_gate_scale: float,
                      rope_interleaved: bool, meta: NSAIndexMeta) -> Tensor | None:
        """Prepare FP8 Q and write K cache without allocating rotated BF16
        Q/K."""
        k_cache, k_s_cache = _get_dsa_indexer_k_cache_views(
            indexer_k_cache, k.size(-1))
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
        return self._maybe_score_and_select(q, q_s, indexer_k_cache, meta)


class TritonNSAIndexFP8Builder(BaseNSAIndexFP8Builder):

    @staticmethod
    def build(topk: int, softmax_scale: float, block_size: int = 128,
              fill: int = -1,
              allow_short_prefill_scoring_skip: bool = False) -> BaseNSAIndexFP8:
        return TritonNSAIndexFP8(
            topk,
            softmax_scale=softmax_scale,
            block_size=block_size,
            fill=fill,
            allow_short_prefill_scoring_skip=allow_short_prefill_scoring_skip,
        )
