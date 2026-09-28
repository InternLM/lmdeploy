# Copyright (c) OpenMMLab. All rights reserved.
"""LMDeploy CUDA adapters for pooled DSA selection."""

from __future__ import annotations

import functools
import inspect

import torch
from torch import Tensor

from lmdeploy.pytorch.kernels.cuda.fill_kv_cache import fill_indexed_key_cache
from lmdeploy.pytorch.kernels.cuda.flatten_kv_cache import flatten_kv_cache
from lmdeploy.pytorch.kernels.cuda.kpool import (
    compress_kpool,
    kpool_prefill_metadata,
    partition_kpool,
    rotate_kpool_query,
    update_kpool,
)
from lmdeploy.pytorch.kernels.cuda.sparse_index_topk import (
    is_sparse_index_topk_supported,
    sparse_index_topk,
)
from lmdeploy.pytorch.nn.kpool import (
    KPOOL_PAGE_SIZE,
    kpool_compress,
    kpool_expand_selected_groups,
    kpool_packed_cache_views,
    kpool_pooled_block_offsets,
    kpool_quantize_fp8,
)

# Fuse the existing integer index expansion instead of materializing its
# [tokens, topk] masks and int64 temporaries separately during prefill or decode.
kpool_expand_groups_cuda = torch.compile(kpool_expand_selected_groups, dynamic=True, fullgraph=True)
kpool_rotate_query_cuda = rotate_kpool_query


def kpool_dense_indices_cuda(q_seqlens, kv_seqlens, rows, pool_size, topk):
    """Reuse causal expansion when every prefill group fits in the budget."""
    _, _, seq, lengths, _, _ = kpool_prefill_metadata(q_seqlens, kv_seqlens, rows, pool_size)
    groups = torch.arange(topk // pool_size, device=q_seqlens.device, dtype=torch.int32)
    return kpool_expand_groups_cuda(groups.expand(rows, -1), lengths, pool_size, topk, seq_lens=seq)


def kpool_decode_update_cuda(keys, scores, tail_keys, tail_scores, state_ids,
                             history_lengths, packed_cache, block_offsets,
                             ape, pool_size, round_scale):
    """Batch verification updates and reuse indexed cache writes."""
    closed_keys, closed_scores, groups, valid = update_kpool(
        keys, scores, tail_keys, tail_scores, state_ids, history_lengths, pool_size)
    compress = (compress_kpool if torch.cuda.get_device_capability(keys.device)[0] >= 9
                else kpool_compress_quantize_cuda)
    values, scales = compress(
        closed_keys, closed_scores, ape, mode='decode', round_scale=round_scale)
    cache_keys, cache_scales = kpool_packed_cache_views(packed_cache, keys.size(-1))
    fill_indexed_key_cache(values, scales, groups, valid, block_offsets,
                           cache_keys, cache_scales, page_step=pool_size)


def kpool_prefill_update_cuda(keys, scores, tail_keys, tail_scores, state_ids,
                              q_seqlens, kv_seqlens, packed_cache, block_offsets,
                              ape, pool_size, round_scale):
    """Batch ragged pool assembly without a host read or cache-arena copy."""
    closed_keys, closed_scores, group_ids, requests, valid = partition_kpool(
        keys, scores, tail_keys, tail_scores, state_ids, q_seqlens, kv_seqlens, pool_size)
    if closed_keys.size(0):
        values, scales = kpool_compress_quantize_cuda(
            closed_keys, closed_scores, ape, mode='extend', round_scale=round_scale)
        cache_keys, cache_scales = kpool_packed_cache_views(packed_cache, keys.size(-1))
        fill_indexed_key_cache(values, scales, group_ids, valid, block_offsets,
                               cache_keys, cache_scales, page_step=cache_keys.size(1) * pool_size // KPOOL_PAGE_SIZE,
                               request_ids=requests)


def kpool_write_decode_cuda(packed_cache, block_offsets, group_ids, values, scales, pool_size, valid):
    """Masked scatter for one decode step, with no read-modify-write.

    The scheduler gives live requests private writable owners. Thus at most
    one valid row writes each destination in this launch. Keep MTP steps in
    separate launches: different steps may close the same logical pool.
    """
    cache_keys, cache_scales = kpool_packed_cache_views(packed_cache, values.size(-1))
    fill_indexed_key_cache(values, scales, group_ids, valid, block_offsets,
                           cache_keys, cache_scales,
                           page_step=cache_keys.size(1) * pool_size // KPOOL_PAGE_SIZE)


@functools.lru_cache
def _get_deep_gemm():
    try:
        import deep_gemm
    except ImportError as error:
        raise RuntimeError(
            'GLM-5.3 KPool scoring requires DeepGEMM.') from error
    required = (
        'fp8_mqa_logits',
        'fp8_paged_mqa_logits',
        'get_paged_mqa_logits_metadata',
        'get_num_sms',
    )
    missing = [name for name in required if not hasattr(deep_gemm, name)]
    if missing:
        raise RuntimeError(
            f'GLM-5.3 KPool requires DeepGEMM APIs: {missing}.')
    return deep_gemm


@functools.lru_cache
def _mqa_has_local_columns() -> bool:
    """New DeepGEMM can return compact, request-local logits columns.

    Older pybind wheels expose only their signature docstring. Conservatively
    use the shared absolute-column API when no compact signature is advertised.
    """
    func = _get_deep_gemm().fp8_mqa_logits
    try:
        return 'max_seqlen_k' in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return 'max_seqlen_k' in (func.__doc__ or '')


def kpool_compress_quantize_cuda(
    slot_k: Tensor,
    slot_score: Tensor,
    ape: Tensor,
    *,
    mode: str,
    round_scale: bool,
) -> tuple[Tensor, Tensor]:
    """Reuse fused compression for both prefill and decode closed pools."""
    if slot_k.is_cuda and torch.cuda.get_device_capability(slot_k.device)[0] >= 9:
        return compress_kpool(slot_k, slot_score, ape, mode=mode, round_scale=round_scale)
    pooled = kpool_compress(slot_k, slot_score, ape, mode=mode)
    return kpool_quantize_fp8(
        pooled,
        block_size=pooled.size(-1),
        round_scale=round_scale,
    )


def kpool_select_prefill_cuda(query_fp8, query_weight, packed_cache,
                              q_seqlens, kv_seqlens, block_offsets, kv_flatten_size,
                              pool_size, topk):
    """Score and select all ragged prefill requests without host length
    reads."""
    _validate_query(query_fp8, query_weight)
    rows = query_fp8.size(0)
    if not rows:
        return torch.empty((0, topk + pool_size - 1), device=query_fp8.device, dtype=torch.int32)
    counts, starts, seq, lengths, query_starts, query_ends = kpool_prefill_metadata(
        q_seqlens, kv_seqlens, rows, pool_size)
    keys, scales = kpool_packed_cache_views(packed_cache, query_fp8.size(-1))
    blocks = kpool_pooled_block_offsets(block_offsets, pool_size, keys.size(1)).contiguous()
    # Sum(floor(kv / pool)) can be up to batch - 1 below floor(sum(kv) / pool).
    # Give the shared flatten kernel enough pages to zero this padded tail.
    tail_pages = (counts.numel() + keys.size(1) - 1) // keys.size(1)
    if tail_pages > blocks.size(1):
        blocks = torch.nn.functional.pad(blocks, (0, tail_pages - blocks.size(1)))
    # Reuse the same-dtype flatten operator by copying both fields as bytes.
    flat_keys, flat_scales = flatten_kv_cache(
        keys.view(torch.uint8).unsqueeze(2), scales.view(torch.uint8).unsqueeze(2),
        counts, blocks, start_loc=starts, out_size=max(1, kv_flatten_size // pool_size))
    max_groups = block_offsets.size(1) * KPOOL_PAGE_SIZE // pool_size
    local_columns = _mqa_has_local_columns()
    logits = _get_deep_gemm().fp8_mqa_logits(
        query_fp8.contiguous(),
        (flat_keys[0].view(torch.float8_e4m3fn), flat_scales[0].view(torch.float32).flatten()),
        query_weight.contiguous(), query_starts, query_ends,
        clean_logits=False, **({'max_seqlen_k': max_groups} if local_columns else {}))
    # Legacy logits address the concatenated cache; normalize columns before
    # selection. Both APIs mask the unwritten suffix by each query's length.
    selected = kpool_select_groups_cuda(
        logits, lengths, group_topk=topk // pool_size,
        row_starts=None if local_columns else query_starts,
        max_group_length=min(max_groups, logits.size(1)))
    return kpool_expand_groups_cuda(selected, lengths, pool_size, topk, seq_lens=seq)


def kpool_select_groups_cuda(
    logits: Tensor,
    group_lengths: Tensor,
    *,
    group_topk: int,
    row_starts: Tensor | None = None,
    max_group_length: int | None = None,
) -> Tensor:
    """Select pooled groups with LMDeploy's shared sparse-index Top-K
    kernel."""
    if not is_sparse_index_topk_supported(group_topk):
        raise ValueError(
            'The GLM-5.3 KPool selector only supports group_topk=512 or 2048, '
            f'got {group_topk}.')
    if logits.ndim != 2 or logits.dtype != torch.float32:
        raise ValueError(
            'KPool logits must be a two-dimensional float32 tensor, got '
            f'shape={tuple(logits.shape)}, dtype={logits.dtype}.')
    if logits.stride(-1) != 1:
        raise ValueError('KPool logits must be contiguous on the group axis.')
    if group_lengths.shape != (logits.size(0), ):
        raise ValueError('group_lengths must contain one value per logits row.')
    if row_starts is not None and row_starts.shape != (logits.size(0), ):
        raise ValueError('row_starts must contain one value per logits row.')
    if max_group_length is None:
        max_group_length = logits.size(1)
    if max_group_length < 0 or max_group_length > logits.size(1):
        raise ValueError(
            'max_group_length must be inside the logits width, got '
            f'{max_group_length} for width={logits.size(1)}.')
    if max_group_length == 0:
        return torch.full(
            (logits.size(0), group_topk),
            -1,
            dtype=torch.int32,
            device=logits.device,
        )
    lengths = group_lengths.to(
        device=logits.device, dtype=torch.int32).contiguous()
    if row_starts is not None:
        row_starts = row_starts.to(
            device=logits.device, dtype=torch.int32).contiguous()
    score_window = logits[:, :max_group_length]
    if row_starts is not None:
        columns = torch.arange(
            max_group_length, dtype=torch.int64, device=logits.device)
        gather_ids = row_starts.to(torch.int64)[:, None] + columns[None]
        gather_ids = gather_ids.clamp(max=logits.size(1) - 1)
        score_window = logits.gather(1, gather_ids)
    q_seqlens = torch.ones(
        logits.size(0), dtype=torch.int32, device=logits.device)
    return sparse_index_topk(
        score_window,
        q_seqlens,
        lengths.clamp(max=max_group_length),
        group_topk,
        fill=-1,
        descending=True,
        sorted=False,
        # The BF16 sparse-MLA consumer is sensitive to index tile ordering.
        # Keep the same selected set/order across AR and MTP verification.
        deterministic=True,
    )


def _validate_query(query_fp8: Tensor, query_weight: Tensor) -> None:
    if query_fp8.ndim != 3:
        raise ValueError('query_fp8 must have shape [rows, heads, head_dim].')
    if query_weight.shape != query_fp8.shape[:2]:
        raise ValueError(
            'query_weight must have shape [rows, heads], got '
            f'{tuple(query_weight.shape)} for query {tuple(query_fp8.shape)}.')
    if query_fp8.dtype != torch.float8_e4m3fn:
        raise TypeError(
            f'query_fp8 must use float8_e4m3fn, got {query_fp8.dtype}.')
    if query_weight.dtype != torch.float32:
        raise TypeError(
            f'query_weight must use float32, got {query_weight.dtype}.')


def kpool_score_contiguous_cuda(
    query_fp8: Tensor,
    query_weight: Tensor,
    pooled_key_fp8: Tensor,
    pooled_key_scale: Tensor,
    group_lengths: Tensor,
) -> Tensor:
    """Score ragged pooled history with DeepGEMM's contiguous MQA primitive."""
    _validate_query(query_fp8, query_weight)
    rows = query_fp8.size(0)
    if pooled_key_fp8.ndim != 2 or pooled_key_fp8.size(1) != query_fp8.size(2):
        raise ValueError(
            'pooled_key_fp8 must have shape [groups, query_head_dim].')
    if pooled_key_scale.shape == (pooled_key_fp8.size(0), 1):
        pooled_key_scale = pooled_key_scale.squeeze(1)
    if pooled_key_scale.shape != (pooled_key_fp8.size(0), ):
        raise ValueError('pooled_key_scale must contain one scale per group.')
    if group_lengths.shape != (rows, ):
        raise ValueError('group_lengths must contain one value per query row.')
    if pooled_key_fp8.size(0) == 0:
        return torch.empty(
            (rows, 0), dtype=torch.float32, device=query_fp8.device)

    starts = torch.zeros(rows, dtype=torch.int32, device=query_fp8.device)
    ends = group_lengths.to(device=query_fp8.device,
                            dtype=torch.int32).contiguous()
    return _get_deep_gemm().fp8_mqa_logits(
        query_fp8.contiguous(),
        (pooled_key_fp8.contiguous(), pooled_key_scale.contiguous()),
        query_weight.contiguous(),
        starts,
        ends,
        clean_logits=True,
    )


def kpool_score_paged_cuda(
    query_fp8: Tensor,
    query_weight: Tensor,
    packed_cache: Tensor,
    group_lengths: Tensor,
    pooled_block_offsets: Tensor,
    page_size: int = 64,
) -> Tensor:
    """Score compact16 pages with Triton, or legacy page64 with DeepGEMM."""
    _validate_query(query_fp8, query_weight)
    rows = query_fp8.size(0)
    if packed_cache.dtype != torch.uint8 or packed_cache.ndim != 4:
        raise ValueError(
            'packed_cache must be a uint8 [blocks, entries, 1, width] tensor.')
    if packed_cache.size(1) != page_size or packed_cache.size(2) != 1:
        raise ValueError(
            f'packed_cache must contain [{page_size}, 1] entries per page.')
    if group_lengths.shape != (rows, ):
        raise ValueError('group_lengths must contain one value per query row.')
    if pooled_block_offsets.ndim != 2:
        raise ValueError('pooled_block_offsets must have shape [rows, pages].')
    if pooled_block_offsets.size(0) == 1 and rows != 1:
        pooled_block_offsets = pooled_block_offsets.expand(rows, -1)
    if pooled_block_offsets.size(0) != rows:
        raise ValueError(
            'pooled_block_offsets must have one row per query row.')
    if pooled_block_offsets.size(1) == 0:
        return torch.empty(
            (rows, 0), dtype=torch.float32, device=query_fp8.device)

    if page_size == 16:
        from lmdeploy.pytorch.kernels.cuda.kpool import score_paged
        return score_paged(query_fp8, query_weight, packed_cache,
                           group_lengths, pooled_block_offsets)
    if page_size != 64:
        raise ValueError(f'Unsupported KPool storage page size: {page_size}.')

    deep_gemm = _get_deep_gemm()
    context_lens = group_lengths.to(
        device=query_fp8.device, dtype=torch.int32).contiguous().view(-1, 1)
    block_table = pooled_block_offsets.to(
        device=query_fp8.device, dtype=torch.int32).contiguous()
    schedule = deep_gemm.get_paged_mqa_logits_metadata(
        context_lens.clamp(min=1), page_size, deep_gemm.get_num_sms())
    return deep_gemm.fp8_paged_mqa_logits(
        query_fp8.contiguous().unsqueeze(1),
        packed_cache,
        query_weight.contiguous(),
        context_lens,
        block_table,
        schedule,
        block_table.size(1) * page_size,
        clean_logits=False,
    )


class CudaKPoolAttention:
    """Sequence-dependent KPool/MLA execution between captured projections.

    The supplied attention operation owns model math and cache updates; this
    CUDA adapter owns raw-token slicing and its PCG output bridge. It never
    captures the dense/sparse decision or request-local state indexing.
    """

    def __init__(self, attention):
        from .step_metadata import register_piecewise_graph_impl
        self.forward = attention
        self._piecewise_forward = None
        register_piecewise_graph_impl(self)

    def supports_piecewise_cuda_graph(self) -> bool:
        return True

    def enable_piecewise_cuda_graph(self) -> None:
        if self._piecewise_forward is not None:
            return
        from .graph_runner.piecewise import (
            ViewTolerantPaddedAdapter,
            eager_boundary,
            get_piecewise_graph_execution,
        )
        original = self.forward

        @eager_boundary(adapter_factory=ViewTolerantPaddedAdapter, reuse_bridge_after_next_step=True)
        def run_eager(hidden_states, q_lora, query, key, value, **kwargs):
            count = get_piecewise_graph_execution().raw_tokens
            return original(hidden_states[:, :count], q_lora[:, :count],
                            query[:count], key[:count], value[:count], **kwargs)

        def forward(hidden_states, q_lora, query, key, value, **kwargs):
            if get_piecewise_graph_execution() is None:
                return original(hidden_states, q_lora, query, key, value, **kwargs)
            return run_eager(hidden_states, q_lora, query, key, value, **kwargs)

        self._piecewise_forward = forward
        self.forward = forward
