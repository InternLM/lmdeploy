# Copyright (c) OpenMMLab. All rights reserved.
"""LMDeploy CUDA adapters for pooled DSA selection."""

from __future__ import annotations

import functools

import torch
from torch import Tensor

from lmdeploy.pytorch.kernels.cuda.sparse_index_topk import (
    is_sparse_index_topk_supported,
    sparse_index_topk,
)
from lmdeploy.pytorch.nn.kpool import kpool_compress, kpool_quantize_fp8


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


def kpool_compress_quantize_cuda(
    slot_k: Tensor,
    slot_score: Tensor,
    ape: Tensor,
    *,
    mode: str,
    round_scale: bool,
) -> tuple[Tensor, Tensor]:
    """Compress and quantize closed pools with LMDeploy's reusable semantics."""
    pooled = kpool_compress(slot_k, slot_score, ape, mode=mode)
    return kpool_quantize_fp8(
        pooled,
        block_size=pooled.size(-1),
        round_scale=round_scale,
    )


def kpool_select_groups_cuda(
    logits: Tensor,
    group_lengths: Tensor,
    *,
    group_topk: int,
    row_starts: Tensor | None = None,
    max_group_length: int | None = None,
) -> Tensor:
    """Select pooled groups with LMDeploy's shared sparse-index Top-K kernel."""
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
        score_window.contiguous(),
        q_seqlens,
        lengths.clamp(max=max_group_length),
        group_topk,
        fill=-1,
        descending=True,
        sorted=False,
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
    """Score pooled decode history with DeepGEMM's paged MQA primitive."""
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
