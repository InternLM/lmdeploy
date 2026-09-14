# Copyright (c) OpenMMLab. All rights reserved.
"""TileLang sparse-MLA backend with reusable logical-index mapping."""

from __future__ import annotations

from typing import Any

import torch

from lmdeploy.pytorch.kernels.cuda.sparse_mla_tilelang import (
    sparse_mla_bf16_fwd,
)

from .sparse_mla import FlashMLAIndexMapper


class TilelangSparseMLADecode:
    """Run sparse MLA from chronological or model-provided logical indices."""

    def __init__(self, index_topk: int, index_kpool: int = 1):
        if index_topk <= 0:
            raise ValueError(f'index_topk must be positive, got {index_topk}')
        if index_kpool <= 0:
            raise ValueError(f'index_kpool must be positive, got {index_kpool}')
        self.index_topk = index_topk
        self.index_kpool = index_kpool
        tail_width = index_kpool - 1
        self.kernel_topk = ((index_topk + tail_width + 63) // 64) * 64
        self.index_mapper = FlashMLAIndexMapper.build()

    def _pad_logical_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Pad model indices with -1 to TileLang's 64-entry tile width."""
        if indices.ndim != 2:
            raise ValueError(
                'Sparse MLA logical indices must have shape [tokens, topk], '
                f'got {tuple(indices.shape)}.')
        if indices.size(1) > self.kernel_topk:
            raise ValueError(
                f'Logical index width {indices.size(1)} exceeds the '
                f'kernel width {self.kernel_topk}.')
        if indices.dtype != torch.int32:
            indices = indices.to(torch.int32)
        return torch.nn.functional.pad(
            indices, (0, self.kernel_topk - indices.size(1)), value=-1)

    def _build_physical_indices(self, query: torch.Tensor,
                                k_cache: torch.Tensor,
                                attn_metadata: Any) -> torch.Tensor:
        kv_seqlens = attn_metadata.kv_seqlens
        block_offsets = attn_metadata.block_offsets
        batch_size = kv_seqlens.numel()
        if query.size(0) != batch_size:
            raise NotImplementedError(
                'TileLang sparse MLA currently requires one decode token per '
                f'request, got {query.size(0)} tokens for {batch_size} requests.')
        if int(attn_metadata.max_kv_seqlen) > self.index_topk:
            raise NotImplementedError(
                'Contexts longer than index_topk require the model KPool indexer.')

        logical = torch.arange(self.kernel_topk,
                               dtype=torch.int32,
                               device=query.device)
        logical = logical.unsqueeze(0).expand(batch_size, -1).clone()
        logical.masked_fill_(logical >= kv_seqlens[:, None], -1)
        return self.index_mapper.map_paged_decode(
            logical,
            block_offsets,
            max_q_seqlen=1,
            block_size=k_cache.size(1),
        )

    def _map_decode_indices(self, logical_indices: torch.Tensor,
                            k_cache: torch.Tensor,
                            attn_metadata: Any) -> torch.Tensor:
        logical_indices = self._pad_logical_indices(logical_indices)
        batch_size = attn_metadata.kv_seqlens.numel()
        if logical_indices.size(0) % batch_size:
            raise ValueError(
                'Decode logical-index rows must be divisible by batch size, '
                f'got rows={logical_indices.size(0)}, batch={batch_size}.')
        max_q_seqlen = logical_indices.size(0) // batch_size
        return self.index_mapper.map_paged_decode(
            logical_indices,
            attn_metadata.block_offsets,
            max_q_seqlen=max_q_seqlen,
            block_size=k_cache.size(1),
        ).flatten(0, 1)[:, None]

    def _map_prefill_indices(self, logical_indices: torch.Tensor,
                             attn_metadata: Any) -> torch.Tensor:
        logical_indices = self._pad_logical_indices(logical_indices)
        return self.index_mapper.map_flat_prefill(
            logical_indices,
            attn_metadata.q_seqlens,
            attn_metadata.cu_seqlens_k,
        )

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                k_cache: torch.Tensor,
                v_cache: torch.Tensor,
                attn_metadata: Any,
                scale: float,
                cache_writer: Any,
                k_scales_zeros: torch.Tensor | None = None,
                v_scales_zeros: torch.Tensor | None = None,
                logical_indices: torch.Tensor | None = None) -> torch.Tensor:
        """Append latent KV, then run BF16 sparse MLA decode."""
        if k_cache.dtype != torch.bfloat16:
            raise TypeError('TileLang sparse MLA requires a BF16 KV cache.')
        cache_writer._lazy_init(query.device)
        cache_impl = cache_writer.impl
        max_q_seqlen = cache_impl._get_max_q_seqlen(query, attn_metadata)
        cache_impl._fill_kv_cache_impl(
            key,
            value,
            k_cache,
            v_cache,
            attn_metadata,
            max_q_seqlen,
            k_scales_zeros=k_scales_zeros,
            v_scales_zeros=v_scales_zeros,
        )
        if logical_indices is None:
            indices = self._build_physical_indices(
                query, k_cache, attn_metadata).flatten(0, 1)[:, None]
        else:
            indices = self._map_decode_indices(
                logical_indices, k_cache, attn_metadata)
        flat_cache = k_cache.flatten(0, 1)
        return sparse_mla_bf16_fwd(query, flat_cache, indices, scale)

    def forward_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        k_cache: torch.Tensor,
        attn_metadata: Any,
        scale: float,
        cache_writer: Any,
        logical_indices: torch.Tensor,
        k_scales_zeros: torch.Tensor | None = None,
        v_scales_zeros: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Append/flatten latent KV, then run sparse MLA prefill."""
        if k_cache.dtype != torch.bfloat16:
            raise TypeError('TileLang sparse MLA requires a BF16 KV cache.')
        flat_cache = cache_writer.fill_and_flatten_latent_kv_cache(
            key,
            k_cache,
            attn_metadata,
            out_dtype=query.dtype,
            k_scales_zeros=k_scales_zeros,
            v_scales_zeros=v_scales_zeros,
        )
        indices = self._map_prefill_indices(logical_indices, attn_metadata)
        return sparse_mla_bf16_fwd(query, flat_cache, indices, scale)
