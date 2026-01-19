# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass
from typing import Literal

import torch

from lmdeploy.pytorch.backends.attention import AttentionImpl, AttentionMetadata
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


@dataclass
class TritonAttentionMetadata(AttentionMetadata):
    """Triton attention metadata.

    This dataclass contains all metadata needed for attention computation
    across different stages (prefill/decoding) and implementations.

    Attributes:
        is_decoding: True for decoding stage, False for prefill.
        block_offsets: Block indices for paged KV cache [batch_size, max_blocks].
        q_start_loc: Start location of each query sequence [batch_size].
        q_seqlens: Length of each query sequence [batch_size].
        kv_start_loc: Start location of each KV sequence [batch_size].
        kv_seqlens: Length of each KV sequence [batch_size].
        quant_policy: Quantization policy (0=none, 4=int4, 8=int8/fp8).
        kv_flatten_size: Total size of flattened KV cache.
        tile_scheduler_metadata: Scheduler metadata for Flash MLA.
        num_splits: Number of splits for Flash MLA.
        cu_seqlens_q: Cumulative query sequence lengths [batch_size + 1].
        cu_seqlens_k: Cumulative KV sequence lengths [batch_size + 1].
        scheduler_metadata: Scheduler metadata for FA3.
        max_kv_seqlen: Maximum KV sequence length in the batch.
        max_q_seqlen: Maximum query sequence length in the batch.
    """
    is_decoding: bool
    block_offsets: torch.Tensor
    q_start_loc: torch.Tensor = None
    q_seqlens: torch.Tensor = None
    kv_start_loc: torch.Tensor = None
    kv_seqlens: torch.Tensor = None
    quant_policy: Literal[0, 4, 8] = 0
    kv_flatten_size: int = None
    # flash mla
    tile_scheduler_metadata: torch.Tensor = None
    num_splits: torch.Tensor = None
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None
    # flash attn
    scheduler_metadata: torch.Tensor = None
    max_kv_seqlen: int = None
    max_q_seqlen: int = None


def _cdiv(a, b):
    """Perform ceiling division (division rounded up).

    Args:
        a: Dividend.
        b: Divisor.

    Returns:
        Ceiling of a / b.
    """
    return (a + b - 1) // b


class TritonAttentionImpl(AttentionImpl[TritonAttentionMetadata]):
    """Triton attention implementation."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float = None,
        num_kv_heads: int = None,
        v_head_size: int = None,
        alibi: bool = False,
        sliding_window: int = None,
        logit_softcapping: float = 0.0,
        causal: bool = True,
        block_sparse_size: int = 1,
        **kwargs,
    ):
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            v_head_size=v_head_size,
            alibi=alibi,
            sliding_window=sliding_window,
            logit_softcapping=logit_softcapping,
            causal=causal,
            **kwargs,
        )
        self.logit_softcapping = -1 if self.logit_softcapping <= 0.0 else self.logit_softcapping
        assert not (alibi and not causal)

        from lmdeploy.pytorch.kernels.cuda import (fill_kv_cache, flash_attn_varlen_func, flash_attn_with_kvcache,
                                                   flatten_kv_cache)

        self.fill_kv_cache = fill_kv_cache
        self.paged_attention_fwd = flash_attn_with_kvcache
        self.flatten_kv_cache = flatten_kv_cache
        self.flash_attention_fwd = flash_attn_varlen_func

        self.block_sparse_size = block_sparse_size

    def _get_max_q_seqlen(
        self,
        query: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
    ) -> int:
        """Get max q seqlen."""
        if attn_metadata.is_decoding:
            max_q_seqlen = self.block_sparse_size
        else:
            if attn_metadata.max_q_seqlen is not None:
                max_q_seqlen = attn_metadata.max_q_seqlen
            else:
                max_q_seqlen = query.numel() // (query.size(-1) * query.size(-2))
        return max_q_seqlen

    def _get_fill_meta(
        self,
        key: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        max_q_seqlen: int,
    ):
        """Get fill meta."""
        fill_seqlens = attn_metadata.q_seqlens
        fill_max_q_seqlen = max_q_seqlen
        fill_q_start_loc = attn_metadata.q_start_loc
        return fill_seqlens, fill_max_q_seqlen, fill_q_start_loc

    def _fill_kv_cache_impl(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        max_q_seqlen: int,
        k_scales_zeros: torch.Tensor = None,
        v_scales_zeros: torch.Tensor = None,
    ):
        """Fill kv cache."""
        kv_seqlens = attn_metadata.kv_seqlens
        block_offsets = attn_metadata.block_offsets
        quant_policy = attn_metadata.quant_policy

        # fill seqlen args
        fill_seqlens, fill_max_q_seqlen, fill_q_start_loc = self._get_fill_meta(
            key,
            attn_metadata,
            max_q_seqlen,
        )

        # fill kv cache
        self.fill_kv_cache(
            key,
            value,
            k_cache,
            v_cache,
            fill_q_start_loc,
            fill_seqlens,
            kv_seq_length=kv_seqlens,
            max_q_seq_length=fill_max_q_seqlen,
            block_offsets=block_offsets,
            k_scales_zeros=k_scales_zeros,
            v_scales_zeros=v_scales_zeros,
            quant_policy=quant_policy,
        )

    def _forward_decoding(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        max_q_seqlen: int,
        k_scales_zeros: torch.Tensor = None,
        v_scales_zeros: torch.Tensor = None,
        learnable_sink: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass for decoding stage.

        Args:
            query: Query tensor.
            k_cache: Key cache tensor.
            v_cache: Value cache tensor.
            attn_metadata: Attention metadata.
            max_q_seqlen: Maximum query sequence length.
            k_scales_zeros: Key quantization scales/zeros.
            v_scales_zeros: Value quantization scales/zeros.
            learnable_sink: Learnable sink tokens.

        Returns:
            Attention output tensor.
        """
        block_offsets = attn_metadata.block_offsets
        quant_policy = attn_metadata.quant_policy

        attn_output = self.paged_attention_fwd(
            query,
            k_cache,
            v_cache,
            cache_seqlens=attn_metadata.kv_seqlens,
            page_table=block_offsets,
            cu_seqlens_q=attn_metadata.cu_seqlens_q,
            max_seqlen_q=max_q_seqlen,
            softmax_scale=self.scale,
            softcap=self.logit_softcapping,
            window_size=self.sliding_window,
            # custom args
            sinks=learnable_sink,
            alibi_slopes=self.alibi_slopes,
            quant_policy=quant_policy,
            k_scales_zeros=k_scales_zeros,
            v_scales_zeros=v_scales_zeros,
        )
        return attn_output

    def _forward_prefill(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        max_q_seqlen: int,
        k_scales_zeros: torch.Tensor = None,
        v_scales_zeros: torch.Tensor = None,
        learnable_sink: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass for prefill stage.

        Args:
            query: Query tensor.
            k_cache: Key cache tensor.
            v_cache: Value cache tensor.
            attn_metadata: Attention metadata.
            max_q_seqlen: Maximum query sequence length.
            k_scales_zeros: Key quantization scales/zeros.
            v_scales_zeros: Value quantization scales/zeros.
            learnable_sink: Learnable sink tokens.

        Returns:
            Attention output tensor.
        """
        block_offsets = attn_metadata.block_offsets
        kv_start_loc = attn_metadata.kv_start_loc
        kv_seqlens = attn_metadata.kv_seqlens
        kv_flatten_size = attn_metadata.kv_flatten_size
        quant_policy = attn_metadata.quant_policy

        # Prepare flattened KV cache
        BLOCK_BS = k_cache.size(1)
        # pad one more block to avoid invalid kv visit
        out_size = (_cdiv(kv_flatten_size, BLOCK_BS) * BLOCK_BS + BLOCK_BS)
        kv_layout = 'hsd'  # custom triton kernel requires 'hsd' while fa3 requires 'shd'

        flatten_k, flatten_v = self.flatten_kv_cache(
            k_cache,
            v_cache,
            kv_seqlens,
            block_offsets,
            start_loc=kv_start_loc,
            out_size=out_size,
            out_dtype=query.dtype,
            k_scales_zeros=k_scales_zeros,
            v_scales_zeros=v_scales_zeros,
            quant_policy=quant_policy,
            flatten_kv_layout=kv_layout,
        )

        attn_output = self.flash_attention_fwd(
            query,
            flatten_k,
            flatten_v,
            cu_seqlens_q=attn_metadata.cu_seqlens_q,
            cu_seqlens_k=attn_metadata.cu_seqlens_k,
            max_seqlen_q=max_q_seqlen,
            max_seqlen_k=attn_metadata.max_kv_seqlen,
            window_size=self.sliding_window,
            softmax_scale=self.scale,
            softcap=self.logit_softcapping,
            causal=self.causal,
            # custom args
            sinks=learnable_sink,
            alibi_slopes=self.alibi_slopes,
            block_sparse_size=self.block_sparse_size,
            kv_layout=kv_layout,
        )
        return attn_output

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        k_scales_zeros: torch.Tensor = None,
        v_scales_zeros: torch.Tensor = None,
        learnable_sink: torch.Tensor = None,
        inplace: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for attention computation.

        This method handles both prefill and decoding stages by:
        1. Computing max query sequence length
        2. Filling KV cache if new key/value are provided
        3. Dispatching to appropriate stage-specific method

        Args:
            query: Query tensor.
            key: Key tensor (None for decoding-only).
            value: Value tensor (None for decoding-only).
            k_cache: Key cache tensor.
            v_cache: Value cache tensor.
            attn_metadata: Attention metadata containing stage info and indices.
            k_scales_zeros: Key quantization scales/zeros.
            v_scales_zeros: Value quantization scales/zeros.
            learnable_sink: Learnable sink tokens.
            inplace: Whether to modify query inplace (unused, kept for compatibility).

        Returns:
            Attention output tensor.
        """
        # Shared preparation
        max_q_seqlen = self._get_max_q_seqlen(query, attn_metadata)

        # Fill KV cache with new key/value if provided
        if key is not None and value is not None:
            self._fill_kv_cache_impl(
                key,
                value,
                k_cache=k_cache,
                v_cache=v_cache,
                attn_metadata=attn_metadata,
                max_q_seqlen=max_q_seqlen,
                k_scales_zeros=k_scales_zeros,
                v_scales_zeros=v_scales_zeros,
            )

        # Validate alibi configuration
        if self.alibi:
            assert self.alibi_slopes is not None, 'alibi_slopes is not set.'

        # Dispatch to stage-specific forward method
        if attn_metadata.is_decoding:
            return self._forward_decoding(
                query,
                k_cache,
                v_cache,
                attn_metadata,
                max_q_seqlen,
                k_scales_zeros=k_scales_zeros,
                v_scales_zeros=v_scales_zeros,
                learnable_sink=learnable_sink,
            )
        else:
            return self._forward_prefill(
                query,
                k_cache,
                v_cache,
                attn_metadata,
                max_q_seqlen,
                k_scales_zeros=k_scales_zeros,
                v_scales_zeros=v_scales_zeros,
                learnable_sink=learnable_sink,
            )
