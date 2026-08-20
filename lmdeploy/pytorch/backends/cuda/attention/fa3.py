# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Hashable
from dataclasses import dataclass

import torch

from lmdeploy.messages import QuantPolicy
from lmdeploy.utils import get_logger

from ..step_metadata import CudaAttentionMetaBuilder
from .default import TritonAttentionImpl, TritonAttentionMetadata, _cdiv

logger = get_logger('lmdeploy')


@dataclass
class FA3AttentionMetadata:
    """Scheduler metadata owned by one FA3 configuration."""

    scheduler_metadata: torch.Tensor = None
    max_kv_seqlen: int = None


def _get_meta_flashattn(
        batch_size: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        num_heads_q: int,
        num_heads_kv: int,
        headdim: int,
        cache_seqlens: torch.Tensor,
        qkv_dtype=torch.bfloat16,
        headdim_v=None,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k_new: torch.Tensor | None = None,
        page_size: int | None = None,
        causal=True,
        window_size=(-1, -1),
        num_splits=0,
        has_softcap=False,
):
    """Build FlashAttention scheduler metadata."""
    from flash_attn_interface import get_scheduler_metadata

    return get_scheduler_metadata(
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads_q,
        num_heads_kv,
        headdim,
        cache_seqlens,
        qkv_dtype=qkv_dtype,
        headdim_v=headdim_v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_k_new,
        page_size=page_size,
        causal=causal,
        window_size=window_size,
        has_softcap=has_softcap,
        num_splits=num_splits,
    )


def _normalize_sliding_window(sliding_window) -> tuple[int, int]:
    if sliding_window is None:
        return (-1, -1)
    if isinstance(sliding_window, int):
        return (sliding_window, sliding_window)
    return sliding_window


def _build_fa3_metadata(batch_size: int,
                        kv_seqlens,
                        block_offsets,
                        step_context,
                        num_heads_q: int,
                        num_heads_kv: int,
                        head_size: int,
                        sliding_window,
                        causal: bool = True,
                        v_head_size: int | None = None,
                        block_size: int | None = None,
                        max_seqlen_q: int | None = None,
                        max_seqlen_k: int | None = None,
                        has_softcap: bool = False) -> FA3AttentionMetadata:
    """Build scheduler metadata from one selected FA3 implementation."""
    if block_size is None:
        block_size = step_context.model_config.block_size
    if max_seqlen_q is None:
        max_seqlen_q = step_context.input_ids.size(1) // batch_size
    if max_seqlen_k is None:
        # FA3 derives seqlen_k from the padded page table when
        # cu_seqlens_k is absent. The scheduler must use the same bound.
        assert block_offsets is not None
        max_seqlen_k = block_offsets.size(1) * block_size

    scheduler_metadata = _get_meta_flashattn(
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        headdim=head_size,
        headdim_v=v_head_size,
        cache_seqlens=kv_seqlens.to(torch.int32),
        qkv_dtype=step_context.model_config.dtype,
        page_size=block_size,
        causal=causal,
        window_size=_normalize_sliding_window(sliding_window),
        has_softcap=has_softcap,
    )
    return FA3AttentionMetadata(
        scheduler_metadata=scheduler_metadata,
        max_kv_seqlen=max_seqlen_k,
    )


def build_fa3_metadata(sequence_metadata, step_context, **kwargs) -> FA3AttentionMetadata:
    """Build scheduler metadata from one selected FA3 implementation."""
    return _build_fa3_metadata(
        sequence_metadata.q_seqlens.size(0),
        sequence_metadata.kv_seqlens,
        sequence_metadata.block_offsets,
        step_context,
        **kwargs,
    )


def update_fa3_metadata(attn_metadata, step_context, **kwargs) -> None:
    """Populate the legacy single-group FA3 metadata fields."""
    metadata = build_fa3_metadata(attn_metadata, step_context, **kwargs)
    attn_metadata.scheduler_metadata = metadata.scheduler_metadata
    attn_metadata.max_kv_seqlen = metadata.max_kv_seqlen


def build_fa3_graph_metadata(step_context,
                             batch_size: int,
                             kv_seqlens,
                             block_size: int,
                             max_seqlen_q: int,
                             max_seqlen_k: int) -> FA3AttentionMetadata:
    """Build legacy graph metadata from the model-level FA3 configuration."""
    num_heads, num_kv_heads = step_context.model_config.get_num_qkv_head_by_tp()
    model_config = step_context.model_config
    return _build_fa3_metadata(
        batch_size,
        kv_seqlens,
        None,
        step_context,
        num_heads_q=num_heads,
        num_heads_kv=num_kv_heads,
        head_size=model_config.head_dim,
        sliding_window=model_config.sliding_window,
        block_size=block_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )


@dataclass(frozen=True)
class FA3AttentionMetaBuilder(CudaAttentionMetaBuilder[torch.Tensor | None, FA3AttentionMetadata]):
    """Build metadata requested by one selected FA3 configuration."""

    num_heads: int
    num_kv_heads: int
    head_size: int
    v_head_size: int
    sliding_window: tuple[int, int]
    causal: bool
    has_softcap: bool

    @property
    def key(self) -> Hashable:
        return (type(self), self.num_heads, self.num_kv_heads, self.head_size, self.v_head_size, self.sliding_window,
                self.causal, self.has_softcap)

    @staticmethod
    def _needs_scheduler(step_context, sequence_metadata) -> bool:
        if not step_context.is_decoding:
            return False
        batch_size = sequence_metadata.q_seqlens.size(0)
        return step_context.input_ids.size(1) > batch_size

    def build(self, step_context, sequence_metadata) -> FA3AttentionMetadata:
        if not self._needs_scheduler(step_context, sequence_metadata):
            return FA3AttentionMetadata()
        return build_fa3_metadata(
            sequence_metadata,
            step_context,
            num_heads_q=self.num_heads,
            num_heads_kv=self.num_kv_heads,
            head_size=self.head_size,
            v_head_size=self.v_head_size,
            sliding_window=self.sliding_window,
            causal=self.causal,
            has_softcap=self.has_softcap,
        )

    def apply_legacy_metadata(self, attn_metadata, metadata: FA3AttentionMetadata) -> None:
        attn_metadata.scheduler_metadata = metadata.scheduler_metadata
        if metadata.max_kv_seqlen is not None:
            attn_metadata.max_kv_seqlen = metadata.max_kv_seqlen

    def make_cudagraph_buffer(self, graph_meta, input_buffers, step_context) -> torch.Tensor | None:
        if graph_meta.decode_query_len <= 1:
            return None
        metadata = _build_fa3_metadata(
            input_buffers['q_seqlens'].size(0),
            input_buffers['kv_seqlens'],
            input_buffers['block_offsets'],
            step_context,
            num_heads_q=self.num_heads,
            num_heads_kv=self.num_kv_heads,
            head_size=self.head_size,
            v_head_size=self.v_head_size,
            sliding_window=self.sliding_window,
            causal=self.causal,
            has_softcap=self.has_softcap,
            max_seqlen_q=graph_meta.decode_query_len,
            max_seqlen_k=graph_meta.num_blocks * graph_meta.block_size,
        )
        return metadata.scheduler_metadata

    def fill_cudagraph_buffer(self, graph_meta, input_buffers, step_context,
                              buffer: torch.Tensor | None) -> FA3AttentionMetadata:
        if buffer is None:
            return FA3AttentionMetadata()
        metadata = _build_fa3_metadata(
            input_buffers['q_seqlens'].size(0),
            input_buffers['kv_seqlens'],
            input_buffers['block_offsets'],
            step_context,
            num_heads_q=self.num_heads,
            num_heads_kv=self.num_kv_heads,
            head_size=self.head_size,
            v_head_size=self.v_head_size,
            sliding_window=self.sliding_window,
            causal=self.causal,
            has_softcap=self.has_softcap,
            max_seqlen_q=graph_meta.decode_query_len,
            max_seqlen_k=graph_meta.num_blocks * graph_meta.block_size,
        )
        num_meta = metadata.scheduler_metadata.size(0)
        buffer[:num_meta].copy_(metadata.scheduler_metadata)
        buffer[num_meta:].zero_()
        return FA3AttentionMetadata(
            scheduler_metadata=buffer[:num_meta],
            max_kv_seqlen=metadata.max_kv_seqlen,
        )


class FA3Impl(TritonAttentionImpl):
    """Flash Attention 3 implementation.

    This implementation leverages Flash Attention 3's optimized kernels for both
    prefill and decoding stages. FA3 provides significant performance improvements
    on Ampere and above (SM80+) with CUDA >= 12.3.

    Key features:
    - Optimized prefill using flash_attn_varlen_func
    - Speculative decoding support with multi-token queries
    - Standard single-token decoding with paged attention
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float = None,
        num_kv_heads: int = None,
        v_head_size: int = None,
        alibi: bool = False,
        sliding_window: tuple = None,
        logit_softcapping: float = 0.0,
        causal: bool = True,
        **kwargs,
    ):
        assert alibi is False, 'alibi not supported for FA3'
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
        # TritonAttentionImpl uses -1 as its disabled-softcap sentinel, while
        # FlashAttention-3 requires exactly 0.0 to select non-softcap kernels.
        self.logit_softcapping = max(float(logit_softcapping), 0.0)
        from lmdeploy.pytorch.third_party.flash_attn_interface import flash_attn_varlen_func, flash_attn_with_kvcache
        self.flash_attn_varlen_func_v3 = flash_attn_varlen_func
        self.flash_attn_with_kvcache_v3 = flash_attn_with_kvcache

    def get_step_metadata_provider(self):
        """Describe metadata required by this selected implementation."""
        return FA3AttentionMetaBuilder(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            v_head_size=self.v_head_size,
            sliding_window=self.sliding_window,
            causal=self.causal,
            has_softcap=self.logit_softcapping > 0,
        )

    def _get_scheduler_metadata(self, attn_metadata: TritonAttentionMetadata):
        kernel_metadata = self.get_step_kernel_metadata(attn_metadata)
        if kernel_metadata is None:
            return attn_metadata.scheduler_metadata
        assert isinstance(kernel_metadata, FA3AttentionMetadata)
        return kernel_metadata.scheduler_metadata

    def _get_max_q_seqlen(
        self,
        query: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
    ) -> int:
        """Get max q seqlen."""
        max_q_seqlen = query.numel() // (query.size(-1) * query.size(-2))
        if attn_metadata.is_decoding:
            batch_size = attn_metadata.q_seqlens.size(0)
            max_q_seqlen = max_q_seqlen // batch_size
        return max_q_seqlen

    def _normalize_sliding_window(self, sliding_window):
        """Normalize sliding window to tuple format.

        Args:
            sliding_window: Sliding window size (None, int, or tuple).

        Returns:
            Tuple of (left_window, right_window) or (-1, -1) if None.
        """
        if sliding_window is None:
            return (-1, -1)
        if isinstance(sliding_window, int):
            return (sliding_window, sliding_window)
        return sliding_window

    def _decoding_speculative(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        max_q_seqlen: int,
    ) -> torch.Tensor:
        """Speculative decoding with multi-token queries.

        This path handles speculative decoding where multiple tokens are generated
        in parallel (max_q_seqlen > 1). Uses FA3's flash_attn_with_kvcache for
        efficient batched computation.

        Args:
            query: Query tensor to unflatten.
            k_cache: Key cache tensor.
            v_cache: Value cache tensor.
            attn_metadata: Attention metadata.
            max_q_seqlen: Maximum query sequence length (> 1).

        Returns:
            Attention output tensor.
        """
        quant_policy = attn_metadata.quant_policy

        # TurboQuant stores packed uint8 data in cache, which FA3's native
        # flash_attn_with_kvcache cannot dequantize directly.
        if quant_policy == QuantPolicy.TURBO_QUANT:
            raise NotImplementedError(
                'quant_policy=QuantPolicy.TURBO_QUANT is not supported with '
                'FA3 speculative decoding (max_q_seqlen > 1). '
                'FA3 speculative decoding accesses raw KV cache directly '
                'and cannot dequantize TurboQuant packed data. '
                'Use standard decoding (max_q_seqlen=1).'
            )

        block_offsets = attn_metadata.block_offsets
        sliding_window = self._normalize_sliding_window(self.sliding_window)

        # Reshape query for batched processing
        query = query.unflatten(0, (-1, max_q_seqlen))

        attn_output = self.flash_attn_with_kvcache_v3(
            query,
            k_cache,
            v_cache,
            cache_seqlens=attn_metadata.kv_seqlens.to(torch.int32),
            max_seqlen_q=max_q_seqlen,
            scheduler_metadata=self._get_scheduler_metadata(attn_metadata),
            page_table=block_offsets,
            softmax_scale=self.scale,
            causal=self.causal,
            window_size=sliding_window,
            softcap=self.logit_softcapping,
        )
        return attn_output

    def _decoding_standard(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        max_q_seqlen: int,
        k_scales_zeros: torch.Tensor = None,
        v_scales_zeros: torch.Tensor = None,
    ) -> torch.Tensor:
        """Standard single-token decoding.

        This path handles standard decoding where only one token is generated
        per request (max_q_seqlen = 1). Uses paged attention for memory efficiency.

        Args:
            query: Query tensor (single token per request).
            k_cache: Key cache tensor.
            v_cache: Value cache tensor.
            attn_metadata: Attention metadata.
            max_q_seqlen: Maximum query sequence length (= 1).
            k_scales_zeros: Key quantization scales/zeros.
            v_scales_zeros: Value quantization scales/zeros.

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
            scheduler_metadata=self._get_scheduler_metadata(attn_metadata),
            softmax_scale=self.scale,
            causal=self.causal,
            softcap=self.logit_softcapping,
            window_size=self.sliding_window,
            # custom args
            k_scales_zeros=k_scales_zeros,
            v_scales_zeros=v_scales_zeros,
            quant_policy=quant_policy,
        )
        return attn_output

    def _forward_decoding(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        max_q_seqlen: int,
        k_scales_zeros: torch.Tensor = None,
        v_scales_zeros: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass for decoding stage.

        Supports two decoding modes:
        1. Speculative decoding: Multiple tokens (max_q_seqlen > 1)
        2. Standard decoding: Single token (max_q_seqlen = 1)

        Args:
            query: Query tensor.
            k_cache: Key cache tensor.
            v_cache: Value cache tensor.
            attn_metadata: Attention metadata.
            max_q_seqlen: Maximum query sequence length.
            k_scales_zeros: Key quantization scales/zeros.
            v_scales_zeros: Value quantization scales/zeros.

        Returns:
            Attention output tensor.
        """
        if max_q_seqlen > 1:
            return self._decoding_speculative(query, k_cache, v_cache, attn_metadata, max_q_seqlen)
        return self._decoding_standard(
            query,
            k_cache,
            v_cache,
            attn_metadata,
            max_q_seqlen,
            k_scales_zeros,
            v_scales_zeros,
        )

    def _forward_prefill(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        max_q_seqlen: int,
        k_scales_zeros: torch.Tensor = None,
        v_scales_zeros: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass for prefill stage.

        Uses FA3's flash_attn_varlen_func for efficient variable-length attention
        computation during the prefill phase.

        Args:
            query: Query tensor.
            k_cache: Key cache tensor.
            v_cache: Value cache tensor.
            attn_metadata: Attention metadata.
            max_q_seqlen: Maximum query sequence length.
            k_scales_zeros: Key quantization scales/zeros.
            v_scales_zeros: Value quantization scales/zeros.

        Returns:
            Attention output tensor.
        """
        block_offsets = attn_metadata.block_offsets
        kv_start_loc = attn_metadata.kv_start_loc
        kv_seqlens = attn_metadata.kv_seqlens
        kv_flatten_size = attn_metadata.kv_flatten_size
        quant_policy = attn_metadata.quant_policy

        # Flatten KV cache for varlen attention
        block_size = k_cache.size(1)
        out_size = _cdiv(kv_flatten_size, block_size) * block_size + block_size
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
            flatten_kv_layout='shd',
        )

        sliding_window = self._normalize_sliding_window(self.sliding_window)

        # For TurboQuant, flattened K/V are in rotated domain.
        # Rotate Q to match, and inverse-rotate output afterwards.
        if quant_policy == QuantPolicy.TURBO_QUANT:
            from lmdeploy.pytorch.kernels.cuda.turbo_quant import (
                hadamard_rotate,
                hadamard_rotate_inv,
            )
            query = hadamard_rotate(query)

        attn_output = self.flash_attn_varlen_func_v3(
            q=query,
            k=flatten_k,
            v=flatten_v,
            cu_seqlens_q=attn_metadata.cu_seqlens_q,
            cu_seqlens_k=attn_metadata.cu_seqlens_k,
            max_seqlen_q=max_q_seqlen,
            max_seqlen_k=attn_metadata.max_kv_seqlen,
            softmax_scale=self.scale,
            causal=self.causal,
            window_size=sliding_window,
            softcap=self.logit_softcapping,
        )

        # Inverse-rotate output back to original domain
        if quant_policy == QuantPolicy.TURBO_QUANT:
            attn_output = hadamard_rotate_inv(attn_output)

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
    ) -> torch.Tensor:
        """Forward pass for FA3 attention computation.

        This method handles both prefill and decoding stages by:
        1. Computing max query sequence length
        2. Filling KV cache if new key/value are provided
        3. Dispatching to appropriate stage-specific method

        Architecture:
        - Decoding: Supports both speculative (multi-token) and standard (single-token)
        - Prefill: Uses flash_attn_varlen_func for efficient varlen attention

        Args:
            query: Query tensor.
            key: Key tensor (None for decoding-only).
            value: Value tensor (None for decoding-only).
            k_cache: Key cache tensor.
            v_cache: Value cache tensor.
            attn_metadata: Attention metadata containing stage info and indices.
            k_scales_zeros: Key quantization scales/zeros.
            v_scales_zeros: Value quantization scales/zeros.
            learnable_sink: Learnable sink tokens (unused in FA3).
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

        # Dispatch to stage-specific forward method
        if attn_metadata.is_decoding:
            return self._forward_decoding(
                query,
                k_cache,
                v_cache,
                attn_metadata,
                max_q_seqlen,
                k_scales_zeros,
                v_scales_zeros,
            )
        else:
            return self._forward_prefill(
                query,
                k_cache,
                v_cache,
                attn_metadata,
                max_q_seqlen,
                k_scales_zeros,
                v_scales_zeros,
            )
