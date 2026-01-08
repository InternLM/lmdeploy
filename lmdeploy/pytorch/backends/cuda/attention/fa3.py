# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.utils import get_logger

from .default import TritonAttentionImpl, TritonAttentionMetadata

logger = get_logger('lmdeploy')


class FA3Impl(TritonAttentionImpl):
    """Flash Attention 3 implementation.

    This implementation leverages Flash Attention 3's optimized kernels for both
    prefill and decoding stages. FA3 provides significant performance improvements
    on Hopper architecture (SM90) with CUDA >= 12.3.

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
        from lmdeploy.pytorch.third_party.flash_attn_interface import flash_attn_varlen_func, flash_attn_with_kvcache
        self.flash_attn_varlen_func_v3 = flash_attn_varlen_func
        self.flash_attn_with_kvcache_v3 = flash_attn_with_kvcache

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
            scheduler_metadata=attn_metadata.scheduler_metadata,
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
            scheduler_metadata=attn_metadata.scheduler_metadata,
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
        else:
            return self._decoding_standard(query, k_cache, v_cache, attn_metadata, max_q_seqlen, k_scales_zeros,
                                           v_scales_zeros)

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
        flatten_k, flatten_v = self.flatten_kv_cache(
            k_cache,
            v_cache,
            kv_seqlens,
            block_offsets,
            start_loc=kv_start_loc,
            out_size=kv_flatten_size,
            out_dtype=query.dtype,
            k_scales_zeros=k_scales_zeros,
            v_scales_zeros=v_scales_zeros,
            quant_policy=quant_policy,
            flatten_kv_layout='shd',
        )

        sliding_window = self._normalize_sliding_window(self.sliding_window)

        attn_output = self.flash_attn_varlen_func_v3(
            q=query,
            k=flatten_k,
            v=flatten_v,
            cu_seqlens_q=attn_metadata.cu_seqlens_q,
            cu_seqlens_k=attn_metadata.cu_seqlens_k,
            max_seqlen_q=max_q_seqlen,
            max_seqlen_k=kv_flatten_size,
            softmax_scale=self.scale,
            causal=self.causal,
            window_size=sliding_window,
            softcap=self.logit_softcapping,
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
