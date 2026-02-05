# Copyright (c) OpenMMLab. All rights reserved.

import functools

import torch

from lmdeploy.utils import get_logger

from .default import TritonAttentionImpl, TritonAttentionMetadata

logger = get_logger('lmdeploy')


def _cdiv(a, b):
    """Perform div up."""
    return (a + b - 1) // b


def _try_dynamic_compile(func, *args, **kwargs):
    """Try compile."""
    try:
        compiled_func = torch.compile(func, dynamic=True)
        compiled_func(*args, **kwargs)
        return compiled_func
    except Exception:
        return func


class NSAIndicesUpdater:
    """NSA indices updater.

    Flash MLA sparse attention requires different indice format for prefill and decoding. This module is used to update
    the indices to meet the requirements.
    """

    def __init__(self):
        self._update_decode_func = None
        self._update_prefill_func = None

    def _update_decode_impl(self, nsa_indices: torch.Tensor, block_offsets: torch.Tensor,
                            block_size: int) -> torch.Tensor:
        """Update for decode impl."""
        block_ids = nsa_indices // block_size
        block_ids = block_ids.clamp_min(0)
        block_ids = block_offsets.gather(1, block_ids)
        block_remain = nsa_indices % block_size
        ret = block_ids * block_size + block_remain
        ret[nsa_indices < 0] = -1
        return ret[:, None]

    def update_decode(self, nsa_indices: torch.Tensor, block_offsets: torch.Tensor, block_size: int) -> torch.Tensor:
        """Update for decode."""
        if self._update_decode_func is None:
            self._update_decode_func = _try_dynamic_compile(self._update_decode_impl, nsa_indices, block_offsets,
                                                            block_size)

        return self._update_decode_func(nsa_indices, block_offsets, block_size)

    def _update_prefill_impl(self, nsa_indices: torch.Tensor, q_seqlens: torch.Tensor, cu_seqlens_k: torch.Tensor):
        """Update for prefill impl."""
        num_tokens = nsa_indices.size(0)
        repeat_cu_seqlens_k = torch.repeat_interleave(cu_seqlens_k[:-1], q_seqlens, output_size=num_tokens)
        neg_mask = nsa_indices < 0
        nsa_indices = nsa_indices + repeat_cu_seqlens_k[:, None]
        nsa_indices[neg_mask] = -1
        return nsa_indices[:, None]

    def update_prefill(self, nsa_indices: torch.Tensor, q_seqlens: torch.Tensor, cu_seqlens_k: torch.Tensor):
        """Update for prefill."""
        if self._update_prefill_func is None:
            self._update_prefill_func = _try_dynamic_compile(self._update_prefill_impl, nsa_indices, q_seqlens,
                                                             cu_seqlens_k)

        return self._update_prefill_func(nsa_indices, q_seqlens, cu_seqlens_k)

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def build():
        return NSAIndicesUpdater()


class FlashMLAImpl(TritonAttentionImpl):
    """Flash MLA (Multi-head Latent Attention) implementation.

    This implementation supports multiple execution paths:
    - Decoding: Uses flash_mla_with_kvcache with paged KV cache
    - Prefill with NSA: Uses flash_mla_sparse_fwd for sparse attention
    - Prefill with FA3: Uses flash_attn_varlen_func with split q_rope/q_nope
    - Prefill fallback: Uses custom Triton kernel
    """

    # MLA-specific constants
    _MLA_HEAD_ALIGNMENT = 64  # Query heads must be multiple of 64 for flash_mla
    _MLA_NOPE_SIZE = 512  # Size of non-positional embeddings
    _MLA_SCALE_SIZE = 16  # Size of FP8 quantization scales

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
        use_fa3: bool = False,
        **kwargs,
    ):
        assert (sliding_window is None
                or all(win == -1 for win in sliding_window)), ('sliding window not supported for FlashMLA')
        assert alibi is False, 'alibi not supported for FlashMLA'
        if logit_softcapping > 0.0:
            logger.warning('logit_softcapping not properly supported for FlashMLA, using -1.0')
            logit_softcapping = -1.0
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

        import flash_mla

        from lmdeploy.pytorch.kernels.cuda.fill_kv_cache import fill_kv_cache_blocked_fp8
        from lmdeploy.pytorch.kernels.cuda.flatten_kv_cache import flatten_kv_cache_mla_fp8
        self.flash_mla_with_kvcache = flash_mla.flash_mla_with_kvcache
        self.flash_mla_sparse_fwd = None
        self.fill_kv_cache_blocked_fp8 = fill_kv_cache_blocked_fp8
        self.flatten_kv_cache_mla_fp8 = flatten_kv_cache_mla_fp8
        assert num_kv_heads == 1, 'MLA requires num kv heads equal to 1'
        self.use_fa3 = use_fa3

        self.nsa_updater = NSAIndicesUpdater.build()

    def _get_flash_mla_sparse_fwd(self):
        if self.flash_mla_sparse_fwd is not None:
            return self.flash_mla_sparse_fwd

        try:
            import flash_mla
            self.flash_mla_sparse_fwd = flash_mla.flash_mla_sparse_fwd
            return self.flash_mla_sparse_fwd
        except Exception:
            logger.exception('Can not import flash_mla_sparse_fwd from flash_mla.')

    def flash_mla_decoding(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        nsa_indices: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
    ):
        """Flash mla decoding."""
        causal = self.causal
        kv_seqlens = attn_metadata.kv_seqlens
        block_offsets = attn_metadata.block_offsets
        is_fp8_kvcache = k_cache.dtype == torch.float8_e4m3fn

        q_seqlens = attn_metadata.q_seqlens
        batch_size = q_seqlens.size(0)
        max_q_seqlen = query.numel() // (query.size(-1) * query.size(-2))
        max_q_seqlen = max_q_seqlen // batch_size
        query = query.unflatten(0, (batch_size, max_q_seqlen))
        if kv_seqlens.dtype == torch.int64:
            kv_seqlens = kv_seqlens.to(torch.int32)

        # update nsa indice according to flash-mla requirement
        if nsa_indices is not None:
            block_size = k_cache.size(1)
            nsa_indices = self.nsa_updater.update_decode(nsa_indices, block_offsets, block_size)
            causal = False

        attn_output, _ = self.flash_mla_with_kvcache(query,
                                                     k_cache=k_cache,
                                                     block_table=block_offsets,
                                                     cache_seqlens=kv_seqlens,
                                                     head_dim_v=self.v_head_size,
                                                     softmax_scale=self.scale,
                                                     tile_scheduler_metadata=attn_metadata.tile_scheduler_metadata,
                                                     num_splits=attn_metadata.num_splits,
                                                     causal=causal,
                                                     is_fp8_kvcache=is_fp8_kvcache,
                                                     indices=nsa_indices)

        attn_output = attn_output.flatten(0, 1)
        return attn_output

    def _prefill_sparse(self, query: torch.Tensor, flatten_k: torch.Tensor, nsa_indices: torch.Tensor,
                        attn_metadata: TritonAttentionMetadata) -> torch.Tensor:
        """Sparse prefill using flash_mla_sparse_fwd.

        This path is used when NSA (Non-contiguous Sparse Attention) indices are provided.
        Requires FP8 KV cache and flash_mla library.

        Args:
            query: Query tensor.
            flatten_k: Flattened key cache.
            nsa_indices: Sparse attention indices.
            attn_metadata: Attention metadata.

        Returns:
            Attention output tensor.
        """
        q_seqlens = attn_metadata.q_seqlens
        flash_mla_sparse_fwd = self._get_flash_mla_sparse_fwd()

        num_q_heads = query.size(1)
        # flash_mla_sparse_fwd requires query heads to be multiple of alignment
        if num_q_heads % self._MLA_HEAD_ALIGNMENT != 0:
            padding = self._MLA_HEAD_ALIGNMENT - num_q_heads % self._MLA_HEAD_ALIGNMENT
            query = torch.nn.functional.pad(query, (0, 0, 0, padding))

        nsa_indices = self.nsa_updater.update_prefill(nsa_indices, q_seqlens, attn_metadata.cu_seqlens_k)
        output = flash_mla_sparse_fwd(
            query,
            flatten_k,
            nsa_indices,
            sm_scale=self.scale,
        )
        attn_output = output[0]
        attn_output = attn_output[:, :num_q_heads]
        return attn_output

    def _prefill_triton(
        self,
        query: torch.Tensor,
        flatten_k: torch.Tensor,
        flatten_v: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
    ) -> torch.Tensor:
        """Triton-based prefill fallback.

        This is the fallback path when Flash Attention 3 is not available.
        Uses custom Triton kernel for attention computation.

        Args:
            query: Query tensor.
            flatten_k: Flattened key cache.
            flatten_v: Flattened value cache.
            attn_metadata: Attention metadata.

        Returns:
            Attention output tensor.
        """
        max_q_seqlen = query.numel() // (query.size(-1) * query.size(-2))

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
        )

        return attn_output

    def _prefill_fa3(
        self,
        query: torch.Tensor,
        flatten_k: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
    ) -> torch.Tensor:
        """Flash Attention 3 optimized prefill.

        This path uses Flash Attention 3's optimized kernels with split
        rope (positional) and nope (non-positional) components.

        Args:
            query: Query tensor.
            flatten_k: Flattened key cache.
            attn_metadata: Attention metadata.

        Returns:
            Attention output tensor.
        """
        max_q_seqlen = query.numel() // (query.size(-1) * query.size(-2))
        kv_flatten_size = attn_metadata.kv_flatten_size
        causal = self.causal

        # Split query and key into rope (positional) and nope (non-positional) parts
        q_rope = query[:, :, self.v_head_size:]
        q_nope = query[:, :, :self.v_head_size]
        k_rope = flatten_k.view(kv_flatten_size, self.num_kv_heads, -1)[:, :, self.v_head_size:]
        c_kv = flatten_k.view(kv_flatten_size, self.num_kv_heads, -1)[:, :, :self.v_head_size]
        from lmdeploy.pytorch.third_party.flash_attn_interface import flash_attn_varlen_func
        attn_output = flash_attn_varlen_func(
            q=q_rope,
            k=k_rope,
            v=c_kv,
            qv=q_nope,
            cu_seqlens_q=attn_metadata.cu_seqlens_q,
            cu_seqlens_k=attn_metadata.cu_seqlens_k,
            max_seqlen_q=max_q_seqlen,
            max_seqlen_k=kv_flatten_size,
            softmax_scale=self.scale,
            causal=causal,
            window_size=(-1, -1) if self.sliding_window is None else self.sliding_window,
        )
        return attn_output

    def run_flatten_kv_cache(self,
                             k_cache: torch.Tensor,
                             v_cache: torch.Tensor,
                             attn_metadata: TritonAttentionMetadata,
                             out_dtype: torch.dtype,
                             is_nsa: bool,
                             k_scales_zeros: torch.Tensor = None,
                             v_scales_zeros: torch.Tensor = None):
        """Flatten kv cache for prefill."""

        kv_start_loc = attn_metadata.kv_start_loc
        kv_seqlens = attn_metadata.kv_seqlens
        block_offsets = attn_metadata.block_offsets
        kv_flatten_size = attn_metadata.kv_flatten_size
        quant_policy = attn_metadata.quant_policy
        is_fp8_kvcache = k_cache.dtype == torch.float8_e4m3fn
        BLOCK_BS = k_cache.size(1)

        # pad one more block to avoid invalid kv visit
        if self.use_fa3 or is_nsa:
            out_size = kv_flatten_size
            flatten_kv_layout = 'shd'
        else:
            out_size = (_cdiv(kv_flatten_size, BLOCK_BS) * BLOCK_BS + BLOCK_BS)
            flatten_kv_layout = 'hsd'

        if is_fp8_kvcache:
            flatten_k = self.flatten_kv_cache_mla_fp8(
                k_cache,
                kv_seqlens,
                block_offsets,
                start_loc=kv_start_loc,
                out_size=out_size,
                out_dtype=out_dtype,
                flatten_kv_layout=flatten_kv_layout,
            )
            flatten_v = flatten_k[..., :self._MLA_NOPE_SIZE]
        else:
            flatten_k, flatten_v = self.flatten_kv_cache(
                k_cache,
                v_cache,
                kv_seqlens,
                block_offsets,
                start_loc=kv_start_loc,
                out_size=out_size,
                out_dtype=out_dtype,
                k_scales_zeros=k_scales_zeros,
                v_scales_zeros=v_scales_zeros,
                quant_policy=quant_policy,
                flatten_kv_layout=flatten_kv_layout,
            )

        return flatten_k, flatten_v

    def _get_max_q_seqlen(
        self,
        query: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
    ) -> int:
        """Get max q seqlen."""
        q_seqlens = attn_metadata.q_seqlens
        max_q_seqlen = query.numel() // (query.size(-1) * query.size(-2))
        batch_size = q_seqlens.size(0)
        if attn_metadata.is_decoding:
            max_q_seqlen = max_q_seqlen // batch_size
        return max_q_seqlen

    def _fill_kv_cache_impl(self,
                            key: torch.Tensor,
                            value: torch.Tensor,
                            k_cache: torch.Tensor,
                            v_cache: torch.Tensor,
                            attn_metadata: TritonAttentionMetadata,
                            max_q_seqlen: int,
                            k_scales_zeros: torch.Tensor = None,
                            v_scales_zeros: torch.Tensor = None):
        """Fill kv cache."""
        is_fp8_kvcache = k_cache.dtype == torch.float8_e4m3fn
        if not is_fp8_kvcache:
            return super()._fill_kv_cache_impl(
                key,
                value,
                k_cache,
                v_cache,
                attn_metadata,
                max_q_seqlen,
                k_scales_zeros=k_scales_zeros,
                v_scales_zeros=v_scales_zeros,
            )

        block_offsets = attn_metadata.block_offsets
        kv_seqlens = attn_metadata.kv_seqlens
        quant_policy = attn_metadata.quant_policy
        assert quant_policy == 0

        # fill seqlen args
        fill_seqlens, fill_max_q_seqlen, fill_q_start_loc = self._get_fill_meta(
            key,
            attn_metadata,
            max_q_seqlen,
        )

        # Split k_cache into nope, scale, and pe components
        scale_offset = self._MLA_NOPE_SIZE
        scale_end = scale_offset + self._MLA_SCALE_SIZE
        k_cache_scale = k_cache[..., scale_offset:scale_end].view(torch.float32)
        k_cache_nope = k_cache[..., :self._MLA_NOPE_SIZE]
        k_cache_pe = k_cache[..., scale_end:].view(key.dtype)
        self.fill_kv_cache_blocked_fp8(
            key[..., :self._MLA_NOPE_SIZE],
            None,
            k_cache_nope,
            None,
            k_cache_scale,
            None,
            cu_seqlen_q=attn_metadata.cu_seqlens_q,
            kv_seqlens=attn_metadata.kv_seqlens,
            max_q_seqlen=max_q_seqlen,
            block_offsets=block_offsets,
            group_size=128,
            scale_fmt='ue8m0',
        )
        self.fill_kv_cache(
            key[..., self._MLA_NOPE_SIZE:],
            None,
            k_cache_pe,
            None,
            fill_q_start_loc,
            fill_seqlens,
            kv_seq_length=kv_seqlens,
            max_q_seq_length=fill_max_q_seqlen,
            block_offsets=block_offsets,
        )

    def _forward_decoding(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        nsa_indices: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass for decoding stage.

        Uses flash_mla_with_kvcache for efficient decoding with paged KV cache.
        Supports both regular and sparse (NSA) attention patterns.

        Args:
            query: Query tensor.
            k_cache: Key cache tensor.
            attn_metadata: Attention metadata.
            nsa_indices: Optional sparse attention indices.

        Returns:
            Attention output tensor.
        """
        return self.flash_mla_decoding(query, k_cache, nsa_indices, attn_metadata)

    def _forward_prefill(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        nsa_indices: torch.Tensor = None,
        k_scales_zeros: torch.Tensor = None,
        v_scales_zeros: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass for prefill stage.

        Supports three execution paths:
        1. Sparse (NSA + FP8): flash_mla_sparse_fwd for sparse attention
        2. FA3 optimized: flash_attn_varlen_func with split q_rope/q_nope
        3. Triton fallback: Custom Triton kernel implementation

        Args:
            query: Query tensor.
            k_cache: Key cache tensor.
            v_cache: Value cache tensor.
            attn_metadata: Attention metadata.
            nsa_indices: Optional sparse attention indices.
            k_scales_zeros: Key quantization scales/zeros.
            v_scales_zeros: Value quantization scales/zeros.

        Returns:
            Attention output tensor.
        """
        # Flatten KV cache once for all prefill paths
        flatten_k, flatten_v = self.run_flatten_kv_cache(
            k_cache,
            v_cache,
            attn_metadata,
            out_dtype=query.dtype,
            is_nsa=nsa_indices is not None,
            k_scales_zeros=k_scales_zeros,
            v_scales_zeros=v_scales_zeros,
        )

        # Dispatch to appropriate prefill implementation
        if nsa_indices is not None:
            return self._prefill_sparse(query, flatten_k, nsa_indices, attn_metadata)
        elif self.use_fa3:
            return self._prefill_fa3(query, flatten_k, attn_metadata)
        else:
            return self._prefill_triton(query, flatten_k, flatten_v, attn_metadata)

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
        nsa_indices: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for MLA attention computation.

        This method handles both prefill and decoding stages by:
        1. Validating NSA requirements (FP8 KV cache)
        2. Computing max query sequence length
        3. Filling KV cache if new key/value are provided
        4. Dispatching to appropriate stage-specific method

        Architecture:
        - Decoding: Uses flash_mla_with_kvcache with paged KV cache
        - Prefill: Three paths based on availability and requirements
          * Sparse (NSA + FP8): flash_mla_sparse_fwd
          * FA3 optimized: flash_attn_varlen_func with split q_rope/q_nope
          * Triton fallback: Custom triton kernel

        Args:
            query: Query tensor.
            key: Key tensor (None for decoding-only).
            value: Value tensor (None for decoding-only).
            k_cache: Key cache tensor.
            v_cache: Value cache tensor.
            attn_metadata: Attention metadata containing stage info and indices.
            k_scales_zeros: Key quantization scales/zeros.
            v_scales_zeros: Value quantization scales/zeros.
            nsa_indices: Optional sparse attention indices.

        Returns:
            Attention output tensor.
        """
        # Validate NSA requirements
        is_nsa = nsa_indices is not None
        if is_nsa:
            is_fp8_kvcache = k_cache.dtype == torch.float8_e4m3fn
            assert is_fp8_kvcache, 'NSA sparse attention requires FP8 KV cache'

        # Shared preparation
        max_q_seqlen = self._get_max_q_seqlen(query, attn_metadata)

        # Fill KV cache with new key/value if provided
        self._fill_kv_cache_impl(
            key,
            value,
            k_cache,
            v_cache,
            attn_metadata,
            max_q_seqlen,
            k_scales_zeros=k_scales_zeros,
            v_scales_zeros=v_scales_zeros,
        )

        # Dispatch to stage-specific forward method
        if attn_metadata.is_decoding:
            return self._forward_decoding(query, k_cache, attn_metadata, nsa_indices)
        else:
            return self._forward_prefill(
                query,
                k_cache,
                v_cache,
                attn_metadata,
                nsa_indices,
                k_scales_zeros,
                v_scales_zeros,
            )
