# Copyright (c) OpenMMLab. All rights reserved.

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any

import torch

from lmdeploy.messages import QuantPolicy
from lmdeploy.utils import get_logger

from ..step_metadata import CudaAttentionMetaBuilder
from .default import TritonAttentionImpl, TritonAttentionMetadata

logger = get_logger('lmdeploy')


@dataclass
class FlashMLAAttentionMetadata:
    """Scheduler metadata owned by one FlashMLA configuration."""

    # FlashMLA <= 0.x returns tensors here. FlashMLA 1.x returns a
    # FlashMLASchedMeta object that lazily owns those tensors.
    tile_scheduler_metadata: Any = None
    num_splits: torch.Tensor | None = None
    scheduler_depends_on_step: bool = False


def needs_flash_mla_scheduler(is_fp8_kvcache: bool, index_topk: int | None) -> bool:
    """Whether the selected FlashMLA path consumes paged scheduler metadata."""
    # BF16 sparse decode uses sparse_fwd rather than the paged decode kernel.
    return is_fp8_kvcache or index_topk is None


def _build_flash_mla_metadata(kv_seqlens,
                              num_attention_heads: int,
                              decoding_query_len: int,
                              is_fp8_kvcache: bool,
                              index_topk: int | None) -> FlashMLAAttentionMetadata:
    """Build scheduler metadata from one selected FlashMLA implementation."""
    if not needs_flash_mla_scheduler(is_fp8_kvcache, index_topk):
        return FlashMLAAttentionMetadata()

    import flash_mla

    num_attention_heads *= decoding_query_len
    num_heads_q = None if index_topk is None else num_attention_heads
    tile_scheduler_metadata, num_splits = flash_mla.get_mla_metadata(
        kv_seqlens.to(torch.int32),
        num_attention_heads,
        num_heads_k=1,
        num_heads_q=num_heads_q,
        is_fp8_kvcache=is_fp8_kvcache,
        topk=index_topk,
    )
    return FlashMLAAttentionMetadata(
        tile_scheduler_metadata=tile_scheduler_metadata,
        num_splits=num_splits,
        # The dense scheduler reads kv_seqlens. The current sparse decode call
        # uses a fixed top-k width and does not pass topk_length.
        scheduler_depends_on_step=index_topk is None,
    )


def build_flash_mla_metadata(sequence_metadata, **kwargs) -> FlashMLAAttentionMetadata:
    """Build scheduler metadata from one selected FlashMLA implementation."""
    return _build_flash_mla_metadata(sequence_metadata.kv_seqlens, **kwargs)


def update_flash_mla_metadata(attn_metadata,
                              num_attention_heads: int,
                              decoding_query_len: int,
                              is_fp8_kvcache: bool,
                              index_topk: int | None) -> None:
    """Populate the legacy single-group FlashMLA metadata fields."""
    metadata = build_flash_mla_metadata(
        attn_metadata,
        num_attention_heads=num_attention_heads,
        decoding_query_len=decoding_query_len,
        is_fp8_kvcache=is_fp8_kvcache,
        index_topk=index_topk,
    )
    attn_metadata.tile_scheduler_metadata = metadata.tile_scheduler_metadata
    attn_metadata.num_splits = metadata.num_splits


def build_flash_mla_graph_metadata(step_context, kv_seqlens,
                                   decoding_query_len: int) -> FlashMLAAttentionMetadata:
    """Build legacy graph metadata from the model-level FlashMLA
    configuration."""
    num_attention_heads, _ = step_context.model_config.get_num_qkv_head_by_tp()
    model_config = step_context.model_config
    return _build_flash_mla_metadata(
        kv_seqlens,
        num_attention_heads=num_attention_heads,
        decoding_query_len=decoding_query_len,
        is_fp8_kvcache=model_config.use_mla_fp8_cache,
        index_topk=model_config.mla_index_topk,
    )


@dataclass(frozen=True)
class FlashMLAAttentionMetaBuilder(
        CudaAttentionMetaBuilder[FlashMLAAttentionMetadata, FlashMLAAttentionMetadata]):
    """Build metadata requested by one selected FlashMLA configuration."""

    num_attention_heads: int
    index_topk: int | None = None

    @property
    def key(self) -> Hashable:
        return (type(self), self.num_attention_heads, self.index_topk)

    def build(self, step_context, sequence_metadata) -> FlashMLAAttentionMetadata:
        if not step_context.is_decoding:
            return FlashMLAAttentionMetadata()
        batch_size = sequence_metadata.q_seqlens.size(0)
        return build_flash_mla_metadata(
            sequence_metadata,
            num_attention_heads=self.num_attention_heads,
            decoding_query_len=step_context.input_ids.size(1) // batch_size,
            is_fp8_kvcache=step_context.model_config.use_mla_fp8_cache,
            index_topk=self.index_topk,
        )

    def apply_legacy_metadata(self, attn_metadata, metadata: FlashMLAAttentionMetadata) -> None:
        attn_metadata.tile_scheduler_metadata = metadata.tile_scheduler_metadata
        attn_metadata.num_splits = metadata.num_splits

    def make_cudagraph_buffer(self, graph_meta, input_buffers,
                              step_context) -> FlashMLAAttentionMetadata:
        return _build_flash_mla_metadata(
            torch.ones(graph_meta.max_batchs, dtype=torch.int32, device=graph_meta.device),
            num_attention_heads=self.num_attention_heads,
            decoding_query_len=graph_meta.decode_query_len,
            is_fp8_kvcache=step_context.model_config.use_mla_fp8_cache,
            index_topk=self.index_topk,
        )

    def fill_cudagraph_buffer(self, graph_meta, input_buffers, step_context,
                              buffer: FlashMLAAttentionMetadata) -> FlashMLAAttentionMetadata:
        tile_scheduler_metadata = buffer.tile_scheduler_metadata
        if not isinstance(tile_scheduler_metadata, torch.Tensor):
            # FlashMLA 1.x initializes this object during the first kernel
            # call. The pre-capture lifecycle decides whether the warmup
            # scheduler is reusable or must be replaced.
            assert buffer.num_splits is None
            return buffer

        metadata = _build_flash_mla_metadata(
            input_buffers['kv_seqlens'],
            num_attention_heads=self.num_attention_heads,
            decoding_query_len=graph_meta.decode_query_len,
            is_fp8_kvcache=step_context.model_config.use_mla_fp8_cache,
            index_topk=self.index_topk,
        )
        tile_scheduler_metadata.copy_(metadata.tile_scheduler_metadata)
        assert buffer.num_splits is not None and metadata.num_splits is not None
        buffer.num_splits.copy_(metadata.num_splits)
        return buffer

    def prepare_cudagraph_capture(self, graph_meta, input_buffers, step_context,
                                  buffer: FlashMLAAttentionMetadata) -> None:
        scheduler = buffer.tile_scheduler_metadata
        if isinstance(scheduler, torch.Tensor) or not buffer.scheduler_depends_on_step:
            return

        # FlashMLA 1.x only launches its scheduler kernel when these fields are
        # empty. Warmup initialized the old object, so capture must use a fresh
        # one to record metadata generation from the graph's input buffers.
        import flash_mla
        scheduler, num_splits = flash_mla.get_mla_metadata()
        assert num_splits is None
        buffer.tile_scheduler_metadata = scheduler
        buffer.num_splits = num_splits


def _cdiv(a, b):
    """Perform div up."""
    return (a + b - 1) // b


class FlashMLAImpl(TritonAttentionImpl):
    """Dense MLA attention.

    Prefill: FA3 when available; otherwise the Triton MLA kernel.
    Decode: paged FlashMLA.

    DSA-specific index mapping and sparse execution live in
    :class:`FlashMLASparseImpl`.
    """

    # MLA-specific constants
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
        self.fill_kv_cache_blocked_fp8 = fill_kv_cache_blocked_fp8
        self.flatten_kv_cache_mla_fp8 = flatten_kv_cache_mla_fp8
        assert num_kv_heads == 1, 'MLA requires num kv heads equal to 1'
        self.use_fa3 = use_fa3

    def get_step_metadata_provider(self):
        """Describe metadata required by this selected implementation."""
        return FlashMLAAttentionMetaBuilder(num_attention_heads=self.num_heads)

    def _get_scheduler_metadata(self, attn_metadata: TritonAttentionMetadata):
        kernel_metadata = self.get_step_kernel_metadata(attn_metadata)
        if kernel_metadata is None:
            return attn_metadata.tile_scheduler_metadata, attn_metadata.num_splits
        assert isinstance(kernel_metadata, FlashMLAAttentionMetadata)
        return kernel_metadata.tile_scheduler_metadata, kernel_metadata.num_splits

    def _decoding_paged(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        indices: torch.Tensor = None,
        causal: bool = None,
    ):
        """Run paged FlashMLA decode with optional provider-ready indices."""
        if causal is None:
            causal = self.causal
        kv_seqlens = attn_metadata.kv_seqlens
        block_offsets = attn_metadata.block_offsets
        is_fp8_kvcache = k_cache.dtype == torch.float8_e4m3fn

        q_seqlens = attn_metadata.q_seqlens
        batch_size = q_seqlens.size(0)
        max_q_seqlen = query.numel() // (query.size(-1) * query.size(-2))
        max_q_seqlen = max_q_seqlen // batch_size
        query = query.unflatten(0, (batch_size, max_q_seqlen))
        num_q_heads = query.size(2)
        if kv_seqlens.dtype == torch.int64:
            kv_seqlens = kv_seqlens.to(torch.int32)

        tile_scheduler_metadata, num_splits = self._get_scheduler_metadata(attn_metadata)

        attn_output, _ = self.flash_mla_with_kvcache(query,
                                                     k_cache=k_cache,
                                                     block_table=block_offsets,
                                                     cache_seqlens=kv_seqlens,
                                                     head_dim_v=self.v_head_size,
                                                     softmax_scale=self.scale,
                                                     tile_scheduler_metadata=tile_scheduler_metadata,
                                                     num_splits=num_splits,
                                                     causal=causal,
                                                     is_fp8_kvcache=is_fp8_kvcache,
                                                     indices=indices)

        attn_output = attn_output[:, :, :num_q_heads]
        attn_output = attn_output.flatten(0, 1)
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

    def _flatten_prefill_kv_cache(self,
                                  k_cache: torch.Tensor,
                                  v_cache: torch.Tensor,
                                  attn_metadata: TritonAttentionMetadata,
                                  out_dtype: torch.dtype,
                                  kv_layout: str,
                                  k_scales_zeros: torch.Tensor = None,
                                  v_scales_zeros: torch.Tensor = None):
        """Flatten paged KV into the layout required by prefill."""

        kv_start_loc = attn_metadata.kv_start_loc
        kv_seqlens = attn_metadata.kv_seqlens
        block_offsets = attn_metadata.block_offsets
        kv_flatten_size = attn_metadata.kv_flatten_size
        quant_policy = attn_metadata.quant_policy
        is_fp8_kvcache = k_cache.dtype == torch.float8_e4m3fn
        block_size = k_cache.size(1)

        # pad one more block to avoid invalid kv visit
        if kv_layout == 'shd':
            out_size = kv_flatten_size
        else:
            out_size = _cdiv(kv_flatten_size, block_size) * block_size + block_size

        if is_fp8_kvcache:
            flatten_k = self.flatten_kv_cache_mla_fp8(
                k_cache,
                kv_seqlens,
                block_offsets,
                start_loc=kv_start_loc,
                out_size=out_size,
                out_dtype=out_dtype,
                flatten_kv_layout=kv_layout,
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
                flatten_kv_layout=kv_layout,
            )
            if flatten_v.size(-1) == 0:
                # BF16 MLA stores the latent value in the leading K payload;
                # its standalone V cache is intentionally empty.
                flatten_v = flatten_k[..., :self.v_head_size]

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
            # The BF16 MLA V cache aliases K, and the base writer skips its duplicate store.
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
        assert quant_policy == QuantPolicy.NONE

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
        """Forward pass for dense MLA decoding.

        Args:
            query: Query tensor.
            k_cache: Key cache tensor.
            attn_metadata: Attention metadata.
            nsa_indices: Must be ``None`` for dense MLA.

        Returns:
            Attention output tensor.
        """
        if nsa_indices is not None:
            raise RuntimeError('Sparse MLA indices require FlashMLASparseImpl.')
        return self._decoding_paged(query, k_cache, attn_metadata)

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
        """Forward pass for dense MLA prefill.

        Args:
            query: Query tensor.
            k_cache: Key cache tensor.
            v_cache: Value cache tensor.
            attn_metadata: Attention metadata.
            nsa_indices: Must be ``None`` for dense MLA.
            k_scales_zeros: Key quantization scales/zeros.
            v_scales_zeros: Value quantization scales/zeros.

        Returns:
            Attention output tensor.
        """
        if nsa_indices is not None:
            raise RuntimeError('Sparse MLA indices require FlashMLASparseImpl.')

        kv_layout = 'shd' if self.use_fa3 else 'hsd'
        flatten_k, flatten_v = self._flatten_prefill_kv_cache(
            k_cache,
            v_cache,
            attn_metadata,
            out_dtype=query.dtype,
            kv_layout=kv_layout,
            k_scales_zeros=k_scales_zeros,
            v_scales_zeros=v_scales_zeros,
        )

        if self.use_fa3:
            return self._prefill_fa3(query, flatten_k, attn_metadata)
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
        """Forward pass for dense MLA attention.

        This method handles both prefill and decoding stages by:
        1. Computing max query sequence length
        2. Filling KV cache if new key/value are provided
        3. Dispatching to the cache-specific stage implementation

        Prefill: FA3 when available; otherwise the Triton MLA kernel.
        Decode: paged FlashMLA.

        Args:
            query: Query tensor.
            key: Key tensor (None for decoding-only).
            value: Value tensor (None for decoding-only).
            k_cache: Key cache tensor.
            v_cache: Value cache tensor.
            attn_metadata: Attention metadata containing stage info and indices.
            k_scales_zeros: Key quantization scales/zeros.
            v_scales_zeros: Value quantization scales/zeros.
            nsa_indices: Must be ``None`` for dense MLA.

        Returns:
            Attention output tensor.
        """
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
