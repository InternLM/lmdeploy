# Copyright (c) OpenMMLab. All rights reserved.

from dataclasses import dataclass, field

import torch

from lmdeploy.pytorch.backends.attention import V4AttentionMetadata
from lmdeploy.pytorch.kernels.cuda.v4_sparse_indices import (
    build_decode_compressed_sparse_indices,
    build_decode_prefix_compressed_sparse_indices,
    build_decode_window_sparse_indices,
    build_prefill_sparse_indices,
)


@dataclass
class _V4DecodeWindowMeta:
    """Layer-invariant decode metadata for the local FP8 window cache."""

    is_padded: torch.Tensor = None
    window_pos: torch.Tensor = None
    indices: torch.Tensor = None
    topk_length: torch.Tensor = None
    disabled_indices: torch.Tensor = None
    disabled_topk_length: torch.Tensor = None


@dataclass
class _V4DecodeCompressMeta:
    """Layer-invariant decode fallback for compressed KV attention."""

    indices: torch.Tensor = None
    topk_length: torch.Tensor = None


@dataclass
class _V4PrefillWindowMeta:
    """Layer-invariant prefill metadata for local-window attention and
    writes."""

    slot: torch.Tensor = None
    ring_pos: torch.Tensor = None


@dataclass
class _V4PrefillSharedMeta:
    """Prefill tensors shared by r4/r128 compressed attention paths."""

    token_seq: torch.Tensor = None
    token_pos: torch.Tensor = None
    uncompressed_kv_lens: torch.Tensor = None


@dataclass
class _V4PrefillRatioMeta:
    """Prefill metadata for one compression ratio."""

    max_flat_kv_len: int = None
    total_flat_kv_tokens: int = None
    compress_width: int = 0
    flat_kv_lens: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None


@dataclass
class _V4RatioMeta:
    """Metadata and cached scheduler state keyed by compression ratio."""

    decode: _V4DecodeCompressMeta = None
    prefill: _V4PrefillRatioMeta = None
    flash_mla_sched_meta: object = None


@dataclass
class CudaV4AttentionMetadata(V4AttentionMetadata):
    """CUDA-specific V4 attention metadata with pre-computed indices.

    Adds layer-invariant index tensors that are computed once per step
    in ``from_step_context``, so TritonV4AttentionImpl does not
    recompute them per layer.
    """

    decode_window: _V4DecodeWindowMeta = None
    prefill_window: _V4PrefillWindowMeta = None
    prefill_shared: _V4PrefillSharedMeta = None
    ratio_meta: dict[int, _V4RatioMeta] = field(default_factory=dict)

    def get_ratio_meta(self, ratio: int) -> _V4RatioMeta:
        ratio_meta = self.ratio_meta.get(ratio)
        if ratio_meta is None:
            ratio_meta = _V4RatioMeta()
            self.ratio_meta[ratio] = ratio_meta
        return ratio_meta

    @classmethod
    def from_step_context(cls, attn_metadata, step_ctx, **kwargs) -> 'CudaV4AttentionMetadata':
        window_size = kwargs.get('window_size', 0)
        slot = kwargs.get('slot', None)

        meta = super().from_step_context(attn_metadata, step_ctx)

        if window_size > 0 and slot is not None:
            if meta.is_decoding:
                cls._precompute_decode(meta, window_size, slot < 0)
            else:
                cls._precompute_prefill(meta, window_size, slot)

        return meta

    @staticmethod
    def _precompute_decode(meta, window_size, is_padded):
        kv_seqlens = meta.kv_seqlens
        block_offsets = meta.block_offsets
        block_size = meta.block_size

        window_indices, window_topk_length, window_pos, disabled_indices, disabled_topk_length = (
            build_decode_window_sparse_indices(
                kv_seqlens, meta.start_pos, is_padded, window_size)
        )
        meta.decode_window = _V4DecodeWindowMeta(
            is_padded=is_padded,
            window_pos=window_pos,
            indices=window_indices,
            topk_length=window_topk_length,
            disabled_indices=disabled_indices,
            disabled_topk_length=disabled_topk_length,
        )

        for ratio in (4, 128):
            num_compressed = torch.div(kv_seqlens, ratio, rounding_mode='floor').to(torch.int32)
            indices = None
            if ratio == 128:
                max_comp = max(block_offsets.size(1) * block_size // ratio, 1)
                indices = build_decode_prefix_compressed_sparse_indices(
                    num_compressed, block_offsets, block_size, ratio, max_topk=max_comp)
            meta.get_ratio_meta(ratio).decode = _V4DecodeCompressMeta(
                indices=indices,
                topk_length=num_compressed,
            )

    @staticmethod
    def _precompute_prefill(meta, window_size, slot):
        from lmdeploy.pytorch.backends.cuda.attention.v4_utils import (
            build_prefill_token_meta,
        )
        kv_seqlens = meta.kv_seqlens
        q_seqlens = meta.q_seqlens
        start_pos = meta.start_pos
        total_lens = kv_seqlens
        max_kv = meta.max_kv_seqlen
        sum_kv = meta.sum_kv_seqlen

        # Uncompressed region = prev_window (ring buffer) + raw_kv (current chunk)
        # prev_window_len = min(start_pos, window_size), raw_kv_len = q_seqlens
        prev_window_lens = start_pos.clamp(max=window_size)
        uncompressed_kv_lens = (prev_window_lens + q_seqlens)

        # Safe upper bounds (no CUDA sync): max_unkv <= window_size + max_q,
        # sum_unkv <= sum(kv_seqlens) = sum_kv (since min(sp,ws) <= sp)
        max_q = meta.max_q_seqlen
        max_unkv = min(window_size, max_kv) + max_q
        cu_q_seqlens = meta.cu_q_seqlens
        token_meta = build_prefill_token_meta(q_seqlens, cu_q_seqlens)

        # Pre-compute layer-invariant tensors to eliminate per-layer gather/scatter setup.
        meta.prefill_shared = _V4PrefillSharedMeta(
            token_seq=token_meta.seq_id,
            token_pos=token_meta.token_pos,
            uncompressed_kv_lens=uncompressed_kv_lens,
        )

        # Pre-compute flat_kv_lens + cu_seqlens_k per compress_ratio.
        raw_kv_lens_t = q_seqlens.long()
        flat_kv_lens = (prev_window_lens + raw_kv_lens_t).to(torch.int32)
        cu_seqlens_k = torch.zeros(kv_seqlens.numel() + 1, dtype=torch.int32, device=kv_seqlens.device)
        torch.cumsum(flat_kv_lens, dim=0, out=cu_seqlens_k[1:])
        meta.get_ratio_meta(0).prefill = _V4PrefillRatioMeta(
            max_flat_kv_len=max_unkv,
            total_flat_kv_tokens=sum_kv,
            flat_kv_lens=flat_kv_lens,
            cu_seqlens_k=cu_seqlens_k,
        )

        for ratio in (4, 128):
            num_compressed = torch.div(kv_seqlens, ratio, rounding_mode='floor').long()
            flat_kv_lens = (prev_window_lens + raw_kv_lens_t + num_compressed).to(torch.int32)
            cu_seqlens_k = torch.zeros(kv_seqlens.numel() + 1, dtype=torch.int32, device=kv_seqlens.device)
            torch.cumsum(flat_kv_lens, dim=0, out=cu_seqlens_k[1:])
            meta.get_ratio_meta(ratio).prefill = _V4PrefillRatioMeta(
                max_flat_kv_len=max_unkv + max_kv // ratio,
                total_flat_kv_tokens=sum_kv + sum_kv // ratio,
                compress_width=max_kv // ratio,
                flat_kv_lens=flat_kv_lens,
                cu_seqlens_k=cu_seqlens_k,
            )

        # Pre-compute window write indices (reused across all layers)
        token_seq = token_meta.seq_id
        token_slot = slot[token_seq]
        token_pos_in_seq = token_meta.token_pos
        token_abs_pos = start_pos[token_seq] + token_pos_in_seq
        cutoff_pos = (total_lens[token_seq] - window_size).clamp(min=0)
        ring_pos = torch.remainder(token_abs_pos, window_size)
        invalid = token_abs_pos < cutoff_pos
        meta.prefill_window = _V4PrefillWindowMeta(slot=token_slot.clamp(min=0))
        meta.prefill_window.ring_pos = torch.where(invalid, -1, ring_pos)

class _V4AttentionExecutorBase:
    """Shared state and helpers for V4 FlashMLA execution paths."""

    def __init__(self, impl: 'TritonV4AttentionImpl'):
        self.impl = impl

    @property
    def head_size(self):
        return self.impl.head_size

    @property
    def scale(self):
        return self.impl.scale

    @property
    def window_size(self):
        return self.impl.window_size

    @property
    def compress_ratio(self):
        return self.impl.compress_ratio

    @property
    def flash_mla(self):
        return self.impl.flash_mla

    @property
    def _flatten_v4_kv(self):
        return self.impl._flatten_v4_kv

    @property
    def _pack_window_fp8(self):
        return self.impl._pack_window_fp8

    @staticmethod
    def _pad_query_heads(query: torch.Tensor, attn_sink: torch.Tensor):
        num_heads = query.size(2) if query.dim() == 4 else query.size(1)
        if num_heads in (64, 128):
            return query, attn_sink, num_heads
        if num_heads < 64:
            padded_heads = 64
        elif num_heads < 128:
            padded_heads = 128
        else:
            raise RuntimeError(f'Unsupported h_q for FlashMLA sparse decode: {num_heads}')

        # torch.empty + copy_ valid heads: skips zero-fill of padded heads since
        # FlashMLA computes per-head attention independently (MQA, h_kv=1) and
        # padded-head output is discarded by the caller's slice anyway.
        # This halves the data volume vs F.pad for TP>1 (e.g. TP=4: 16→64 heads).
        if query.dim() == 4:
            padded_q = torch.empty(query.size(0), query.size(1), padded_heads, query.size(3),
                                   dtype=query.dtype, device=query.device)
            padded_q[:, :, :num_heads, :].copy_(query)
        else:
            padded_q = torch.empty(query.size(0), padded_heads, query.size(2),
                                   dtype=query.dtype, device=query.device)
            padded_q[:, :num_heads, :].copy_(query)
        padded_s = torch.empty(padded_heads, dtype=attn_sink.dtype, device=attn_sink.device)
        padded_s[:num_heads].copy_(attn_sink)
        return padded_q, padded_s, num_heads


class _V4DecodeExecutor(_V4AttentionExecutorBase):
    """Decode-time sparse FlashMLA path."""

    def _write_window(self, kv, attn_caches, slot, attn_metadata):
        """Write decode KV to FP8 window cache."""
        window_pos = attn_metadata.decode_window.window_pos
        slot_idx = slot.clamp(min=0)
        kv_decode = kv[:, 0]  # [bsz, head_dim]
        self._pack_window_fp8(kv_decode, attn_caches['window_state_fp8'], slot_idx, window_pos)
        return attn_caches['window_state_fp8'].index_select(0, slot_idx)

    def forward(self, query, kv, attn_sink, attn_metadata: CudaV4AttentionMetadata,
                caches, slot, index_out=None):
        # Model sends [1, bsz, n_heads, head_dim] for decode; FlashMLA expects [bsz, 1, ...]
        if query.size(0) == 1 and query.size(1) > 1:
            query = query.transpose(0, 1).contiguous()
            kv = kv.transpose(0, 1).contiguous()

        total_lens = attn_metadata.kv_seqlens
        bsz = total_lens.numel()
        block_offsets = attn_metadata.block_offsets
        block_size = attn_metadata.block_size

        # Phase 1: Write window state + FP8 pack
        window_state_fp8 = self._write_window(kv, caches, slot, attn_metadata)

        # Phase 2: Window indices (pre-computed once per step)
        decode_window = attn_metadata.decode_window
        extra_indices_in_kvcache = decode_window.indices
        extra_topk_length = decode_window.topk_length
        is_padded = decode_window.is_padded

        # Phase 3: Compressed indices (indexer / fallback / no-compress)
        indices_in_kvcache = None
        topk_length = None
        fallback_topk_length = None
        compressed_cache_fp8 = None

        if self.compress_ratio:
            compressed_cache_fp8 = caches['compressed_kv_fp8']
            if index_out is not None:
                indices_in_kvcache = index_out.indices_in_kvcache.unsqueeze(1)  # [bsz, 1, topk_width]
                topk_length = index_out.topk_length
            else:
                decode_ratio = attn_metadata.get_ratio_meta(self.compress_ratio).decode
                indices_in_kvcache = decode_ratio.indices
                topk_length = decode_ratio.topk_length
                fallback_topk_length = decode_ratio.topk_length
        else:
            # No compression: pre-padded -1 sentinel + zero topk_length disables compressed path.
            indices_in_kvcache = decode_window.disabled_indices
            topk_length = decode_window.disabled_topk_length

        # Phase 4: Apply is_padded correction + convert to physical indices
        # Padded sequences (slot < 0) attend to only 1 KV entry to avoid OOB access.
        extra_k_cache = window_state_fp8.view(bsz, self.window_size, 1, -1)
        if self.compress_ratio:
            topk_length = torch.where(is_padded, 1, topk_length)
        extra_indices = extra_indices_in_kvcache
        # Compressed indices: fused logical→physical conversion plus FlashMLA padding.
        if index_out is not None:
            indices_in_kvcache = build_decode_compressed_sparse_indices(
                indices_in_kvcache, block_offsets, block_size, self.compress_ratio)
        elif self.compress_ratio and indices_in_kvcache is None:
            indices_in_kvcache = build_decode_prefix_compressed_sparse_indices(
                fallback_topk_length, block_offsets, block_size, self.compress_ratio)

        # Phase 5: FlashMLA sparse decode
        padded_indices = indices_in_kvcache
        padded_extra_indices = extra_indices
        padded_query, padded_sink, original_heads = self._pad_query_heads(query, attn_sink)
        # When compression is active, primary k_cache = compressed FP8;
        # when no compression, window cache serves as primary.
        k_cache = compressed_cache_fp8.unsqueeze(2) if compressed_cache_fp8 is not None else extra_k_cache

        ratio_meta = attn_metadata.get_ratio_meta(self.compress_ratio)
        sched_meta = ratio_meta.flash_mla_sched_meta
        if sched_meta is None:
            sched_meta, _ = self.flash_mla.get_mla_metadata()
            ratio_meta.flash_mla_sched_meta = sched_meta

        output, _ = self.flash_mla.flash_mla_with_kvcache(
            padded_query,
            k_cache=k_cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=self.head_size,
            tile_scheduler_metadata=sched_meta,
            softmax_scale=self.scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=padded_indices,
            attn_sink=padded_sink,
            extra_k_cache=extra_k_cache,
            extra_indices_in_kvcache=padded_extra_indices,
            topk_length=topk_length,
            extra_topk_length=extra_topk_length,
        )
        output = output[:, :, :original_heads]
        # Transpose back to [1, bsz, ...] to match model's expected layout
        if output.size(1) == 1 and output.size(0) > 1:
            output = output.transpose(0, 1).contiguous()
        return output


class _V4PrefillExecutor(_V4AttentionExecutorBase):
    """Prefill-time flatten, sparse index assembly, and FlashMLA path."""

    def _write_window(self, kv, attn_caches, attn_metadata):
        """Batched ring-buffer FP8 pack for all prefill sequences."""
        kv_flat = kv.squeeze(0)  # [total_tokens, head_dim]
        prefill_window = attn_metadata.prefill_window
        self._pack_window_fp8(kv_flat, attn_caches['window_state_fp8'],
                              prefill_window.slot, prefill_window.ring_pos)

    def _select_compress_topk(self, index_out, attn_metadata: CudaV4AttentionMetadata):
        """Select indexer topk indices and the fallback compressed width.

        Returns (compress_topk, compress_width):
          - compress_topk: [total_q_tokens, topk_width] from indexer output,
            or None for full-prefix fallback.
          - compress_width: number of compressed columns the fused prefill
            index builder should emit.
        """
        if not self.compress_ratio:
            return None, 0

        prefill_ratio = attn_metadata.get_ratio_meta(self.compress_ratio).prefill
        if index_out is not None:
            return index_out.indices_in_kvcache, index_out.indices_in_kvcache.size(1)

        return None, prefill_ratio.compress_width

    def forward(self, query, kv, attn_sink, attn_metadata: CudaV4AttentionMetadata,
                caches, slot, index_out=None):
        start_pos = attn_metadata.start_pos
        total_lens = attn_metadata.kv_seqlens
        q_seqlens = attn_metadata.q_seqlens
        block_offsets = attn_metadata.block_offsets

        # Phase 1: Select indexer topk source or full-prefix fallback.
        compress_topk, compress_width = self._select_compress_topk(index_out, attn_metadata)

        # Phase 2: Flatten KV (before window write to preserve prev ring buffer)
        prefill_ratio = attn_metadata.get_ratio_meta(self.compress_ratio).prefill
        max_flat_kv_len = prefill_ratio.max_flat_kv_len
        total_flat_kv_tokens = prefill_ratio.total_flat_kv_tokens
        cu_seqlens_k = prefill_ratio.cu_seqlens_k
        flat_kv_lens = prefill_ratio.flat_kv_lens

        fp8_compressed_kv_cache = caches['compressed_kv_fp8'] if self.compress_ratio else None
        raw_kv = kv.squeeze(0)  # [total_q_tokens, head_dim]
        flat_kv, cu_seqlens_k = self._flatten_v4_kv(
            caches['window_state_fp8'],
            block_offsets, total_lens,
            self.window_size, self.compress_ratio,
            total_flat_kv_tokens, max_flat_kv_len,
            cu_seqlens_k=cu_seqlens_k,
            flat_kv_lens=flat_kv_lens,
            cu_q_seqlens=attn_metadata.cu_q_seqlens,
            fp8_compressed_kv_cache=fp8_compressed_kv_cache, slot=slot,
            raw_kv=raw_kv, raw_kv_lens=q_seqlens,
            start_pos=start_pos)

        # Phase 3: Write window state + FP8 pack
        self._write_window(kv, caches, attn_metadata)

        # Phase 4: Build FlashMLA-ready sparse indices directly.
        prefill_shared = attn_metadata.prefill_shared
        topk_indices, topk_length = build_prefill_sparse_indices(
            start_pos=start_pos,
            total_lens=total_lens,
            token_seq=prefill_shared.token_seq,
            token_pos=prefill_shared.token_pos,
            cu_seqlens_k=cu_seqlens_k,
            uncompressed_kv_lens=prefill_shared.uncompressed_kv_lens,
            window_size=self.window_size,
            compress_ratio=self.compress_ratio,
            compress_topk=compress_topk,
            compress_width=compress_width)

        # Phase 5: FlashMLA sparse prefill
        q_flat, attn_sink, num_heads = self._pad_query_heads(query.squeeze(0), attn_sink)

        out = self.flash_mla.flash_mla_sparse_fwd(
            q_flat, flat_kv, topk_indices,
            sm_scale=self.scale, attn_sink=attn_sink,
            topk_length=topk_length)
        return out[0][:, :num_heads].unsqueeze(0)  # [1, total_q_tokens, n_heads, head_dim]


class TritonV4AttentionImpl:
    """DeepSeek V4 attention using batched FlashMLA sparse decode/prefill.

    The model layer calls this backend once per layer. This class owns kernel imports and dispatch; decode/prefill
    execution details stay in local executor classes.
    """

    def __init__(self, head_size: int, scale: float, window_size: int, compress_ratio: int):
        self.head_size = head_size
        self.scale = scale
        self.window_size = window_size
        self.compress_ratio = compress_ratio
        import flash_mla
        self.flash_mla = flash_mla
        from lmdeploy.pytorch.kernels.cuda.v4_flatten_kv import flatten_v4_kv
        from lmdeploy.pytorch.kernels.cuda.v4_pack_window import pack_window_tokens_fp8
        self._flatten_v4_kv = flatten_v4_kv
        self._pack_window_fp8 = pack_window_tokens_fp8
        self._decode_executor = _V4DecodeExecutor(self)
        self._prefill_executor = _V4PrefillExecutor(self)

    def forward(self, query, kv, attn_sink, attn_metadata: CudaV4AttentionMetadata,
                caches, slot, index_out=None):
        """Dispatch V4 attention to decode or prefill execution."""
        if attn_metadata.is_decoding:
            return self._decode_executor.forward(
                query, kv, attn_sink, attn_metadata, caches, slot,
                index_out=index_out)
        return self._prefill_executor.forward(
            query, kv, attn_sink, attn_metadata, caches, slot,
            index_out=index_out)


class TritonV4AttentionBuilder:
    """Builder for DeepSeek V4 sparse attention."""

    @staticmethod
    def build(head_size: int, scale: float, window_size: int, compress_ratio: int,
              **kwargs) -> TritonV4AttentionImpl:
        return TritonV4AttentionImpl(head_size=head_size,
                                     scale=scale,
                                     window_size=window_size,
                                     compress_ratio=compress_ratio)
