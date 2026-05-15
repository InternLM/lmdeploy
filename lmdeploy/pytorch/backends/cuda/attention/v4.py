# Copyright (c) OpenMMLab. All rights reserved.

import functools
from dataclasses import dataclass

import torch

from lmdeploy.pytorch.backends.attention import V4AttentionMetadata


@torch.compile(dynamic=True)
def _decode_padded_and_offset(is_padded, extra_topk_length, topk_length,
                              extra_indices_in_kvcache, batch_offsets):
    extra_topk_length = torch.where(is_padded, 1, extra_topk_length)
    topk_length = torch.where(is_padded, 1, topk_length)
    extra_indices = torch.where(extra_indices_in_kvcache >= 0,
                                extra_indices_in_kvcache + batch_offsets,
                                extra_indices_in_kvcache)
    return extra_topk_length, topk_length, extra_indices


@dataclass
class CudaV4AttentionMetadata(V4AttentionMetadata):
    """CUDA-specific V4 attention metadata with pre-computed indices.

    Adds layer-invariant index tensors that are computed once per step
    in ``from_step_context``, so TritonV4AttentionImpl does not
    recompute them per layer.
    """

    # --- Decode pre-computed (all int32 to eliminate per-layer .to(torch.int32)) ---
    is_padded: torch.Tensor = None                      # [bsz] bool
    batch_offsets: torch.Tensor = None                  # [bsz, 1, 1] int32
    decode_window_pos: torch.Tensor = None              # [bsz] int32 (pre-computed remainder)
    extra_indices_in_kvcache: torch.Tensor = None       # [bsz, 1, window_size] int32
    extra_topk_length: torch.Tensor = None              # [bsz] int32
    compress_fallback_indices_r4: torch.Tensor = None   # [bsz, 1, max_comp] int32
    compress_fallback_topk_r4: torch.Tensor = None      # [bsz] int32
    compress_fallback_indices_r128: torch.Tensor = None  # [bsz, 1, max_comp] int32
    compress_fallback_topk_r128: torch.Tensor = None     # [bsz] int32

    # --- FlashMLA schedule meta (per compress_ratio, computed once, reused across layers) ---
    flash_mla_sched_meta_r4: object = None
    flash_mla_sched_meta_r128: object = None
    flash_mla_sched_meta_r0: object = None

    # --- Prefill pre-computed ---
    prefill_uncompressed_kv_lens: torch.Tensor = None   # [bsz] long (prev_window + raw_kv)
    prefill_max_flat_kv_len_r4: int = None
    prefill_total_flat_kv_tokens_r4: int = None
    prefill_max_flat_kv_len_r128: int = None
    prefill_total_flat_kv_tokens_r128: int = None
    prefill_max_compress_width: int = None
    prefill_window_topk: torch.Tensor = None            # [total_q_tokens, window_size]
    prefill_compress_topk_r4: torch.Tensor = None       # [total_q_tokens, max_width]
    prefill_compress_topk_r128: torch.Tensor = None     # [total_q_tokens, max_width]
    prefill_num_vis_compress_r4: torch.Tensor = None    # [total_q_tokens] int32
    prefill_num_vis_compress_r128: torch.Tensor = None  # [total_q_tokens] int32

    # --- Prefill pre-computed (layer-invariant, eliminates per-layer repeat_interleave/searchsorted) ---
    prefill_seq_id: torch.Tensor = None                 # [total_q_tokens] int64
    prefill_compress_offset: torch.Tensor = None         # [total_q_tokens, 1] (uncompressed_kv_lens[seq_id])
    prefill_flat_kv_lens_r4: torch.Tensor = None         # [bsz] int32
    prefill_cu_seqlens_k_r4: torch.Tensor = None         # [bsz+1] int32
    prefill_repeat_cu_r4: torch.Tensor = None            # [total_q_tokens] int32
    prefill_flat_kv_lens_r128: torch.Tensor = None       # [bsz] int32
    prefill_cu_seqlens_k_r128: torch.Tensor = None       # [bsz+1] int32
    prefill_repeat_cu_r128: torch.Tensor = None          # [total_q_tokens] int32

    # --- Prefill window write indices (pre-computed, reused across all layers) ---
    prefill_window_slot: torch.Tensor = None              # [total_q_tokens] int64
    prefill_window_ring_pos: torch.Tensor = None          # [total_q_tokens] int64 (-1 for invalid)

    @classmethod
    def from_step_context(cls, attn_metadata, step_ctx, **kwargs) -> 'CudaV4AttentionMetadata':
        window_size = kwargs.get('window_size', 0)
        slot = kwargs.get('slot', None)

        meta = super().from_step_context(attn_metadata, step_ctx)

        if window_size > 0 and slot is not None:
            meta.is_padded = slot < 0
            if meta.is_decoding:
                cls._precompute_decode(meta, window_size)
            else:
                cls._precompute_prefill(meta, window_size, slot)

        return meta

    @staticmethod
    def _precompute_decode(meta, window_size):
        from lmdeploy.pytorch.backends.cuda.attention.v4_utils import build_prefix_positions, build_window_positions
        kv_seqlens = meta.kv_seqlens
        block_offsets = meta.block_offsets
        block_size = meta.block_size

        # Pre-compute decode window_pos (eliminates per-layer torch.remainder)
        meta.decode_window_pos = torch.remainder(meta.start_pos, window_size).to(torch.int32)

        window_positions, window_lens, _ = build_window_positions(kv_seqlens, window_size)
        meta.extra_indices_in_kvcache = window_positions.unsqueeze(1).to(torch.int32)
        meta.extra_topk_length = window_lens.to(torch.int32)

        bsz = kv_seqlens.numel()
        meta.batch_offsets = (
            torch.arange(bsz, device=kv_seqlens.device, dtype=torch.int32).view(-1, 1, 1) * window_size)

        for ratio in (4, 128):
            num_compressed = torch.div(kv_seqlens, ratio, rounding_mode='floor').to(torch.int32)
            max_comp = max(block_offsets.size(1) * block_size // ratio, 1)
            comp_positions, _ = build_prefix_positions(num_compressed, max_comp)
            indices = comp_positions.unsqueeze(1).to(torch.int32)
            if ratio == 4:
                meta.compress_fallback_indices_r4 = indices
                meta.compress_fallback_topk_r4 = num_compressed
            else:
                meta.compress_fallback_indices_r128 = indices
                meta.compress_fallback_topk_r128 = num_compressed

    @staticmethod
    def _precompute_prefill(meta, window_size, slot):
        from lmdeploy.pytorch.backends.cuda.attention.v4_utils import (
            build_compress_topk_indices,
            build_window_topk_indices,
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
        meta.prefill_uncompressed_kv_lens = (prev_window_lens + q_seqlens)

        # Safe upper bounds (no CUDA sync): max_unkv <= window_size + max_q,
        # sum_unkv <= sum(kv_seqlens) = sum_kv (since min(sp,ws) <= sp)
        max_q = meta.max_q_seqlen
        max_unkv = min(window_size, max_kv) + max_q

        for ratio in (4, 128):
            mfk = max_unkv + max_kv // ratio
            tfk = sum_kv + sum_kv // ratio
            if ratio == 4:
                meta.prefill_max_flat_kv_len_r4 = mfk
                meta.prefill_total_flat_kv_tokens_r4 = tfk
            else:
                meta.prefill_max_flat_kv_len_r128 = mfk
                meta.prefill_total_flat_kv_tokens_r128 = tfk

        meta.prefill_max_compress_width = max_kv // 4

        meta.prefill_window_topk, _ = build_window_topk_indices(
            total_lens, window_size,
            q_seqlens=q_seqlens, start_pos=start_pos, causal=True)
        meta.prefill_window_topk = meta.prefill_window_topk.to(torch.int32)

        for ratio in (4, 128):
            max_width = max_kv // ratio
            compress_topk, num_vis_r = build_compress_topk_indices(
                total_lens, ratio,
                offset=meta.prefill_uncompressed_kv_lens,
                q_seqlens=q_seqlens, start_pos=start_pos,
                causal=True, max_width=max_width)
            if ratio == 4:
                meta.prefill_compress_topk_r4 = compress_topk.to(torch.int32)
                meta.prefill_num_vis_compress_r4 = num_vis_r.to(torch.int32)
            else:
                meta.prefill_compress_topk_r128 = compress_topk.to(torch.int32)
                meta.prefill_num_vis_compress_r128 = num_vis_r.to(torch.int32)

        # Pre-compute layer-invariant tensors to eliminate per-layer repeat_interleave/searchsorted
        cu_q_seqlens = meta.cu_q_seqlens
        total_q_tokens = cu_q_seqlens[-1]
        token_seq = torch.arange(total_q_tokens, device=kv_seqlens.device)
        meta.prefill_seq_id = torch.searchsorted(cu_q_seqlens[1:], token_seq, right=True)
        meta.prefill_compress_offset = (
            meta.prefill_uncompressed_kv_lens[meta.prefill_seq_id].unsqueeze(-1).to(torch.int32))

        # Pre-compute flat_kv_lens + cu_seqlens_k + repeat_cu per compress_ratio
        raw_kv_lens_t = q_seqlens.long()
        for ratio in (4, 128):
            num_compressed = torch.div(kv_seqlens, ratio, rounding_mode='floor').long()
            flat_kv_lens = (prev_window_lens + raw_kv_lens_t + num_compressed).to(torch.int32)
            cu_seqlens_k = torch.zeros(kv_seqlens.numel() + 1, dtype=torch.int32, device=kv_seqlens.device)
            torch.cumsum(flat_kv_lens, dim=0, out=cu_seqlens_k[1:])
            repeat_cu = torch.repeat_interleave(
                cu_seqlens_k[:-1], q_seqlens, output_size=total_q_tokens)
            if ratio == 4:
                meta.prefill_flat_kv_lens_r4 = flat_kv_lens
                meta.prefill_cu_seqlens_k_r4 = cu_seqlens_k
                meta.prefill_repeat_cu_r4 = repeat_cu
            else:
                meta.prefill_flat_kv_lens_r128 = flat_kv_lens
                meta.prefill_cu_seqlens_k_r128 = cu_seqlens_k
                meta.prefill_repeat_cu_r128 = repeat_cu

        # Pre-compute window write indices (reused across all layers)
        token_seq = meta.prefill_seq_id
        cu_q = cu_q_seqlens
        token_slot = slot[token_seq]
        token_pos_in_seq = torch.arange(total_q_tokens, device=kv_seqlens.device) - cu_q[token_seq]
        token_abs_pos = start_pos[token_seq] + token_pos_in_seq
        cutoff_pos = (total_lens[token_seq] - window_size).clamp(min=0)
        ring_pos = torch.remainder(token_abs_pos, window_size)
        invalid = token_abs_pos < cutoff_pos
        meta.prefill_window_slot = token_slot.clamp(min=0)
        meta.prefill_window_ring_pos = torch.where(invalid, -1, ring_pos)


class V4IndicesUpdater:
    """V4 indices updater.

    Converts logical compressed KV positions to physical paged cache indices for decode, mirroring NSAIndicesUpdater in
    mla.py. Also converts per-seq-local topk to global flat indices for prefill.
    """

    def __init__(self):
        self._update_decode_func = None
        self._update_prefill_func = None

    def _update_decode_impl(self, logical_topk: torch.Tensor, block_offsets: torch.Tensor,
                            block_size: int, compress_ratio: int) -> torch.Tensor:
        """Convert logical compressed positions to physical KV cache
        indices."""
        bsz = logical_topk.size(0)
        safe_logical_topk = logical_topk.clamp(min=0)
        token_positions = safe_logical_topk * compress_ratio
        block_idx = torch.div(token_positions, block_size, rounding_mode='floor')
        max_block_idx = block_offsets.size(1)
        safe_block_idx = block_idx.clamp(max=max_block_idx - 1)
        block_idx_valid = block_idx < max_block_idx
        phys_block = block_offsets.gather(1, safe_block_idx.view(bsz, -1)).view_as(logical_topk)
        entries_per_block = block_size // compress_ratio
        block_off = torch.remainder(safe_logical_topk, entries_per_block)
        phys_indices = phys_block * entries_per_block + block_off
        valid = (logical_topk >= 0) & block_idx_valid
        return torch.where(valid, phys_indices, phys_indices.new_full((), -1))

    def update_decode(self, logical_topk: torch.Tensor, block_offsets: torch.Tensor,
                      block_size: int, compress_ratio: int) -> torch.Tensor:
        if self._update_decode_func is None:
            self._update_decode_func = _try_dynamic_compile(
                self._update_decode_impl, logical_topk, block_offsets, block_size, compress_ratio)
        return self._update_decode_func(logical_topk, block_offsets, block_size, compress_ratio)

    def _update_prefill_impl(self, topk_indices: torch.Tensor, q_seqlens: torch.Tensor,
                             cu_seqlens_k: torch.Tensor) -> torch.Tensor:
        """Convert per-seq-local topk to global flat indices for prefill.

        Mirrors NSAIndicesUpdater.update_prefill: adds cu_seqlens_k offsets and preserves -1 padding.
        """
        num_tokens = topk_indices.size(0)
        repeat_cu = torch.repeat_interleave(cu_seqlens_k[:-1], q_seqlens, output_size=num_tokens)
        neg_mask = topk_indices < 0
        topk_indices = topk_indices + repeat_cu[:, None]
        topk_indices[neg_mask] = -1
        return topk_indices.unsqueeze(1)  # [total_q, 1, total_topk]

    def update_prefill(self, topk_indices: torch.Tensor, q_seqlens: torch.Tensor,
                       cu_seqlens_k: torch.Tensor) -> torch.Tensor:
        if self._update_prefill_func is None:
            self._update_prefill_func = _try_dynamic_compile(
                self._update_prefill_impl, topk_indices, q_seqlens, cu_seqlens_k)
        return self._update_prefill_func(topk_indices, q_seqlens, cu_seqlens_k)

    @staticmethod
    @functools.cache
    def build():
        return V4IndicesUpdater()



def _try_dynamic_compile(func, *args, **kwargs):
    try:
        compiled_func = torch.compile(func, dynamic=True)
        compiled_func(*args, **kwargs)
        return compiled_func
    except Exception:
        return func


class TritonV4AttentionImpl:
    """DeepSeek V4 attention using batched FlashMLA sparse decode/prefill.

    The unified ``forward()`` dispatches to ``_forward_decoding()`` or
    ``_forward_prefilling()`` based on ``attn_metadata.is_decoding``.
    All window-state management, kernel calls (flatten_v4_kv, pack_window_fp8),
    index construction, and FlashMLA invocations live here — the model layer
    only does projections, RoPE, and compressor/indexer calls.
    """

    def __init__(self, head_size: int, scale: float, window_size: int, compress_ratio: int):
        self.head_size = head_size
        self.scale = scale
        self.window_size = window_size
        self.compress_ratio = compress_ratio
        self.v4_updater = V4IndicesUpdater.build()
        import flash_mla
        self.flash_mla = flash_mla
        from lmdeploy.pytorch.kernels.cuda.v4_flatten_kv import flatten_v4_kv
        from lmdeploy.pytorch.kernels.cuda.v4_pack_window import pack_window_tokens_fp8
        self._flatten_v4_kv = flatten_v4_kv
        self._pack_window_fp8 = pack_window_tokens_fp8

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    @torch.compile(dynamic=True)
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

        pad_heads = padded_heads - num_heads
        query = torch.nn.functional.pad(query, (0, 0, 0, pad_heads))
        attn_sink = torch.nn.functional.pad(attn_sink, (0, pad_heads))
        return query, attn_sink, num_heads

    @staticmethod
    @torch.compile(dynamic=True)
    def _pad_sparse_indices(indices: torch.Tensor | None, block: int = 128):
        if indices is None:
            return None
        topk = indices.size(-1)
        padded_topk = ((topk + block - 1) // block) * block
        if padded_topk == topk:
            return indices
        pad = padded_topk - topk
        return torch.nn.functional.pad(indices, (0, pad), value=-1)

    def _write_window_decode(self, kv, attn_caches, slot, attn_metadata):
        """Write decode KV to FP8 window cache."""
        window_pos = attn_metadata.decode_window_pos
        slot_idx = slot.clamp(min=0)
        kv_decode = kv[:, 0]  # [bsz, head_dim]
        self._pack_window_fp8(kv_decode, attn_caches['window_state_fp8'], slot_idx, window_pos)
        return attn_caches['window_state_fp8'].index_select(0, slot_idx)

    def _write_window_prefill(self, kv, attn_caches, attn_metadata):
        """Batched ring-buffer FP8 pack for all prefill sequences."""
        kv_flat = kv.squeeze(0)  # [total_tokens, head_dim]
        self._pack_window_fp8(kv_flat, attn_caches['window_state_fp8'],
                              attn_metadata.prefill_window_slot, attn_metadata.prefill_window_ring_pos)


    # ------------------------------------------------------------------
    # Unified forward
    # ------------------------------------------------------------------

    def forward(self, query, kv, attn_sink, attn_metadata: CudaV4AttentionMetadata,
                caches, slot, index_out=None):
        """Unified forward — dispatches to decoding or prefilling internally.

        Args:
            query: Q tensor [bsz, 1, n_heads, head_dim] or [1, total_tokens, n_heads, head_dim]
            kv: KV tensor [bsz, 1, head_dim] or [1, total_tokens, head_dim]
            attn_sink: Learnable sink parameter
            attn_metadata: CudaV4AttentionMetadata with sequence info
            caches: dict of attention cache tensors
            slot: state cache slot indices [bsz]
            index_out: V4IndexerOutput from the indexer call (if any)
        """
        if attn_metadata.is_decoding:
            return self._forward_decoding(query, kv, attn_sink, attn_metadata, caches, slot,
                                          index_out=index_out)
        else:
            return self._forward_prefilling(query, kv, attn_sink, attn_metadata, caches, slot,
                                            index_out=index_out)

    # ------------------------------------------------------------------
    # Decode path
    # ------------------------------------------------------------------

    def _forward_decoding(self, query, kv, attn_sink, attn_metadata: CudaV4AttentionMetadata,
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
        window_state_fp8 = self._write_window_decode(kv, caches, slot, attn_metadata)

        # Phase 2: Window indices (pre-computed once per step)
        extra_indices_in_kvcache = attn_metadata.extra_indices_in_kvcache
        extra_topk_length = attn_metadata.extra_topk_length
        is_padded = attn_metadata.is_padded

        # Phase 3: Compressed indices (indexer / fallback / no-compress)
        indices_in_kvcache = None
        topk_length = None
        compressed_cache_fp8 = None

        if self.compress_ratio:
            compressed_cache_fp8 = caches['compressed_kv_fp8']
            if index_out is not None:
                indices_in_kvcache = index_out.indices_in_kvcache.unsqueeze(1)  # [bsz, 1, topk_width]
                topk_length = index_out.topk_length
            elif self.compress_ratio == 4:
                indices_in_kvcache = attn_metadata.compress_fallback_indices_r4
                topk_length = attn_metadata.compress_fallback_topk_r4
            else:
                indices_in_kvcache = attn_metadata.compress_fallback_indices_r128
                topk_length = attn_metadata.compress_fallback_topk_r128
        else:
            # No compression: -1 sentinel + zero topk_length disables compressed path
            indices_in_kvcache = torch.full((bsz, 1, 1), -1, dtype=torch.int32, device=query.device)
            topk_length = torch.zeros(bsz, dtype=torch.int32, device=query.device)

        # Phase 4: Apply is_padded correction + convert to physical indices
        # Padded sequences (slot < 0) attend to only 1 KV entry to avoid OOB access.
        extra_k_cache = window_state_fp8.view(bsz, self.window_size, 1, -1)
        batch_offsets = attn_metadata.batch_offsets
        extra_topk_length, topk_length, extra_indices = _decode_padded_and_offset(
            is_padded, extra_topk_length, topk_length,
            extra_indices_in_kvcache, batch_offsets)
        # Compressed indices: gather-based logical→physical (paged layout)
        # Only indexer output needs conversion; fallback indices are already physical.
        if index_out is not None:
            indices_in_kvcache = self.v4_updater.update_decode(
                indices_in_kvcache, block_offsets, block_size, self.compress_ratio)

        # Phase 5: FlashMLA sparse decode
        padded_indices = self._pad_sparse_indices(indices_in_kvcache)
        padded_extra_indices = self._pad_sparse_indices(extra_indices)
        padded_query, padded_sink, original_heads = self._pad_query_heads(query, attn_sink)
        # When compression is active, primary k_cache = compressed FP8;
        # when no compression, window cache serves as primary.
        k_cache = compressed_cache_fp8.unsqueeze(2) if compressed_cache_fp8 is not None else extra_k_cache

        ratio_key = f'flash_mla_sched_meta_r{self.compress_ratio}'
        sched_meta = getattr(attn_metadata, ratio_key)
        if sched_meta is None:
            sched_meta, _ = self.flash_mla.get_mla_metadata()
            object.__setattr__(attn_metadata, ratio_key, sched_meta)

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

    # ------------------------------------------------------------------
    # Prefill path
    # ------------------------------------------------------------------

    def _select_compress_topk(self, index_out, attn_metadata: CudaV4AttentionMetadata):
        """Select compress_topk indices and per-token causal visibility count.

        Returns (compress_topk, num_vis_compress):
          - compress_topk: [total_q_tokens, max_width] or None
          - num_vis_compress: [total_q_tokens] int32 (raw compress-only count) or None
        """
        if not self.compress_ratio:
            return None, None

        if index_out is not None:
            compress_topk = index_out.indices_in_kvcache
            # Offset indexer's logical indices into flat_kv positions
            compress_topk = compress_topk + attn_metadata.prefill_compress_offset
            # Per-token causal limit from the pre-computed ratio-specific count.
            # index_out.topk_length is per-sequence and ignores causal masking.
            if self.compress_ratio == 128:
                num_vis_compress = attn_metadata.prefill_num_vis_compress_r128
            else:
                num_vis_compress = attn_metadata.prefill_num_vis_compress_r4
            return compress_topk, num_vis_compress

        if self.compress_ratio == 4:
            return (attn_metadata.prefill_compress_topk_r4,
                    attn_metadata.prefill_num_vis_compress_r4)
        else:
            return (attn_metadata.prefill_compress_topk_r128,
                    attn_metadata.prefill_num_vis_compress_r128)

    def _forward_prefilling(self, query, kv, attn_sink, attn_metadata: CudaV4AttentionMetadata,
                            caches, slot, index_out=None):
        start_pos = attn_metadata.start_pos
        total_lens = attn_metadata.kv_seqlens
        q_seqlens = attn_metadata.q_seqlens
        block_offsets = attn_metadata.block_offsets

        # Phase 1: Select compress_topk source + num_vis_compress
        compress_topk, num_vis_compress = self._select_compress_topk(index_out, attn_metadata)

        # Phase 2: Flatten KV (before window write to preserve prev ring buffer)
        if self.compress_ratio == 4:
            max_flat_kv_len = attn_metadata.prefill_max_flat_kv_len_r4
            total_flat_kv_tokens = attn_metadata.prefill_total_flat_kv_tokens_r4
            cu_seqlens_k = attn_metadata.prefill_cu_seqlens_k_r4
            flat_kv_lens = attn_metadata.prefill_flat_kv_lens_r4
            repeat_cu = attn_metadata.prefill_repeat_cu_r4
        elif self.compress_ratio == 128:
            max_flat_kv_len = attn_metadata.prefill_max_flat_kv_len_r128
            total_flat_kv_tokens = attn_metadata.prefill_total_flat_kv_tokens_r128
            cu_seqlens_k = attn_metadata.prefill_cu_seqlens_k_r128
            flat_kv_lens = attn_metadata.prefill_flat_kv_lens_r128
            repeat_cu = attn_metadata.prefill_repeat_cu_r128
        else:
            max_flat_kv_len = attn_metadata.prefill_max_flat_kv_len_r4
            total_flat_kv_tokens = attn_metadata.prefill_total_flat_kv_tokens_r4
            cu_seqlens_k = None
            flat_kv_lens = None
            repeat_cu = None

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
        self._write_window_prefill(kv, caches, attn_metadata)

        # Phase 4: Assemble topk_indices (cat window + compress) + convert to global
        window_topk = attn_metadata.prefill_window_topk
        if compress_topk is not None:
            topk_indices = torch.cat([window_topk, compress_topk], dim=-1)
        else:
            topk_indices = window_topk
        if repeat_cu is not None:
            neg_mask = topk_indices < 0
            topk_indices = torch.where(neg_mask, -1, topk_indices + repeat_cu[:, None])
            topk_indices = topk_indices.unsqueeze(1)
        else:
            topk_indices = self.v4_updater.update_prefill(topk_indices, q_seqlens, cu_seqlens_k)
        topk_indices = self._pad_sparse_indices(topk_indices).to(torch.int32)

        # Phase 5: Compute topk_length + FlashMLA sparse prefill
        q_flat, attn_sink, num_heads = self._pad_query_heads(query.squeeze(0), attn_sink)

        topk_width = topk_indices.size(-1)
        if num_vis_compress is not None:
            topk_length = self.window_size + num_vis_compress
        else:
            topk_length = torch.full((q_flat.size(0),), self.window_size,
                                     dtype=torch.int32, device=query.device)
        topk_length = topk_length.clamp(max=topk_width)

        out = self.flash_mla.flash_mla_sparse_fwd(
            q_flat, flat_kv, topk_indices,
            sm_scale=self.scale, attn_sink=attn_sink,
            topk_length=topk_length)
        return out[0][:, :num_heads].unsqueeze(0)  # [1, total_q_tokens, n_heads, head_dim]


class TritonV4AttentionBuilder:
    """Builder for DeepSeek V4 sparse attention."""

    @staticmethod
    def build(head_size: int, scale: float, window_size: int, compress_ratio: int,
              **kwargs) -> TritonV4AttentionImpl:
        return TritonV4AttentionImpl(head_size=head_size,
                                     scale=scale,
                                     window_size=window_size,
                                     compress_ratio=compress_ratio)
