# Copyright (c) OpenMMLab. All rights reserved.

import functools

import torch

from lmdeploy.pytorch.backends.attention import V4AttentionMetadata
from lmdeploy.pytorch.models.deepseek_v4_utils import (
    build_compress_topk_indices,
    build_prefix_positions,
    build_window_positions,
    build_window_topk_indices,
)


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
        block_idx = torch.div(token_positions, block_size, rounding_mode='floor').long()
        max_block_idx = block_offsets.size(1)
        safe_block_idx = block_idx.clamp(max=max_block_idx - 1)
        block_idx_valid = block_idx < max_block_idx
        phys_block = block_offsets.gather(1, safe_block_idx.view(bsz, -1)).view_as(logical_topk).long()
        entries_per_block = block_size // compress_ratio
        block_off = torch.remainder(safe_logical_topk, entries_per_block).long()
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
        repeat_cu = torch.repeat_interleave(cu_seqlens_k[:-1], q_seqlens.long(), output_size=num_tokens)
        neg_mask = topk_indices < 0
        topk_indices = topk_indices + repeat_cu[:, None]
        topk_indices[neg_mask] = -1
        return topk_indices.unsqueeze(1).to(torch.int32)  # [total_q, 1, total_topk]

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
    """Try to compile a function with torch.compile, fall back to eager."""
    try:
        return torch.compile(func, dynamic=True)
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
        if query.dim() == 4:
            query = torch.nn.functional.pad(query, (0, 0, 0, pad_heads))
        else:
            query = torch.nn.functional.pad(query, (0, 0, 0, pad_heads))
        attn_sink = torch.nn.functional.pad(attn_sink, (0, pad_heads))
        return query, attn_sink, num_heads

    @staticmethod
    def _pad_sparse_indices(indices: torch.Tensor | None, block: int = 128):
        if indices is None:
            return None
        topk = indices.size(-1)
        padded_topk = ((topk + block - 1) // block) * block
        if padded_topk == topk:
            return indices
        pad = padded_topk - topk
        return torch.nn.functional.pad(indices, (0, pad), value=-1)

    def _write_window_decode(self, kv, attn_caches, slot, start_pos, total_lens):
        """Write decode KV to window state and pack FP8."""
        window_pos = torch.remainder(start_pos, self.window_size).long()
        slot_idx = slot.long().clamp(min=0)
        kv_decode = kv[:, 0]  # [bsz, head_dim]
        attn_caches['window_state'][slot_idx, window_pos] = kv_decode
        self._pack_window_fp8(kv_decode, attn_caches['window_state_fp8'], slot_idx, window_pos)
        return attn_caches['window_state_fp8'].index_select(0, slot_idx)

    def _write_window_prefill(self, kv, attn_caches, slot, start_pos, q_seqlens, total_lens):
        """Batched ring-buffer write for all prefill sequences + FP8 pack."""
        kv_flat = kv.squeeze(0)  # [total_tokens, head_dim]
        num_seqs = start_pos.numel()
        cu_q = torch.cat([start_pos.new_zeros(1), q_seqlens.cumsum(0)])
        total_tokens = kv_flat.size(0)

        # Build per-token indices
        token_slot = slot.repeat_interleave(q_seqlens.long())
        token_seq = torch.arange(num_seqs, device=slot.device).repeat_interleave(q_seqlens.long())
        token_pos_in_seq = torch.arange(total_tokens, device=slot.device) - cu_q[token_seq]
        token_start = start_pos.repeat_interleave(q_seqlens.long())
        token_abs_pos = token_start + token_pos_in_seq
        token_total = total_lens[token_seq]
        cutoff_pos = (token_total - self.window_size).clamp(min=0)
        valid = token_abs_pos >= cutoff_pos
        ring_pos = torch.remainder(token_abs_pos, self.window_size)

        valid_slot = token_slot[valid].long()
        valid_ring = ring_pos[valid].long()
        valid_kv = kv_flat[valid]
        attn_caches['window_state'][valid_slot, valid_ring] = valid_kv

        # Batched FP8 pack
        slot_idx = slot.long().clamp(min=0)
        selected = attn_caches['window_state'][slot_idx]
        num_seqs_local = slot.numel()
        slot_expanded = slot_idx.repeat_interleave(self.window_size)
        pos_expanded = torch.arange(self.window_size, device=slot.device).repeat(num_seqs_local).long()
        kv_tokens = selected.reshape(-1, self.head_size)
        self._pack_window_fp8(kv_tokens, attn_caches['window_state_fp8'], slot_expanded, pos_expanded)

    def _build_window_indices(self, total_lens):
        """Build window ring-buffer positions and lengths."""
        window_positions, window_lens, _ = build_window_positions(total_lens.long(), self.window_size)
        extra_indices_in_kvcache = window_positions.unsqueeze(1).to(torch.int32)
        extra_topk_length = window_lens.to(torch.int32)
        return extra_indices_in_kvcache, extra_topk_length

    # ------------------------------------------------------------------
    # Unified forward
    # ------------------------------------------------------------------

    def forward(self, query, kv, attn_sink, attn_metadata: V4AttentionMetadata,
                caches, slot, index_out=None):
        """Unified forward — dispatches to decoding or prefilling internally.

        Args:
            query: Q tensor [bsz, 1, n_heads, head_dim] or [1, total_tokens, n_heads, head_dim]
            kv: KV tensor [bsz, 1, head_dim] or [1, total_tokens, head_dim]
            attn_sink: Learnable sink parameter
            attn_metadata: V4AttentionMetadata with sequence info
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

    def _forward_decoding(self, query, kv, attn_sink, attn_metadata: V4AttentionMetadata,
                          caches, slot, index_out=None):
        # Model sends [1, bsz, n_heads, head_dim] for decode; FlashMLA expects [bsz, 1, ...]
        if query.size(0) == 1 and query.size(1) > 1:
            query = query.transpose(0, 1).contiguous()
            kv = kv.transpose(0, 1).contiguous()

        kv_seqlens = attn_metadata.kv_seqlens
        q_seqlens = attn_metadata.q_seqlens
        start_pos = (kv_seqlens.to(torch.long) - q_seqlens.to(torch.long))
        total_lens = kv_seqlens
        bsz = kv_seqlens.numel()
        block_offsets = attn_metadata.block_offsets.long()
        block_size = attn_metadata.block_size

        # Write window state + FP8 pack
        window_state_fp8 = self._write_window_decode(kv, caches, slot, start_pos, total_lens)

        # Window indices
        extra_indices_in_kvcache, extra_topk_length = self._build_window_indices(total_lens)
        is_padded = slot < 0
        extra_topk_length = torch.where(is_padded, 1, extra_topk_length)

        # Compressed indices
        indices_in_kvcache = None
        topk_length = None
        compressed_cache_fp8 = None

        if self.compress_ratio:
            compressed_cache_fp8 = caches['compressed_kv_fp8']
            if index_out is not None:
                indices_in_kvcache = index_out.indices_in_kvcache
                topk_length = index_out.topk_length
            else:
                num_compressed = torch.div(total_lens, self.compress_ratio, rounding_mode='floor').long()
                max_comp = max(block_offsets.size(1) * block_size // self.compress_ratio, 1)
                comp_positions, _ = build_prefix_positions(num_compressed, max_comp)
                indices_in_kvcache = comp_positions.unsqueeze(1).to(torch.int32)
                topk_length = num_compressed.to(torch.int32)
        else:
            indices_in_kvcache = torch.full((bsz, 1, 1), -1, dtype=torch.int32, device=query.device)
            topk_length = torch.zeros(bsz, dtype=torch.int32, device=query.device)

        topk_length = torch.where(is_padded, 1, topk_length)

        # FlashMLA sparse decode
        extra_k_cache = window_state_fp8.view(bsz, self.window_size, 1, -1)
        extra_indices = extra_indices_in_kvcache
        batch_offsets = torch.arange(bsz, device=extra_indices.device, dtype=torch.int32).view(-1, 1, 1) * self.window_size  # noqa: E501
        extra_indices = torch.where(extra_indices >= 0, extra_indices + batch_offsets, extra_indices)

        if block_offsets is not None and self.compress_ratio and indices_in_kvcache is not None:
            indices_in_kvcache = self.v4_updater.update_decode(
                indices_in_kvcache, block_offsets.long(), block_size, self.compress_ratio)

        padded_indices = self._pad_sparse_indices(indices_in_kvcache)
        padded_extra_indices = self._pad_sparse_indices(extra_indices)
        padded_query, padded_sink, original_heads = self._pad_query_heads(query, attn_sink)

        k_cache = compressed_cache_fp8.unsqueeze(2) if compressed_cache_fp8 is not None else extra_k_cache

        sched_meta, _ = self.flash_mla.get_mla_metadata()

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
            indices=padded_indices.to(torch.int32),
            attn_sink=padded_sink,
            extra_k_cache=extra_k_cache,
            extra_indices_in_kvcache=padded_extra_indices.to(torch.int32),
            topk_length=topk_length.to(torch.int32),
            extra_topk_length=extra_topk_length.to(torch.int32),
        )
        output = output[:, :, :original_heads]
        # Transpose back to [1, bsz, ...] to match model's expected layout
        if output.size(1) == 1 and output.size(0) > 1:
            output = output.transpose(0, 1).contiguous()
        return output

    # ------------------------------------------------------------------
    # Prefill path
    # ------------------------------------------------------------------

    def _forward_prefilling(self, query, kv, attn_sink, attn_metadata: V4AttentionMetadata,
                            caches, slot, index_out=None):
        kv_seqlens = attn_metadata.kv_seqlens
        q_seqlens = attn_metadata.q_seqlens
        start_pos = (kv_seqlens.to(torch.long) - q_seqlens.to(torch.long))
        total_lens = kv_seqlens
        block_offsets = attn_metadata.block_offsets

        # CPU-side upper bounds for flatten_v4_kv (avoids GPU .item() sync)
        max_kv = attn_metadata.max_kv_seqlen
        sum_kv = attn_metadata.sum_kv_seqlen
        cr = self.compress_ratio if self.compress_ratio else 1
        max_flat_kv_len = min(max_kv, self.window_size) + max_kv // cr
        total_flat_kv_tokens = sum_kv + sum_kv // cr
        max_compress_width = max_kv // cr

        # Pre-compute window_kv_lens for Indexer offset
        window_kv_lens = total_lens.clamp(max=self.window_size)

        # Write window state + FP8 pack (batched)
        self._write_window_prefill(kv, caches, slot, start_pos, q_seqlens, total_lens)

        # Build compress topk
        compress_topk = None
        if self.compress_ratio:
            if index_out is not None:
                compress_topk = index_out.indices_in_kvcache.squeeze(0)
                cu_q_seqlens = attn_metadata.cu_q_seqlens
                total_tokens = compress_topk.size(0)
                token_seq = torch.arange(total_tokens, device=compress_topk.device)
                seq_id = torch.searchsorted(cu_q_seqlens[1:], token_seq, right=True)
                compress_topk = compress_topk + window_kv_lens[seq_id].unsqueeze(-1)
            else:
                compress_topk = build_compress_topk_indices(
                    total_lens, self.compress_ratio,
                    offset=window_kv_lens,
                    q_seqlens=q_seqlens,
                    start_pos=start_pos,
                    causal=True,
                    max_width=max_compress_width)

        # Flatten window + compressed KV into contiguous tensor
        fp8_compressed_kv_cache = caches['compressed_kv_fp8'] if self.compress_ratio else None
        flat_kv, cu_seqlens_k = self._flatten_v4_kv(
            caches['window_state'], None,
            block_offsets.long(), total_lens.long(),
            self.window_size, self.compress_ratio,
            total_flat_kv_tokens, max_flat_kv_len,
            fp8_compressed_kv_cache=fp8_compressed_kv_cache, slot=slot)

        # Build topk indices
        q_flat = query.squeeze(0)
        window_topk = build_window_topk_indices(
            total_lens, self.window_size,
            q_seqlens=q_seqlens,
            start_pos=start_pos,
            causal=True)

        if compress_topk is not None:
            topk_indices = torch.cat([window_topk, compress_topk], dim=-1)
        else:
            topk_indices = window_topk

        # Convert per-seq-local indices to global flat indices
        topk_indices = self.v4_updater.update_prefill(topk_indices, q_seqlens, cu_seqlens_k)

        # FlashMLA sparse prefill
        topk_indices = self._pad_sparse_indices(topk_indices).to(torch.int32)

        num_heads = q_flat.size(1)
        target = 64 if num_heads < 64 else (128 if num_heads < 128 else num_heads)
        if target != num_heads:
            pad = target - num_heads
            q_flat = torch.nn.functional.pad(q_flat, (0, 0, 0, pad))
            attn_sink = torch.nn.functional.pad(attn_sink, (0, pad))

        out = self.flash_mla.flash_mla_sparse_fwd(
            q_flat, flat_kv, topk_indices,
            sm_scale=self.scale, attn_sink=attn_sink)
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
