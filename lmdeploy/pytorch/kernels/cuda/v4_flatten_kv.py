# Copyright (c) OpenMMLab. All rights reserved.
"""Triton kernel to flatten V4 window + compressed KV caches into a contiguous
BF16 tensor for ``flash_mla_sparse_fwd`` prefill.

When the compressed KV cache is in V4 FlashMLA sparse FP8 format, the kernel
dequantizes in-place (per-tile FP8→BF16 using e8m0fnu scales), avoiding the
large intermediate BF16 allocation that ``dequantize_v4_flashmla_sparse`` would
require.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _flatten_v4_kv_kernel(
    window_kv_ptr,
    compressed_kv_ptr,
    out_ptr,
    start_loc_ptr,
    flat_kv_lens_ptr,
    block_offsets_ptr,
    slot_ptr,
    stride_wkv_b,
    stride_wkv_s,
    stride_wkv_d,
    stride_ckv_b,
    stride_ckv_s,
    stride_ckv_d,
    stride_out_s,
    stride_out_d,
    stride_boff,
    fp8_nope_rope_ptr,
    fp8nr_stride_b,
    fp8nr_stride_s,
    fp8nr_stride_d,
    fp8_rope_bf16_ptr,
    fp8rbf16_stride_b,
    fp8rbf16_stride_s,
    fp8rbf16_stride_d,
    fp8_scales_u8_ptr,
    fp8sc_stride_b,
    fp8sc_stride_s,
    fp8sc_stride_d,
    raw_kv_ptr,
    stride_rkv_s,
    stride_rkv_d,
    cu_raw_kv_ptr,
    raw_kv_lens_ptr,
    start_pos_ptr,
    WINDOW_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_COMPRESS: tl.constexpr,
    HAS_SLOT: tl.constexpr,
    HAS_FP8_COMPRESS: tl.constexpr,
    HAS_RAW_KV: tl.constexpr,
):
    batch_id = tl.program_id(0)
    token_id = tl.program_id(1)

    flat_kv_len = tl.load(flat_kv_lens_ptr + batch_id)
    start_loc = tl.load(start_loc_ptr + batch_id)

    if token_id >= flat_kv_len:
        return

    # Ring buffer covers only the previous window (before current chunk).
    # For first-time prefill (start_pos=0), prev_window_len=0 and all tokens
    # come from raw_kv. For chunked prefill, prev_window_len=min(start_pos, WINDOW_SIZE).
    start_pos_val = tl.load(start_pos_ptr + batch_id)
    prev_window_len = tl.minimum(start_pos_val, WINDOW_SIZE)
    offs_d = tl.arange(0, HEAD_DIM)

    if token_id < prev_window_len:
        # ---- Previous window region (from ring buffer) ----
        window_start = start_pos_val - WINDOW_SIZE if start_pos_val > WINDOW_SIZE else 0
        actual_pos = window_start + token_id
        ring_pos = actual_pos % WINDOW_SIZE

        if HAS_SLOT:
            slot_val = tl.load(slot_ptr + batch_id)
            if slot_val < 0:
                val = tl.zeros((HEAD_DIM,), dtype=out_ptr.dtype.element_ty)
            else:
                src_ptr = (window_kv_ptr + slot_val.to(tl.int64) * stride_wkv_b
                           + ring_pos * stride_wkv_s
                           + offs_d * stride_wkv_d)
                val = tl.load(src_ptr, mask=actual_pos < start_pos_val, other=0.0)
        else:
            src_ptr = (window_kv_ptr + batch_id * stride_wkv_b
                       + ring_pos * stride_wkv_s
                       + offs_d * stride_wkv_d)
            val = tl.load(src_ptr, mask=actual_pos < start_pos_val, other=0.0)

        out_ptr_off = (start_loc + token_id) * stride_out_s + offs_d * stride_out_d
        tl.store(out_ptr + out_ptr_off, val)

    elif HAS_RAW_KV and token_id < prev_window_len + tl.load(raw_kv_lens_ptr + batch_id):
        # ---- Current chunk region (from raw KV input) ----
        raw_kv_len = tl.load(raw_kv_lens_ptr + batch_id)
        local_pos = token_id - prev_window_len
        cu_raw_kv_val = tl.load(cu_raw_kv_ptr + batch_id)
        src_ptr = (raw_kv_ptr + (cu_raw_kv_val + local_pos) * stride_rkv_s
                   + offs_d * stride_rkv_d)
        val = tl.load(src_ptr)

        out_ptr_off = (start_loc + token_id) * stride_out_s + offs_d * stride_out_d
        tl.store(out_ptr + out_ptr_off, val)

    elif HAS_FP8_COMPRESS:
        # ---- Compressed region: FP8 dequantize path ----
        raw_kv_len = tl.load(raw_kv_lens_ptr + batch_id) if HAS_RAW_KV else 0
        comp_pos = token_id - prev_window_len - raw_kv_len
        page_id = comp_pos // BLOCK_SIZE
        page_off = comp_pos % BLOCK_SIZE
        phys_block = tl.load(block_offsets_ptr + batch_id * stride_boff + page_id)
        phys_block_64 = phys_block.to(tl.int64)

        D_NOPE: tl.constexpr = 448
        D_ROPE: tl.constexpr = 64
        TILE_SIZE: tl.constexpr = 64
        NUM_TILES: tl.constexpr = 7

        # NoPE tiles: dequantize FP8→BF16 and write directly
        offs_tile = tl.arange(0, TILE_SIZE)
        for tile_idx in range(NUM_TILES):
            d_base = tile_idx * TILE_SIZE
            nope_ptrs = (fp8_nope_rope_ptr
                         + phys_block_64 * fp8nr_stride_b
                         + page_off * fp8nr_stride_s
                         + (d_base + offs_tile) * fp8nr_stride_d)
            nope_fp8 = tl.load(nope_ptrs)

            sc_ptr = (fp8_scales_u8_ptr
                      + phys_block_64 * fp8sc_stride_b
                      + page_off * fp8sc_stride_s
                      + tile_idx * fp8sc_stride_d)
            scale_byte = tl.load(sc_ptr).to(tl.int32)
            scale_bits = scale_byte << 23
            scale_f32 = tl.cast(scale_bits, tl.float32, bitcast=True)

            dequant = (nope_fp8.to(tl.float32) * scale_f32).to(tl.bfloat16)
            out_off_tile = ((start_loc + token_id) * stride_out_s
                            + (d_base + offs_tile) * stride_out_d)
            tl.store(out_ptr + out_off_tile, dequant)

        # RoPE: read from bf16 view, write directly
        rope_offs = tl.arange(0, D_ROPE)
        rope_ptrs = (fp8_rope_bf16_ptr
                     + phys_block_64 * fp8rbf16_stride_b
                     + page_off * fp8rbf16_stride_s
                     + rope_offs * fp8rbf16_stride_d)
        rope_val = tl.load(rope_ptrs)
        out_off_rope = ((start_loc + token_id) * stride_out_s
                        + (D_NOPE + rope_offs) * stride_out_d)
        tl.store(out_ptr + out_off_rope, rope_val)

    else:
        # ---- Compressed region: BF16 path ----
        raw_kv_len = tl.load(raw_kv_lens_ptr + batch_id) if HAS_RAW_KV else 0
        if not HAS_COMPRESS:
            val = tl.zeros((HEAD_DIM,), dtype=out_ptr.dtype.element_ty)
        else:
            comp_pos = token_id - prev_window_len - raw_kv_len
            page_id = comp_pos // BLOCK_SIZE
            page_off = comp_pos % BLOCK_SIZE
            phys_block = tl.load(block_offsets_ptr + batch_id * stride_boff + page_id)
            src_ptr = (compressed_kv_ptr + phys_block.to(tl.int64) * stride_ckv_b
                       + page_off * stride_ckv_s
                       + offs_d * stride_ckv_d)
            val = tl.load(src_ptr)

        out_ptr_off = (start_loc + token_id) * stride_out_s + offs_d * stride_out_d
        tl.store(out_ptr + out_ptr_off, val)


def flatten_v4_kv(
    window_kv_cache: torch.Tensor,
    compressed_kv_cache: torch.Tensor | None,
    block_offsets: torch.Tensor,
    kv_seqlens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    total_kv_tokens: int,
    max_flat_kv_len: int,
    cu_seqlens_k: torch.Tensor | None = None,
    fp8_compressed_kv_cache: torch.Tensor | None = None,
    slot: torch.Tensor | None = None,
    raw_kv: torch.Tensor | None = None,
    raw_kv_lens: torch.Tensor | None = None,
    start_pos: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten V4 window + compressed KV caches into a contiguous BF16 tensor.

    When ``fp8_compressed_kv_cache`` is provided and ``compressed_kv_cache`` is
    None, the kernel dequantizes FP8 data in-place (per-tile FP8→BF16 with
    e8m0fnu scales) instead of allocating a full intermediate BF16 tensor.

    Flat KV layout per sequence:
        [prev_window (ring buffer) | raw_kv (current chunk) | compressed_kv]

    For first-time prefill (start_pos=0), prev_window is empty and all current
    tokens come from raw_kv. For chunked prefill (start_pos>0), prev_window
    contains the last min(start_pos, window_size) tokens from the ring buffer.

    Args:
        window_kv_cache: When ``slot`` is None, a [bsz, window_size, head_dim]
            BF16 ring buffer already batch-gathered.  When ``slot`` is provided,
            the full global cache [num_total_slots, window_size, head_dim] indexed
            by ``slot[batch_id]``.
        compressed_kv_cache: [num_blocks, entries_per_block, head_dim] BF16 paged
            cache, or None if no compression or reading from FP8.
        block_offsets: [bsz, num_blocks] page table.
        kv_seqlens: [bsz] total KV length per sequence.
        window_size: sliding window size.
        compress_ratio: compression ratio (4 or 128), 0 if no compression.
        total_kv_tokens: safe upper bound on the total number of flat KV
            tokens across all sequences. Used for output allocation; slight
            over-estimation is acceptable.
        max_flat_kv_len: safe upper bound on the maximum flat KV length
            across sequences. Used for Triton grid size; slight
            over-estimation is acceptable (excess programs exit immediately).
        cu_seqlens_k: optional [bsz+1] int32 cumulative KV sequence lengths.
            If None, computed from kv_seqlens.
        fp8_compressed_kv_cache: optional [num_blocks, entries_per_block, 584]
            FP8 V4 FlashMLA sparse paged cache. When provided and
            compressed_kv_cache is None, the kernel dequantizes FP8 data
            in-place instead of using a Python-side dequantization.
        slot: optional [bsz] int64 slot indices into the global
            ``window_kv_cache``.  When provided, the kernel uses
            ``slot[batch_id]`` instead of ``batch_id`` to index the window
            cache.  Negative slot values produce all-zero output for that
            sequence's window region (handles padded / unallocated slots).
        raw_kv: optional [total_q_tokens, head_dim] raw KV from the model's
            wkv projection for the current prefill chunk. When provided,
            the kernel reads current-chunk tokens from this tensor instead
            of the ring buffer, so the ring buffer's previous window is
            preserved for chunked prefill.
        raw_kv_lens: optional [bsz] per-sequence raw KV entry count
            (= q_seqlens). Required when raw_kv is provided.
        start_pos: optional [bsz] start positions per sequence. Required
            when raw_kv is provided. Used to compute prev_window_len =
            min(start_pos, window_size).

    Returns:
        flat_kv: [total_kv_tokens, 1, head_dim] BF16 flat tensor.
        cu_seqlens_k: [bsz+1] int32 cumulative sequence lengths.
    """
    bsz = kv_seqlens.numel()
    head_dim = window_kv_cache.size(-1)
    device = kv_seqlens.device

    has_raw_kv = raw_kv is not None

    if has_raw_kv:
        prev_window_lens = start_pos.clamp(max=window_size).long()
        raw_kv_lens_t = raw_kv_lens.long()
    else:
        prev_window_lens = kv_seqlens.clamp(max=window_size).long()
        raw_kv_lens_t = kv_seqlens.new_zeros(bsz, dtype=torch.long)

    if compress_ratio > 0:
        num_compressed = torch.div(kv_seqlens, compress_ratio, rounding_mode='floor').long()
    else:
        num_compressed = kv_seqlens.new_zeros(kv_seqlens.shape, dtype=torch.long)
    flat_kv_lens = (prev_window_lens + raw_kv_lens_t + num_compressed).to(torch.int32)

    if cu_seqlens_k is None:
        cu_seqlens_k = torch.zeros(bsz + 1, dtype=torch.int32, device=device)
        torch.cumsum(flat_kv_lens, dim=0, out=cu_seqlens_k[1:])

    flat_kv = window_kv_cache.new_empty(total_kv_tokens, 1, head_dim)

    has_compress = compress_ratio > 0 and (compressed_kv_cache is not None
                                           or fp8_compressed_kv_cache is not None)
    has_fp8_compress = (compress_ratio > 0 and fp8_compressed_kv_cache is not None
                        and compressed_kv_cache is None)
    block_size = compressed_kv_cache.size(1) if compressed_kv_cache is not None else (
        fp8_compressed_kv_cache.size(1) if fp8_compressed_kv_cache is not None else 1)
    has_slot = slot is not None

    if max_flat_kv_len == 0:
        return flat_kv, cu_seqlens_k

    grid = (bsz, max_flat_kv_len)

    # Build FP8 views (same pattern as v4_compressor.py / v4_pack_window.py)
    if has_fp8_compress:
        num_blocks = fp8_compressed_kv_cache.size(0)
        entries_per_block = fp8_compressed_kv_cache.size(1)
        D_NOPE = 448
        D_ROPE_BF16 = 64
        NR_DIM = D_NOPE + 2 * D_ROPE_BF16  # 576 bytes per token

        fp8_flat = fp8_compressed_kv_cache.view(num_blocks, -1)
        fp8_nope_rope = fp8_flat[:, :entries_per_block * NR_DIM].view(
            num_blocks, entries_per_block, NR_DIM)

        fp8_nope_rope_bf16 = fp8_nope_rope.view(torch.bfloat16)
        fp8_rope_bf16 = fp8_nope_rope_bf16[:, :, D_NOPE // 2:]

        fp8_scales_u8 = fp8_flat[:, entries_per_block * NR_DIM:].view(
            num_blocks, entries_per_block, 8).view(torch.uint8)
    else:
        dummy = torch.empty(1, 1, 16, dtype=torch.bfloat16, device=device)
        fp8_nope_rope = dummy
        fp8_rope_bf16 = dummy.view(torch.bfloat16)
        fp8_scales_u8 = dummy.view(torch.uint8)

    # Build raw KV tensors
    if has_raw_kv:
        cu_raw_kv = torch.zeros(bsz + 1, dtype=torch.int32, device=device)
        torch.cumsum(raw_kv_lens_t.to(torch.int32), dim=0, out=cu_raw_kv[1:])
    else:
        cu_raw_kv = kv_seqlens.new_zeros(bsz + 1, dtype=torch.int32)
        # raw_kv_ptr must still be a valid tensor for the kernel call
        raw_kv = window_kv_cache  # placeholder, HAS_RAW_KV=False so it's never read

    _flatten_v4_kv_kernel[grid](
        window_kv_cache,
        compressed_kv_cache if has_compress and not has_fp8_compress else window_kv_cache,
        flat_kv,
        cu_seqlens_k[:-1],  # start_loc
        flat_kv_lens,
        block_offsets.long(),
        slot if has_slot else kv_seqlens,  # placeholder when no slot
        window_kv_cache.stride(0),
        window_kv_cache.stride(1),
        window_kv_cache.stride(2),
        compressed_kv_cache.stride(0) if has_compress and not has_fp8_compress else 0,
        compressed_kv_cache.stride(1) if has_compress and not has_fp8_compress else 0,
        compressed_kv_cache.stride(2) if has_compress and not has_fp8_compress else 0,
        flat_kv.stride(0),
        flat_kv.stride(2),
        block_offsets.stride(0),
        fp8_nope_rope,
        fp8_nope_rope.stride(0),
        fp8_nope_rope.stride(1),
        fp8_nope_rope.stride(2),
        fp8_rope_bf16,
        fp8_rope_bf16.stride(0),
        fp8_rope_bf16.stride(1),
        fp8_rope_bf16.stride(2),
        fp8_scales_u8,
        fp8_scales_u8.stride(0),
        fp8_scales_u8.stride(1),
        fp8_scales_u8.stride(2),
        raw_kv,
        raw_kv.stride(0) if has_raw_kv else 0,
        raw_kv.stride(1) if has_raw_kv else 0,
        cu_raw_kv[:-1],
        raw_kv_lens_t.to(torch.int32),
        start_pos.to(torch.int32) if has_raw_kv else kv_seqlens.to(torch.int32),
        WINDOW_SIZE=window_size,
        COMPRESS_RATIO=compress_ratio,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        HAS_COMPRESS=has_compress,
        HAS_SLOT=has_slot,
        HAS_FP8_COMPRESS=has_fp8_compress,
        HAS_RAW_KV=has_raw_kv,
    )

    return flat_kv, cu_seqlens_k
