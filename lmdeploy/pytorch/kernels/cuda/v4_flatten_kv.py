# Copyright (c) OpenMMLab. All rights reserved.
"""Triton kernel to flatten V4 window + compressed KV caches into a contiguous
BF16 tensor for ``flash_mla_sparse_fwd`` prefill."""

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
    total_lens_ptr,
    block_offsets_ptr,
    stride_wkv_b,
    stride_wkv_s,
    stride_wkv_d,
    stride_ckv_b,
    stride_ckv_s,
    stride_ckv_d,
    stride_out_s,
    stride_out_d,
    stride_boff,
    WINDOW_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_COMPRESS: tl.constexpr,
):
    batch_id = tl.program_id(0)
    token_id = tl.program_id(1)

    flat_kv_len = tl.load(flat_kv_lens_ptr + batch_id)
    start_loc = tl.load(start_loc_ptr + batch_id)

    if token_id >= flat_kv_len:
        return

    offs_d = tl.arange(0, HEAD_DIM)

    window_kv_len = tl.minimum(tl.load(total_lens_ptr + batch_id), WINDOW_SIZE)

    if token_id < window_kv_len:
        # ---- Window region ----
        # Chronological ordering within the ring buffer.
        total_len = tl.load(total_lens_ptr + batch_id)
        window_start = total_len - WINDOW_SIZE if total_len > WINDOW_SIZE else 0
        actual_pos = window_start + token_id
        ring_pos = actual_pos % WINDOW_SIZE

        src_ptr = (window_kv_ptr + batch_id * stride_wkv_b
                   + ring_pos * stride_wkv_s
                   + offs_d * stride_wkv_d)
        val = tl.load(src_ptr, mask=actual_pos < total_len, other=0.0)
    else:
        # ---- Compressed region ----
        if not HAS_COMPRESS:
            val = tl.zeros((HEAD_DIM,), dtype=out_ptr.dtype.element_ty)
        else:
            comp_pos = token_id - window_kv_len
            page_id = comp_pos // BLOCK_SIZE
            page_off = comp_pos % BLOCK_SIZE
            phys_block = tl.load(block_offsets_ptr + batch_id * stride_boff + page_id)
            src_ptr = (compressed_kv_ptr + phys_block.to(tl.int64) * stride_ckv_b
                       + page_off * stride_ckv_s
                       + offs_d * stride_ckv_d)
            val = tl.load(src_ptr)

    # Write to flat output
    out_ptr_off = (start_loc + token_id) * stride_out_s + offs_d * stride_out_d
    tl.store(out_ptr + out_ptr_off, val)


def flatten_v4_kv(
    window_kv_cache: torch.Tensor,
    compressed_kv_cache: torch.Tensor | None,
    block_offsets: torch.Tensor,
    kv_seqlens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    cu_seqlens_k: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten V4 window + compressed KV caches into a contiguous BF16 tensor.

    Args:
        window_kv_cache: [bsz, window_size, head_dim] BF16 ring buffer.
        compressed_kv_cache: [num_blocks, block_size, head_dim] BF16 paged
            cache, or None if no compression.
        block_offsets: [bsz, num_blocks] page table.
        kv_seqlens: [bsz] total KV length per sequence.
        window_size: sliding window size.
        compress_ratio: compression ratio (4 or 128), 0 if no compression.
        cu_seqlens_k: optional [bsz+1] int32 cumulative KV sequence lengths.
            If None, computed from kv_seqlens.

    Returns:
        flat_kv: [total_kv_tokens, 1, head_dim] BF16 flat tensor.
        cu_seqlens_k: [bsz+1] int32 cumulative sequence lengths.
    """
    bsz = kv_seqlens.numel()
    head_dim = window_kv_cache.size(-1)
    device = kv_seqlens.device
    dtype = window_kv_cache.dtype

    window_kv_lens = kv_seqlens.clamp(max=window_size)
    num_compressed = torch.div(kv_seqlens, compress_ratio, rounding_mode='floor').long() if compress_ratio > 0 else kv_seqlens.new_zeros(kv_seqlens.shape, dtype=torch.long)
    flat_kv_lens = (window_kv_lens + num_compressed).to(torch.int32)

    if cu_seqlens_k is None:
        cu_seqlens_k = torch.zeros(bsz + 1, dtype=torch.int32, device=device)
        torch.cumsum(flat_kv_lens, dim=0, out=cu_seqlens_k[1:])
    total_kv_tokens = cu_seqlens_k[-1].item()

    flat_kv = window_kv_cache.new_empty(total_kv_tokens, 1, head_dim)

    has_compress = compress_ratio > 0 and compressed_kv_cache is not None
    block_size = compressed_kv_cache.size(1) if has_compress else 1

    # Launch one program per (batch_item, token_position)
    max_flat_kv_len = int(flat_kv_lens.max().item()) if bsz > 0 else 0
    if max_flat_kv_len == 0:
        return flat_kv, cu_seqlens_k

    grid = (bsz, max_flat_kv_len)

    _flatten_v4_kv_kernel[grid](
        window_kv_cache,
        compressed_kv_cache if has_compress else window_kv_cache,  # placeholder
        flat_kv,
        cu_seqlens_k[:-1],  # start_loc
        flat_kv_lens,
        kv_seqlens.to(torch.int32),
        block_offsets.long(),
        stride_wkv_b=window_kv_cache.stride(0),
        stride_wkv_s=window_kv_cache.stride(1),
        stride_wkv_d=window_kv_cache.stride(2),
        stride_ckv_b=compressed_kv_cache.stride(0) if has_compress else 0,
        stride_ckv_s=compressed_kv_cache.stride(1) if has_compress else 0,
        stride_ckv_d=compressed_kv_cache.stride(2) if has_compress else 0,
        stride_out_s=flat_kv.stride(0),
        stride_out_d=flat_kv.stride(2),
        stride_boff=block_offsets.stride(0),
        WINDOW_SIZE=window_size,
        COMPRESS_RATIO=compress_ratio,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        HAS_COMPRESS=has_compress,
    )

    return flat_kv, cu_seqlens_k
