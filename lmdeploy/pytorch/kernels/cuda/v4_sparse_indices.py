# Copyright (c) OpenMMLab. All rights reserved.
"""Sparse-index assembly helpers for DeepSeek-V4 FlashMLA paths."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pad_sparse_indices_kernel(
    indices,
    out,
    stride_i_row,
    stride_i_col,
    stride_o_row,
    stride_o_col,
    TOPK: tl.constexpr,
    PADDED_TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK + tl.arange(0, BLOCK)
    valid = offs < TOPK
    vals = tl.load(indices + row * stride_i_row + offs * stride_i_col,
                   mask=valid, other=-1).to(tl.int32)
    vals = tl.where(valid, vals, -1)
    tl.store(out + row * stride_o_row + offs * stride_o_col,
             vals, mask=offs < PADDED_TOPK)


def pad_sparse_indices(indices: torch.Tensor | None, block: int = 128):
    """Pad FlashMLA sparse indices on the last dimension with ``-1``."""
    if indices is None:
        return None
    topk = indices.size(-1)
    padded_topk = triton.cdiv(topk, block) * block
    if padded_topk == topk:
        return indices

    out_shape = tuple(indices.shape[:-1]) + (padded_topk,)
    out = torch.empty(out_shape, dtype=torch.int32, device=indices.device)
    rows = indices.numel() // topk
    indices_2d = indices.reshape(rows, topk)
    out_2d = out.reshape(rows, padded_topk)
    grid = (rows, triton.cdiv(padded_topk, block))
    _pad_sparse_indices_kernel[grid](
        indices_2d,
        out_2d,
        stride_i_row=indices_2d.stride(0),
        stride_i_col=indices_2d.stride(1),
        stride_o_row=out_2d.stride(0),
        stride_o_col=out_2d.stride(1),
        TOPK=topk,
        PADDED_TOPK=padded_topk,
        BLOCK=block,
    )
    return out


@triton.jit
def _build_decode_window_sparse_indices_kernel(
    kv_seqlens,
    start_pos,
    is_padded,
    indices,
    topk_length,
    window_pos,
    disabled_indices,
    disabled_topk_length,
    stride_i_b,
    stride_i_h,
    stride_i_k,
    stride_di_b,
    stride_di_h,
    stride_di_k,
    WINDOW_SIZE: tl.constexpr,
    PADDED_WINDOW_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK + tl.arange(0, BLOCK)

    kv_len = tl.load(kv_seqlens + batch).to(tl.int32)
    seq_start = tl.load(start_pos + batch).to(tl.int32)
    padded = tl.load(is_padded + batch)
    window_len = tl.minimum(kv_len, WINDOW_SIZE)
    first_abs_pos = kv_len - window_len
    valid = offs < window_len
    ring_pos = (first_abs_pos + offs) % WINDOW_SIZE
    vals = tl.where(valid, batch * WINDOW_SIZE + ring_pos, -1)

    out_ptrs = indices + batch * stride_i_b + offs * stride_i_k
    tl.store(out_ptrs, vals, mask=offs < PADDED_WINDOW_SIZE)

    if tile == 0:
        tl.store(topk_length + batch, tl.where(padded, 1, window_len))
        tl.store(window_pos + batch, seq_start % WINDOW_SIZE)
        tl.store(disabled_topk_length + batch, tl.where(padded, 1, 0))
        tl.store(
            disabled_indices + batch * stride_di_b + offs * stride_di_k,
            -1,
            mask=offs < BLOCK,
        )


def build_decode_window_sparse_indices(
    kv_seqlens: torch.Tensor,
    start_pos: torch.Tensor,
    is_padded: torch.Tensor,
    window_size: int,
    block: int = 128,
):
    """Build decode window metadata directly in FlashMLA sparse-index form.

    Returns:
        - indices: [bsz, 1, padded_window_size] int32, already offset into the
          flattened ``extra_k_cache`` layout and padded with ``-1``.
        - topk_length: [bsz] int32, with padded slots clamped to length 1.
        - window_pos: [bsz] int32 ring-buffer write position.
        - disabled_indices: [bsz, 1, block] int32, pre-padded ``-1``
          primary indices for no-compression decode layers.
        - disabled_topk_length: [bsz] int32, zero for active rows and one
          for padded rows.
    """
    bsz = kv_seqlens.numel()
    padded_window_size = triton.cdiv(window_size, block) * block
    indices = torch.empty((bsz, 1, padded_window_size), dtype=torch.int32, device=kv_seqlens.device)
    topk_length = torch.empty((bsz, ), dtype=torch.int32, device=kv_seqlens.device)
    window_pos = torch.empty((bsz, ), dtype=torch.int32, device=kv_seqlens.device)
    disabled_indices = torch.empty((bsz, 1, block), dtype=torch.int32, device=kv_seqlens.device)
    disabled_topk_length = torch.empty((bsz, ), dtype=torch.int32, device=kv_seqlens.device)
    if bsz == 0:
        return indices, topk_length, window_pos, disabled_indices, disabled_topk_length

    grid = (bsz, triton.cdiv(padded_window_size, block))
    _build_decode_window_sparse_indices_kernel[grid](
        kv_seqlens,
        start_pos,
        is_padded,
        indices,
        topk_length,
        window_pos,
        disabled_indices,
        disabled_topk_length,
        stride_i_b=indices.stride(0),
        stride_i_h=indices.stride(1),
        stride_i_k=indices.stride(2),
        stride_di_b=disabled_indices.stride(0),
        stride_di_h=disabled_indices.stride(1),
        stride_di_k=disabled_indices.stride(2),
        WINDOW_SIZE=window_size,
        PADDED_WINDOW_SIZE=padded_window_size,
        BLOCK=block,
    )
    return indices, topk_length, window_pos, disabled_indices, disabled_topk_length


@triton.jit
def _build_decode_compressed_sparse_indices_kernel(
    logical_topk,
    block_offsets,
    out,
    stride_l_b,
    stride_l_h,
    stride_l_k,
    stride_bo_b,
    stride_bo_i,
    stride_o_b,
    stride_o_h,
    stride_o_k,
    TOPK: tl.constexpr,
    PADDED_TOPK: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK + tl.arange(0, BLOCK)
    valid_col = offs < TOPK

    logical = tl.load(
        logical_topk + batch * stride_l_b + offs * stride_l_k,
        mask=valid_col,
        other=-1,
    ).to(tl.int64)
    safe_logical = tl.maximum(logical, 0)

    token_positions = safe_logical * COMPRESS_RATIO
    block_idx = token_positions // BLOCK_SIZE
    safe_block_idx = tl.minimum(block_idx, NUM_BLOCKS - 1)
    block_idx_valid = block_idx < NUM_BLOCKS
    phys_block = tl.load(
        block_offsets + batch * stride_bo_b + safe_block_idx * stride_bo_i,
        mask=valid_col,
        other=0,
    ).to(tl.int64)

    entries_per_block: tl.constexpr = BLOCK_SIZE // COMPRESS_RATIO
    block_off = safe_logical % entries_per_block
    phys_indices = phys_block * entries_per_block + block_off
    valid = valid_col & (logical >= 0) & block_idx_valid
    vals = tl.where(valid, phys_indices, -1).to(tl.int32)

    tl.store(
        out + batch * stride_o_b + offs * stride_o_k,
        vals,
        mask=offs < PADDED_TOPK,
    )


def build_decode_compressed_sparse_indices(
    logical_topk: torch.Tensor,
    block_offsets: torch.Tensor,
    block_size: int,
    compress_ratio: int,
    block: int = 128,
) -> torch.Tensor:
    """Convert decode compressed logical topk to padded physical indices.

    This is the fused equivalent of the old decode updater plus final
    FlashMLA index padding. ``logical_topk`` is expected in
    ``[bsz, 1, topk]`` form from the V4 indexer.
    """
    assert logical_topk.dim() == 3
    bsz = logical_topk.size(0)
    topk = logical_topk.size(-1)
    padded_topk = triton.cdiv(topk, block) * block
    out = torch.empty((bsz, 1, padded_topk), dtype=torch.int32, device=logical_topk.device)
    if bsz == 0:
        return out

    grid = (bsz, triton.cdiv(padded_topk, block))
    _build_decode_compressed_sparse_indices_kernel[grid](
        logical_topk,
        block_offsets,
        out,
        stride_l_b=logical_topk.stride(0),
        stride_l_h=logical_topk.stride(1),
        stride_l_k=logical_topk.stride(2),
        stride_bo_b=block_offsets.stride(0),
        stride_bo_i=block_offsets.stride(1),
        stride_o_b=out.stride(0),
        stride_o_h=out.stride(1),
        stride_o_k=out.stride(2),
        TOPK=topk,
        PADDED_TOPK=padded_topk,
        NUM_BLOCKS=block_offsets.size(1),
        BLOCK_SIZE=block_size,
        COMPRESS_RATIO=compress_ratio,
        BLOCK=block,
    )
    return out


@triton.jit
def _build_decode_prefix_compressed_sparse_indices_kernel(
    num_compressed,
    block_offsets,
    out,
    stride_bo_b,
    stride_bo_i,
    stride_o_b,
    stride_o_k,
    TOPK: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK + tl.arange(0, BLOCK)

    row_topk = tl.load(num_compressed + batch).to(tl.int32)
    logical = offs.to(tl.int64)
    token_positions = logical * COMPRESS_RATIO
    block_idx = token_positions // BLOCK_SIZE
    safe_block_idx = tl.minimum(block_idx, NUM_BLOCKS - 1)
    block_idx_valid = block_idx < NUM_BLOCKS
    phys_block = tl.load(
        block_offsets + batch * stride_bo_b + safe_block_idx * stride_bo_i,
        mask=offs < TOPK,
        other=0,
    ).to(tl.int64)

    entries_per_block: tl.constexpr = BLOCK_SIZE // COMPRESS_RATIO
    block_off = logical % entries_per_block
    phys_indices = phys_block * entries_per_block + block_off
    valid = (offs < row_topk) & block_idx_valid
    vals = tl.where(valid, phys_indices, -1).to(tl.int32)
    tl.store(out + batch * stride_o_b + offs * stride_o_k, vals, mask=offs < TOPK)


def build_decode_prefix_compressed_sparse_indices(
    num_compressed: torch.Tensor,
    block_offsets: torch.Tensor,
    block_size: int,
    compress_ratio: int,
    max_topk: int | None = None,
    block: int = 128,
) -> torch.Tensor:
    """Build physical padded decode indices for full-prefix compressed KV."""
    if max_topk is None:
        max_topk = max(block_offsets.size(1) * block_size // compress_ratio, 1)
    padded_topk = triton.cdiv(max_topk, block) * block
    bsz = num_compressed.numel()
    out = torch.empty((bsz, 1, padded_topk), dtype=torch.int32, device=num_compressed.device)
    if bsz == 0:
        return out

    grid = (bsz, triton.cdiv(padded_topk, block))
    _build_decode_prefix_compressed_sparse_indices_kernel[grid](
        num_compressed,
        block_offsets,
        out,
        stride_bo_b=block_offsets.stride(0),
        stride_bo_i=block_offsets.stride(1),
        stride_o_b=out.stride(0),
        stride_o_k=out.stride(2),
        TOPK=padded_topk,
        NUM_BLOCKS=block_offsets.size(1),
        BLOCK_SIZE=block_size,
        COMPRESS_RATIO=compress_ratio,
        BLOCK=block,
    )
    return out


@triton.jit
def _assemble_prefill_sparse_indices_kernel(
    window_topk,
    compress_topk,
    repeat_cu,
    compress_offset,
    out,
    stride_w_t,
    stride_w_k,
    stride_c_t,
    stride_c_k,
    stride_co_t,
    stride_o_t,
    stride_o_h,
    stride_o_k,
    WINDOW_TOPK: tl.constexpr,
    COMPRESS_TOPK: tl.constexpr,
    TOTAL_TOPK: tl.constexpr,
    PADDED_TOPK: tl.constexpr,
    HAS_COMPRESS: tl.constexpr,
    HAS_COMPRESS_OFFSET: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK + tl.arange(0, BLOCK)

    in_window = offs < WINDOW_TOPK
    valid_topk = offs < TOTAL_TOPK
    window_vals = tl.load(window_topk + token * stride_w_t + offs * stride_w_k,
                          mask=in_window, other=-1).to(tl.int32)
    vals = window_vals

    if HAS_COMPRESS:
        comp_col = tl.maximum(offs - WINDOW_TOPK, 0)
        in_compress = (offs >= WINDOW_TOPK) & (offs < TOTAL_TOPK)
        comp_vals = tl.load(compress_topk + token * stride_c_t + comp_col * stride_c_k,
                            mask=in_compress, other=-1).to(tl.int32)
        if HAS_COMPRESS_OFFSET:
            comp_off = tl.load(compress_offset + token * stride_co_t).to(tl.int32)
            comp_vals = tl.where(comp_vals >= 0, comp_vals + comp_off, -1)
        vals = tl.where(in_window, window_vals, comp_vals)

    cu = tl.load(repeat_cu + token).to(tl.int32)
    vals = tl.where((vals >= 0) & valid_topk, vals + cu, -1)
    tl.store(out + token * stride_o_t + offs * stride_o_k,
             vals, mask=offs < PADDED_TOPK)


def assemble_prefill_sparse_indices(
    window_topk: torch.Tensor,
    compress_topk: torch.Tensor | None,
    repeat_cu: torch.Tensor,
    compress_offset: torch.Tensor | None = None,
    block: int = 128,
) -> torch.Tensor:
    """Assemble padded prefill sparse indices for ``flash_mla_sparse_fwd``.

    ``window_topk`` and ``compress_topk`` are per-sequence-local positions.
    This kernel concatenates them, optionally applies the per-token compressed
    offset for indexer output, adds the flat-KV ``repeat_cu`` base, unsqueezes
    the head dimension, and pads the final width with ``-1``.
    """
    assert window_topk.dim() == 2
    assert repeat_cu.dim() == 1
    num_tokens = window_topk.size(0)
    window_width = window_topk.size(1)
    has_compress = compress_topk is not None
    compress_width = compress_topk.size(1) if has_compress else 0
    total_topk = window_width + compress_width
    padded_topk = triton.cdiv(total_topk, block) * block
    out = torch.empty((num_tokens, 1, padded_topk), dtype=torch.int32, device=window_topk.device)
    if num_tokens == 0:
        return out

    dummy = window_topk
    if compress_topk is None:
        compress_topk = dummy
    if compress_offset is None:
        compress_offset = dummy
        has_compress_offset = False
    else:
        has_compress_offset = True
        if compress_offset.dim() == 2:
            assert compress_offset.size(1) == 1
            compress_offset = compress_offset.reshape(-1)

    grid = (num_tokens, triton.cdiv(padded_topk, block))
    _assemble_prefill_sparse_indices_kernel[grid](
        window_topk,
        compress_topk,
        repeat_cu,
        compress_offset,
        out,
        stride_w_t=window_topk.stride(0),
        stride_w_k=window_topk.stride(1),
        stride_c_t=compress_topk.stride(0),
        stride_c_k=compress_topk.stride(1),
        stride_co_t=compress_offset.stride(0),
        stride_o_t=out.stride(0),
        stride_o_h=out.stride(1),
        stride_o_k=out.stride(2),
        WINDOW_TOPK=window_width,
        COMPRESS_TOPK=compress_width,
        TOTAL_TOPK=total_topk,
        PADDED_TOPK=padded_topk,
        HAS_COMPRESS=has_compress,
        HAS_COMPRESS_OFFSET=has_compress_offset,
        BLOCK=block,
    )
    return out
