# Copyright (c) OpenMMLab. All rights reserved.
"""Sparse-index assembly helpers for DeepSeek-V4 FlashMLA paths."""

import torch
import triton
import triton.language as tl


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
    stride_i_k,
    stride_di_b,
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
        stride_i_k=indices.stride(2),
        stride_di_b=disabled_indices.stride(0),
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
    stride_l_k,
    stride_bo_b,
    stride_bo_i,
    stride_o_b,
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
        stride_l_k=logical_topk.stride(2),
        stride_bo_b=block_offsets.stride(0),
        stride_bo_i=block_offsets.stride(1),
        stride_o_b=out.stride(0),
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
def _build_prefill_sparse_indices_kernel(
    start_pos,
    total_lens,
    token_seq,
    token_pos,
    cu_seqlens_k,
    uncompressed_kv_lens,
    compress_topk,
    out,
    topk_length,
    stride_c_t,
    stride_c_k,
    stride_o_t,
    stride_o_k,
    WINDOW_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    TOTAL_TOPK: tl.constexpr,
    PADDED_TOPK: tl.constexpr,
    HAS_COMPRESS: tl.constexpr,
    HAS_COMPRESS_TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK + tl.arange(0, BLOCK)

    seq = tl.load(token_seq + token).to(tl.int64)
    pos = tl.load(token_pos + token).to(tl.int64)
    seq_start = tl.load(start_pos + seq).to(tl.int64)
    abs_pos = seq_start + pos
    cu = tl.load(cu_seqlens_k + seq).to(tl.int32)

    num_vis = tl.minimum(abs_pos + 1, WINDOW_SIZE)
    window_start_abs = tl.maximum(seq_start - WINDOW_SIZE, 0)
    first_vis_abs = tl.maximum(abs_pos - WINDOW_SIZE + 1, 0)
    first_flat_pos = first_vis_abs - window_start_abs

    in_window = offs < WINDOW_SIZE
    window_valid = offs < num_vis
    vals = tl.where(window_valid, first_flat_pos + offs, -1).to(tl.int32)

    if HAS_COMPRESS:
        comp_col = tl.maximum(offs - WINDOW_SIZE, 0)
        in_compress = (offs >= WINDOW_SIZE) & (offs < TOTAL_TOPK)
        comp_off = tl.load(uncompressed_kv_lens + seq).to(tl.int32)
        if HAS_COMPRESS_TOPK:
            comp_vals = tl.load(
                compress_topk + token * stride_c_t + comp_col * stride_c_k,
                mask=in_compress,
                other=-1,
            ).to(tl.int32)
            comp_vals = tl.where(comp_vals >= 0, comp_vals + comp_off, -1)
        else:
            row_compressed = tl.load(total_lens + seq).to(tl.int32) // COMPRESS_RATIO
            comp_vals = tl.where(comp_col < row_compressed, comp_col + comp_off, -1)
        vals = tl.where(in_window, vals, comp_vals)

        if tile == 0:
            causal_compressed = (abs_pos + 1) // COMPRESS_RATIO
            row_topk_length = tl.minimum(WINDOW_SIZE + causal_compressed, PADDED_TOPK)
            tl.store(topk_length + token, row_topk_length.to(tl.int32))
    elif tile == 0:
        tl.store(topk_length + token, tl.full((), WINDOW_SIZE, tl.int32))

    vals = tl.where((vals >= 0) & (offs < TOTAL_TOPK), vals + cu, -1)
    tl.store(out + token * stride_o_t + offs * stride_o_k,
             vals, mask=offs < PADDED_TOPK)


def build_prefill_sparse_indices(
    start_pos: torch.Tensor,
    total_lens: torch.Tensor,
    token_seq: torch.Tensor,
    token_pos: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    uncompressed_kv_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int = 0,
    compress_topk: torch.Tensor | None = None,
    compress_width: int = 0,
    block: int = 128,
):
    """Build padded prefill sparse indices and lengths for FlashMLA.

    This fuses the previous window-topk construction, compressed-prefix
    construction, global flat-KV offset application, head-dimension unsqueeze,
    padding, and topk-length construction into one Triton launch.
    """
    assert token_seq.dim() == 1
    assert token_pos.dim() == 1
    num_tokens = token_seq.numel()
    has_compress = compress_ratio != 0
    dummy_compress_topk = token_seq  # Pointer is never loaded when HAS_COMPRESS_TOPK is false.
    if has_compress:
        assert compress_ratio > 0
        if compress_topk is not None:
            assert compress_topk.dim() == 2
            compress_width = compress_topk.size(1)
            stride_c_t = compress_topk.stride(0)
            stride_c_k = compress_topk.stride(1)
            has_compress_topk = True
        else:
            assert compress_width is not None and compress_width >= 0
            compress_topk = dummy_compress_topk
            stride_c_t = 0
            stride_c_k = 0
            has_compress_topk = False
    else:
        compress_width = 0
        compress_topk = dummy_compress_topk
        stride_c_t = 0
        stride_c_k = 0
        has_compress_topk = False

    total_topk = window_size + compress_width
    padded_topk = triton.cdiv(max(total_topk, 1), block) * block
    out = torch.empty((num_tokens, 1, padded_topk), dtype=torch.int32, device=token_seq.device)
    topk_length = torch.empty((num_tokens, ), dtype=torch.int32, device=token_seq.device)
    if num_tokens == 0:
        return out, topk_length

    grid = (num_tokens, triton.cdiv(padded_topk, block))
    _build_prefill_sparse_indices_kernel[grid](
        start_pos,
        total_lens,
        token_seq,
        token_pos,
        cu_seqlens_k,
        uncompressed_kv_lens,
        compress_topk,
        out,
        topk_length,
        stride_c_t=stride_c_t,
        stride_c_k=stride_c_k,
        stride_o_t=out.stride(0),
        stride_o_k=out.stride(2),
        WINDOW_SIZE=window_size,
        COMPRESS_RATIO=compress_ratio,
        TOTAL_TOPK=total_topk,
        PADDED_TOPK=padded_topk,
        HAS_COMPRESS=has_compress,
        HAS_COMPRESS_TOPK=has_compress_topk,
        BLOCK=block,
    )
    return out, topk_length
