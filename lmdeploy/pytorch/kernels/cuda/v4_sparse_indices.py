# Copyright (c) OpenMMLab. All rights reserved.
"""Sparse-index assembly helpers for DeepSeek-V4 FlashMLA paths."""

import torch
import triton
import triton.language as tl


@triton.jit
def _build_decode_window_sparse_indices_kernel(
    kv_seqlens,
    start_pos,
    slot,
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
    RING_STORAGE_CAPACITY: tl.constexpr,
    PADDED_WINDOW_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK + tl.arange(0, BLOCK)

    kv_len = tl.load(kv_seqlens + batch).to(tl.int32)
    seq_start = tl.load(start_pos + batch).to(tl.int32)
    cur_slot = tl.load(slot + batch).to(tl.int64)
    padded = cur_slot < 0
    safe_slot = tl.maximum(cur_slot, 0)
    window_len = tl.minimum(kv_len, WINDOW_SIZE)
    first_abs_pos = kv_len - window_len
    valid = offs < window_len
    ring_pos = (first_abs_pos + offs) % RING_STORAGE_CAPACITY
    vals = tl.where(valid, safe_slot * RING_STORAGE_CAPACITY + ring_pos, -1)
    vals = tl.where(padded & (offs == 0), 0, vals)

    out_ptrs = indices + batch * stride_i_b + offs * stride_i_k
    tl.store(out_ptrs, vals, mask=offs < PADDED_WINDOW_SIZE)

    if tile == 0:
        tl.store(topk_length + batch, tl.where(padded, 1, window_len))
        tl.store(window_pos + batch, seq_start % RING_STORAGE_CAPACITY)
        tl.store(disabled_topk_length + batch, tl.where(padded, 1, 0))
        tl.store(
            disabled_indices + batch * stride_di_b + offs * stride_di_k,
            tl.where(padded & (offs == 0), 0, -1),
            mask=offs < BLOCK,
        )


def build_decode_window_sparse_indices(
    kv_seqlens: torch.Tensor,
    start_pos: torch.Tensor,
    slot: torch.Tensor,
    window_size: int,
    ring_storage_capacity: int | None = None,
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
    if ring_storage_capacity is None:
        ring_storage_capacity = window_size
    if ring_storage_capacity < window_size:
        raise ValueError('V4 ring storage capacity must be at least the logical window size, '
                         f'got {ring_storage_capacity} < {window_size}.')
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
        slot,
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
        RING_STORAGE_CAPACITY=ring_storage_capacity,
        PADDED_WINDOW_SIZE=padded_window_size,
        BLOCK=block,
    )
    return indices, topk_length, window_pos, disabled_indices, disabled_topk_length


@triton.jit
def _build_rectangular_decode_window_sparse_indices_kernel(
    start_pos,
    q_seqlens,
    slot,
    indices,
    topk_length,
    write_slot,
    write_pos,
    token_total_lens,
    disabled_indices,
    disabled_topk_length,
    stride_i_r,
    stride_i_k,
    stride_di_r,
    stride_di_k,
    QUERY_LEN: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    RING_STORAGE_CAPACITY: tl.constexpr,
    PADDED_WINDOW_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Build one qlen=1 sparse-decode row per verifier token."""
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK + tl.arange(0, BLOCK)
    batch = row // QUERY_LEN
    position = row % QUERY_LEN

    seq_start = tl.load(start_pos + batch).to(tl.int64)
    q_len = tl.load(q_seqlens + batch).to(tl.int32)
    cur_slot = tl.load(slot + batch).to(tl.int64)
    active = (cur_slot >= 0) & (position < q_len)
    safe_slot = tl.maximum(cur_slot, 0)

    total_len = seq_start + position + 1
    window_len = tl.minimum(total_len, WINDOW_SIZE)
    first_abs_pos = total_len - window_len
    visible = active & (offs < window_len)
    physical = (safe_slot * RING_STORAGE_CAPACITY
                + (first_abs_pos + offs) % RING_STORAGE_CAPACITY)
    # Padded graph rows still need one safe address because FlashMLA clamps
    # their visible length to one. Their writes remain disabled below.
    vals = tl.where(visible, physical, -1)
    vals = tl.where((~active) & (offs == 0), 0, vals).to(tl.int32)
    tl.store(indices + row * stride_i_r + offs * stride_i_k,
             vals, mask=offs < PADDED_WINDOW_SIZE)

    if tile == 0:
        tl.store(topk_length + row,
                 tl.where(active, window_len, 1).to(tl.int32))
        tl.store(write_slot + row,
                 tl.where(active, cur_slot, -1).to(tl.int64))
        tl.store(write_pos + row,
                 tl.where(active, (seq_start + position)
                          % RING_STORAGE_CAPACITY, -1).to(tl.int64))
        tl.store(token_total_lens + row,
                 tl.where(active, total_len, 1).to(tl.int32))

        disabled = tl.where((~active) & (offs == 0), 0, -1).to(tl.int32)
        tl.store(disabled_indices + row * stride_di_r + offs * stride_di_k,
                 disabled, mask=offs < BLOCK)
        tl.store(disabled_topk_length + row,
                 tl.where(active, 0, 1).to(tl.int32))


def build_rectangular_decode_window_sparse_indices(
    start_pos: torch.Tensor,
    q_seqlens: torch.Tensor,
    slot: torch.Tensor,
    query_len: int,
    window_size: int,
    ring_storage_capacity: int,
    block: int = 128,
):
    """Build packed qlen=1 window metadata for rectangular verification.

    Physical storage uses ``ring_storage_capacity`` while visibility remains
    capped by ``window_size``. Rows are request-major: ``row = b * Q + p``.
    """
    if query_len < 1:
        raise ValueError(f'V4 rectangular query length must be positive, got {query_len}.')
    if ring_storage_capacity < window_size + query_len - 1:
        raise ValueError('V4 rectangular verification requires ring storage capacity '
                         f'>= window_size + query_len - 1, got {ring_storage_capacity} '
                         f'< {window_size + query_len - 1}.')
    bsz = start_pos.numel()
    if q_seqlens.numel() != bsz or slot.numel() != bsz:
        raise ValueError('start_pos, q_seqlens, and slot must have the same length.')
    num_rows = bsz * query_len
    padded_window_size = triton.cdiv(window_size, block) * block
    device = start_pos.device
    indices = torch.empty((num_rows, 1, padded_window_size),
                          dtype=torch.int32, device=device)
    topk_length = torch.empty((num_rows,), dtype=torch.int32, device=device)
    write_slot = torch.empty((num_rows,), dtype=torch.long, device=device)
    write_pos = torch.empty((num_rows,), dtype=torch.long, device=device)
    token_total_lens = torch.empty((num_rows,), dtype=torch.int32, device=device)
    disabled_indices = torch.empty((num_rows, 1, block), dtype=torch.int32,
                                   device=device)
    disabled_topk_length = torch.empty((num_rows,), dtype=torch.int32,
                                       device=device)
    if num_rows == 0:
        return (indices, topk_length, write_slot, write_pos,
                token_total_lens, disabled_indices, disabled_topk_length)

    grid = (num_rows, triton.cdiv(padded_window_size, block))
    _build_rectangular_decode_window_sparse_indices_kernel[grid](
        start_pos,
        q_seqlens,
        slot,
        indices,
        topk_length,
        write_slot,
        write_pos,
        token_total_lens,
        disabled_indices,
        disabled_topk_length,
        stride_i_r=indices.stride(0),
        stride_i_k=indices.stride(2),
        stride_di_r=disabled_indices.stride(0),
        stride_di_k=disabled_indices.stride(2),
        QUERY_LEN=query_len,
        WINDOW_SIZE=window_size,
        RING_STORAGE_CAPACITY=ring_storage_capacity,
        PADDED_WINDOW_SIZE=padded_window_size,
        BLOCK=block,
    )
    return (indices, topk_length, write_slot, write_pos, token_total_lens,
            disabled_indices, disabled_topk_length)


@triton.jit
def _build_rectangular_decode_split_scheduler_kernel(
    primary_topk_length,
    extra_topk_length,
    scheduler_metadata,
    num_splits,
    stride_sched_p,
    BATCH_SIZE: tl.constexpr,
    NUM_PARTITIONS: tl.constexpr,
    TOPK_BLOCK_SIZE: tl.constexpr,
):
    """Partition actual KV blocks without FlashMLA's fixed overhead.

    With ``s_q=Q``, FlashMLA reduces the SM partition budget but still charges
    five accounting-only blocks per request. That collapses short V4 sparse
    requests to one split and changes the BF16 reduction tree versus Q
    independent qlen-one calls. For the common P>=B case, keep partitions
    request-local and balance the real primary/extra KV blocks across them.
    """
    pid = tl.program_id(0)

    total_blocks = 0
    base_splits = NUM_PARTITIONS // BATCH_SIZE
    base_split_total = 0
    extra_eligible = 0
    for batch in range(BATCH_SIZE):
        primary = tl.load(primary_topk_length + batch).to(tl.int32)
        extra = tl.load(extra_topk_length + batch).to(tl.int32)
        primary_blocks = tl.maximum(
            (primary + TOPK_BLOCK_SIZE - 1) // TOPK_BLOCK_SIZE, 1)
        extra_blocks = (extra + TOPK_BLOCK_SIZE - 1) // TOPK_BLOCK_SIZE
        blocks = primary_blocks + extra_blocks
        total_blocks += blocks
        if NUM_PARTITIONS >= BATCH_SIZE:
            request_base = tl.minimum(blocks, base_splits)
            base_split_total += request_base
            extra_eligible += blocks > request_base

    payload = tl.maximum(
        (total_blocks + NUM_PARTITIONS - 1) // NUM_PARTITIONS, 1)

    if pid < NUM_PARTITIONS:
        begin_req = BATCH_SIZE
        end_req = BATCH_SIZE
        begin_block = 0
        end_block = 0
        begin_split_idx = 0
        begin_is_split = 0
        end_is_split = 0
        active = False

        if NUM_PARTITIONS >= BATCH_SIZE:
            # Keep partitions request-local. This avoids coupling one
            # request's reduction tree to the lengths of its neighbours and
            # most closely matches Q independent qlen-one schedules.
            remaining = NUM_PARTITIONS - base_split_total
            remaining = tl.minimum(remaining, extra_eligible)
            split_prefix = 0
            eligible_prefix = 0
            for batch in range(BATCH_SIZE):
                primary = tl.load(primary_topk_length + batch).to(tl.int32)
                extra = tl.load(extra_topk_length + batch).to(tl.int32)
                primary_blocks = tl.maximum(
                    (primary + TOPK_BLOCK_SIZE - 1) // TOPK_BLOCK_SIZE, 1)
                extra_blocks = (extra + TOPK_BLOCK_SIZE - 1) // TOPK_BLOCK_SIZE
                blocks = primary_blocks + extra_blocks
                request_base = tl.minimum(blocks, base_splits)
                eligible = blocks > request_base
                gets_extra = eligible & (eligible_prefix < remaining)
                splits = request_base + gets_extra
                owns = (pid >= split_prefix) & (pid < split_prefix + splits)
                local_split = pid - split_prefix
                begin_req = tl.where(owns, batch, begin_req)
                end_req = tl.where(owns, batch, end_req)
                begin_block = tl.where(
                    owns, local_split * blocks // splits, begin_block)
                end_block = tl.where(
                    owns, (local_split + 1) * blocks // splits, end_block)
                begin_split_idx = tl.where(owns, local_split,
                                           begin_split_idx)
                begin_is_split = tl.where(owns, splits > 1,
                                          begin_is_split)
                end_is_split = tl.where(owns, splits > 1, end_is_split)
                active |= owns
                split_prefix += splits
                eligible_prefix += eligible
        else:
            # Large graph buckets can have fewer CTA partitions than
            # requests. Fall back to contiguous global chunks, which may span
            # requests but still covers every KV block exactly once.
            global_begin = pid * payload
            global_end = tl.minimum(global_begin + payload, total_blocks)
            begin_req_prefix = 0
            end_req_prefix = 0
            begin_req_blocks = 1
            end_req_blocks = 1
            prefix = 0
            for batch in range(BATCH_SIZE):
                primary = tl.load(primary_topk_length + batch).to(tl.int32)
                extra = tl.load(extra_topk_length + batch).to(tl.int32)
                primary_blocks = tl.maximum(
                    (primary + TOPK_BLOCK_SIZE - 1) // TOPK_BLOCK_SIZE, 1)
                extra_blocks = (extra + TOPK_BLOCK_SIZE - 1) // TOPK_BLOCK_SIZE
                blocks = primary_blocks + extra_blocks
                next_prefix = prefix + blocks
                owns_begin = (global_begin >= prefix) & (global_begin <
                                                          next_prefix)
                owns_end = (global_end > prefix) & (global_end <= next_prefix)
                begin_req = tl.where(owns_begin, batch, begin_req)
                begin_block = tl.where(owns_begin, global_begin - prefix,
                                       begin_block)
                begin_req_prefix = tl.where(owns_begin, prefix,
                                            begin_req_prefix)
                begin_req_blocks = tl.where(owns_begin, blocks,
                                            begin_req_blocks)
                end_req = tl.where(owns_end, batch, end_req)
                end_block = tl.where(owns_end, global_end - prefix, end_block)
                end_req_prefix = tl.where(owns_end, prefix, end_req_prefix)
                end_req_blocks = tl.where(owns_end, blocks, end_req_blocks)
                prefix = next_prefix
            active = global_begin < total_blocks
            begin_split_idx = pid - begin_req_prefix // payload
            begin_is_split = (
                (begin_req_prefix + begin_req_blocks - 1) // payload
                > begin_req_prefix // payload)
            end_is_split = (
                (end_req_prefix + end_req_blocks - 1) // payload
                > end_req_prefix // payload)

        sched_ptr = scheduler_metadata + pid * stride_sched_p
        tl.store(sched_ptr, tl.where(active, begin_req, BATCH_SIZE))
        tl.store(sched_ptr + 1, tl.where(active, end_req, 0))
        tl.store(sched_ptr + 2, tl.where(active, begin_block, 0))
        tl.store(sched_ptr + 3, tl.where(active, end_block, 0))
        tl.store(sched_ptr + 4, tl.where(active, begin_split_idx, 0))
        tl.store(sched_ptr + 5, tl.where(active, begin_is_split, 0))
        tl.store(sched_ptr + 6, tl.where(active, end_is_split, 0))
        tl.store(sched_ptr + 7, 0)

    if pid <= BATCH_SIZE:
        cumulative_splits = 0
        prefix = 0
        eligible_prefix = 0
        for batch in range(BATCH_SIZE):
            primary = tl.load(primary_topk_length + batch).to(tl.int32)
            extra = tl.load(extra_topk_length + batch).to(tl.int32)
            primary_blocks = tl.maximum(
                (primary + TOPK_BLOCK_SIZE - 1) // TOPK_BLOCK_SIZE, 1)
            extra_blocks = (extra + TOPK_BLOCK_SIZE - 1) // TOPK_BLOCK_SIZE
            blocks = primary_blocks + extra_blocks
            if NUM_PARTITIONS >= BATCH_SIZE:
                request_base = tl.minimum(blocks, base_splits)
                remaining = NUM_PARTITIONS - base_split_total
                remaining = tl.minimum(remaining, extra_eligible)
                eligible = blocks > request_base
                gets_extra = eligible & (eligible_prefix < remaining)
                cumulative_splits += request_base + gets_extra
                eligible_prefix += eligible
            else:
                first_partition = prefix // payload
                last_partition = (prefix + blocks - 1) // payload
                cumulative_splits += last_partition - first_partition + 1
            prefix += blocks
            if pid == batch + 1:
                tl.store(num_splits + pid, cumulative_splits)
        if pid == 0:
            tl.store(num_splits, 0)


def build_rectangular_decode_split_scheduler(
    primary_topk_length: torch.Tensor,
    extra_topk_length: torch.Tensor,
    scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    topk_block_size: int = 64,
) -> None:
    """Build a graph-capturable split scheduler for ``[B,Q,H,D]`` decode.

    ``primary_topk_length`` and ``extra_topk_length`` are the common lengths
    for each request across its Q verifier rows. Invalid per-row entries must
    already be encoded as ``-1`` in the corresponding sparse-index tensors.
    """
    if primary_topk_length.dim() != 1 or extra_topk_length.dim() != 1:
        raise ValueError('rectangular scheduler lengths must be 1-D')
    if primary_topk_length.shape != extra_topk_length.shape:
        raise ValueError('primary and extra scheduler lengths must match')
    if scheduler_metadata.dim() != 2 or scheduler_metadata.size(1) != 8:
        raise ValueError('FlashMLA scheduler metadata must have shape [P,8]')
    batch_size = primary_topk_length.numel()
    if num_splits.shape != (batch_size + 1, ):
        raise ValueError('FlashMLA num_splits must have shape [B+1]')
    num_partitions = scheduler_metadata.size(0)
    if batch_size == 0 or num_partitions == 0:
        return

    grid = (max(num_partitions, batch_size + 1), )
    _build_rectangular_decode_split_scheduler_kernel[grid](
        primary_topk_length,
        extra_topk_length,
        scheduler_metadata,
        num_splits,
        stride_sched_p=scheduler_metadata.stride(0),
        BATCH_SIZE=batch_size,
        NUM_PARTITIONS=num_partitions,
        TOPK_BLOCK_SIZE=topk_block_size,
        num_warps=1,
    )


@triton.jit
def _build_decode_compressed_sparse_indices_kernel(
    logical_topk,
    block_offsets,
    row_is_padded,
    out,
    stride_l_b,
    stride_l_k,
    stride_bo_b,
    stride_bo_i,
    stride_o_b,
    stride_o_k,
    QUERY_LEN: tl.constexpr,
    TOPK: tl.constexpr,
    PADDED_TOPK: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    HAS_PADDED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // QUERY_LEN
    tile = tl.program_id(1)
    offs = tile * BLOCK + tl.arange(0, BLOCK)
    valid_col = offs < TOPK

    logical = tl.load(
        logical_topk + row * stride_l_b + offs * stride_l_k,
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
    if HAS_PADDED:
        padded = tl.load(row_is_padded + row)
        vals = tl.where(padded & (offs == 0), 0, vals)

    tl.store(
        out + row * stride_o_b + offs * stride_o_k,
        vals,
        mask=offs < PADDED_TOPK,
    )


def build_decode_compressed_sparse_indices(
    logical_topk: torch.Tensor,
    block_offsets: torch.Tensor,
    block_size: int,
    compress_ratio: int,
    query_len: int = 1,
    row_is_padded: torch.Tensor | None = None,
    block: int = 128,
) -> torch.Tensor:
    """Convert decode compressed logical topk to padded physical indices.

    This is the fused equivalent of the old decode updater plus final
    FlashMLA index padding. ``logical_topk`` is expected in
    ``[bsz, 1, topk]`` form from the V4 indexer.
    """
    if logical_topk.dim() == 2:
        logical_topk = logical_topk.unsqueeze(1)
    assert logical_topk.dim() == 3
    num_rows = logical_topk.size(0)
    if query_len < 1 or num_rows % query_len:
        raise ValueError(f'Invalid rectangular compressed-index shape: rows={num_rows}, query_len={query_len}.')
    if block_offsets.size(0) != num_rows // query_len:
        raise ValueError('block_offsets batch does not match compressed-index rows/query_len.')
    if row_is_padded is not None and row_is_padded.numel() != num_rows:
        raise ValueError('row_is_padded must have one entry per compressed-index row.')
    topk = logical_topk.size(-1)
    padded_topk = triton.cdiv(topk, block) * block
    out = torch.empty((num_rows, 1, padded_topk), dtype=torch.int32, device=logical_topk.device)
    if num_rows == 0:
        return out

    grid = (num_rows, triton.cdiv(padded_topk, block))
    _build_decode_compressed_sparse_indices_kernel[grid](
        logical_topk,
        block_offsets,
        row_is_padded,
        out,
        stride_l_b=logical_topk.stride(0),
        stride_l_k=logical_topk.stride(2),
        stride_bo_b=block_offsets.stride(0),
        stride_bo_i=block_offsets.stride(1),
        stride_o_b=out.stride(0),
        stride_o_k=out.stride(2),
        QUERY_LEN=query_len,
        TOPK=topk,
        PADDED_TOPK=padded_topk,
        NUM_BLOCKS=block_offsets.size(1),
        BLOCK_SIZE=block_size,
        COMPRESS_RATIO=compress_ratio,
        HAS_PADDED=row_is_padded is not None,
        BLOCK=block,
    )
    return out


def build_rectangular_decode_compressed_sparse_indices(
    logical_topk: torch.Tensor,
    block_offsets: torch.Tensor,
    block_size: int,
    compress_ratio: int,
    query_len: int,
    row_is_padded: torch.Tensor | None = None,
    block: int = 128,
) -> torch.Tensor:
    """Map request-major rectangular logical top-k rows to paged indices."""
    return build_decode_compressed_sparse_indices(
        logical_topk,
        block_offsets,
        block_size,
        compress_ratio,
        query_len=query_len,
        row_is_padded=row_is_padded,
        block=block,
    )


@triton.jit
def _build_decode_prefix_compressed_sparse_indices_kernel(
    row_lengths,
    block_offsets,
    row_is_padded,
    out,
    stride_bo_b,
    stride_bo_i,
    stride_o_b,
    stride_o_k,
    QUERY_LEN: tl.constexpr,
    INPUT_IS_TOKEN_LENGTH: tl.constexpr,
    TOPK: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    HAS_PADDED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // QUERY_LEN
    tile = tl.program_id(1)
    offs = tile * BLOCK + tl.arange(0, BLOCK)

    row_topk = tl.load(row_lengths + row).to(tl.int32)
    if INPUT_IS_TOKEN_LENGTH:
        row_topk = row_topk // COMPRESS_RATIO
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
    if HAS_PADDED:
        padded = tl.load(row_is_padded + row)
        vals = tl.where(padded & (offs == 0), 0, vals)
    tl.store(out + row * stride_o_b + offs * stride_o_k, vals, mask=offs < TOPK)


def build_decode_prefix_compressed_sparse_indices(
    num_compressed: torch.Tensor,
    block_offsets: torch.Tensor,
    block_size: int,
    compress_ratio: int,
    max_topk: int | None = None,
    query_len: int = 1,
    input_is_token_length: bool = False,
    row_is_padded: torch.Tensor | None = None,
    block: int = 128,
) -> torch.Tensor:
    """Build physical padded decode indices for full-prefix compressed KV."""
    if max_topk is None:
        max_topk = max(block_offsets.size(1) * block_size // compress_ratio, 1)
    padded_topk = triton.cdiv(max_topk, block) * block
    num_rows = num_compressed.numel()
    if query_len < 1 or num_rows % query_len:
        raise ValueError(f'Invalid rectangular prefix-index shape: rows={num_rows}, query_len={query_len}.')
    if block_offsets.size(0) != num_rows // query_len:
        raise ValueError('block_offsets batch does not match prefix-index rows/query_len.')
    if row_is_padded is not None and row_is_padded.numel() != num_rows:
        raise ValueError('row_is_padded must have one entry per prefix-index row.')
    out = torch.empty((num_rows, 1, padded_topk), dtype=torch.int32, device=num_compressed.device)
    if num_rows == 0:
        return out

    grid = (num_rows, triton.cdiv(padded_topk, block))
    _build_decode_prefix_compressed_sparse_indices_kernel[grid](
        num_compressed,
        block_offsets,
        row_is_padded,
        out,
        stride_bo_b=block_offsets.stride(0),
        stride_bo_i=block_offsets.stride(1),
        stride_o_b=out.stride(0),
        stride_o_k=out.stride(2),
        QUERY_LEN=query_len,
        INPUT_IS_TOKEN_LENGTH=input_is_token_length,
        TOPK=padded_topk,
        NUM_BLOCKS=block_offsets.size(1),
        BLOCK_SIZE=block_size,
        COMPRESS_RATIO=compress_ratio,
        HAS_PADDED=row_is_padded is not None,
        BLOCK=block,
    )
    return out


def build_rectangular_decode_prefix_compressed_sparse_indices(
    token_total_lens: torch.Tensor,
    block_offsets: torch.Tensor,
    block_size: int,
    compress_ratio: int,
    query_len: int,
    max_topk: int | None = None,
    row_is_padded: torch.Tensor | None = None,
    block: int = 128,
) -> torch.Tensor:
    """Build full-prefix compressed indices for packed verifier rows."""
    return build_decode_prefix_compressed_sparse_indices(
        token_total_lens,
        block_offsets,
        block_size,
        compress_ratio,
        max_topk=max_topk,
        query_len=query_len,
        input_is_token_length=True,
        row_is_padded=row_is_padded,
        block=block,
    )


@triton.jit(do_not_specialize=['total_topk', 'padded_topk', 'stride_c_t', 'stride_o_t'])
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
    total_topk,
    padded_topk,
    stride_c_t,
    stride_c_k,
    stride_o_t,
    stride_o_k,
    WINDOW_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    HAS_COMPRESS: tl.constexpr,
    HAS_COMPRESS_TOPK: tl.constexpr,
    CAUSAL: tl.constexpr,
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

    window_start_abs = tl.maximum(seq_start - WINDOW_SIZE, 0)
    if CAUSAL:
        num_vis = tl.minimum(abs_pos + 1, WINDOW_SIZE)
        first_vis_abs = tl.maximum(abs_pos - WINDOW_SIZE + 1, 0)
        first_flat_pos = first_vis_abs - window_start_abs
    else:
        # DSpark draft rows share a block-wide mask: trailing prefix window plus
        # every query row, including future mask rows. The flattened KV buffer
        # starts at ``window_start_abs`` and already appends the whole query.
        query_len = tl.load(total_lens + seq).to(tl.int64) - seq_start
        num_vis = tl.minimum(seq_start, WINDOW_SIZE) + query_len
        first_flat_pos = 0

    in_window = offs < WINDOW_SIZE
    window_valid = offs < num_vis
    vals = tl.where(window_valid, first_flat_pos + offs, -1).to(tl.int32)

    if HAS_COMPRESS:
        comp_col = tl.maximum(offs - WINDOW_SIZE, 0)
        in_compress = (offs >= WINDOW_SIZE) & (offs < total_topk)
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
            if CAUSAL:
                num_compressed = (abs_pos + 1) // COMPRESS_RATIO
                row_topk_length = tl.minimum(WINDOW_SIZE + num_compressed,
                                             padded_topk)
            else:
                num_compressed = tl.load(total_lens + seq).to(tl.int32) // COMPRESS_RATIO
                row_topk_length = tl.minimum(num_vis + num_compressed,
                                             padded_topk)
            tl.store(topk_length + token, row_topk_length.to(tl.int32))
    elif tile == 0:
        if CAUSAL:
            row_topk_length = WINDOW_SIZE
        else:
            row_topk_length = num_vis
        tl.store(topk_length + token, row_topk_length.to(tl.int32))

    vals = tl.where((vals >= 0) & (offs < total_topk), vals + cu, -1)
    tl.store(out + token * stride_o_t + offs * stride_o_k,
             vals, mask=offs < padded_topk)


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
    causal: bool = True,
    max_q_seqlen: int = 1,
    block: int = 128,
):
    """Build padded prefill sparse indices and lengths for FlashMLA.

    This fuses the previous window-topk construction, compressed-prefix construction, global flat-KV offset application,
    head-dimension unsqueeze, padding, and topk-length construction into one Triton launch.
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

    if not causal and has_compress:
        raise NotImplementedError('Non-causal V4 sparse prefill does not support compressed KV.')
    query_width = 0 if causal else max_q_seqlen
    total_topk = window_size + query_width + compress_width
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
        total_topk,
        padded_topk,
        stride_c_t=stride_c_t,
        stride_c_k=stride_c_k,
        stride_o_t=out.stride(0),
        stride_o_k=out.stride(2),
        WINDOW_SIZE=window_size,
        COMPRESS_RATIO=compress_ratio,
        HAS_COMPRESS=has_compress,
        HAS_COMPRESS_TOPK=has_compress_topk,
        CAUSAL=causal,
        BLOCK=block,
    )
    return out, topk_length
