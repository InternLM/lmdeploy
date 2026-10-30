# Copyright (c) OpenMMLab. All rights reserved.
"""Copy scheduler-sized blocks in contiguous cache pools."""

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_cache_blocks_kernel(cache,
                              src_block_offsets,
                              dst_block_offsets,
                              copy_mask,
                              cache_stride_outer,
                              cache_stride_block,
                              bytes_per_block: tl.constexpr,
                              HAS_MASK: tl.constexpr,
                              BLOCK_BYTES: tl.constexpr):
    tile_id = tl.program_id(0)
    outer_id = tl.program_id(1)
    pair_id = tl.program_id(2)
    if HAS_MASK:
        if not tl.load(copy_mask + pair_id):
            return

    src_block = tl.load(src_block_offsets + pair_id).to(tl.int64)
    dst_block = tl.load(dst_block_offsets + pair_id).to(tl.int64)
    byte_offsets = tile_id * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
    mask = byte_offsets < bytes_per_block
    outer_offset = outer_id.to(tl.int64) * cache_stride_outer
    src_offsets = outer_offset + src_block * cache_stride_block + byte_offsets
    dst_offsets = outer_offset + dst_block * cache_stride_block + byte_offsets
    values = tl.load(cache + src_offsets, mask=mask)
    tl.store(cache + dst_offsets, values, mask=mask)


def copy_cache_blocks(cache: torch.Tensor,
                      entry_axis: int,
                      src_block_offsets: torch.Tensor,
                      dst_block_offsets: torch.Tensor,
                      pages_per_block: int,
                      copy_mask: torch.Tensor | None = None) -> None:
    """Copy complete logical blocks on the current CUDA stream."""
    num_pairs = src_block_offsets.numel()
    if num_pairs == 0 or cache.numel() == 0:
        return
    if not cache.is_contiguous():
        raise ValueError('Cache pool must be contiguous.')

    physical_pages = cache.size(entry_axis)
    outer_size = 1
    for size in cache.shape[:entry_axis]:
        outer_size *= size
    bytes_per_page = cache.element_size()
    for size in cache.shape[entry_axis + 1:]:
        bytes_per_page *= size

    cache_bytes = cache.view(torch.uint8).reshape(outer_size, physical_pages, bytes_per_page)
    bytes_per_block = bytes_per_page * pages_per_block
    block_bytes = min(4096, triton.next_power_of_2(bytes_per_block))
    num_warps = 8 if block_bytes >= 2048 else 4
    num_tiles = triton.cdiv(bytes_per_block, block_bytes)
    grid = (num_tiles, outer_size, num_pairs)
    _copy_cache_blocks_kernel[grid](
        cache_bytes,
        src_block_offsets,
        dst_block_offsets,
        copy_mask,
        cache_bytes.stride(0),
        cache_bytes.stride(1) * pages_per_block,
        bytes_per_block=bytes_per_block,
        HAS_MASK=copy_mask is not None,
        BLOCK_BYTES=block_bytes,
        num_warps=num_warps,
    )


@triton.jit
def _decode_save_mask_kernel(plan, history, query_lengths, tokens, stop_pos, mask,
                             NUM_COPIES: tl.constexpr, NUM_OUTPUTS: tl.constexpr,
                             IS_DECODING: tl.constexpr, LOOKAHEAD: tl.constexpr,
                             BLOCK: tl.constexpr):
    index = tl.program_id(0)
    row = tl.load(plan + index)
    boundary = tl.load(plan + NUM_COPIES + index)
    start = tl.load(history + row)
    query_length = tl.load(query_lengths + row)
    offsets = tl.arange(0, BLOCK)
    ids = tl.load(tokens + row * NUM_OUTPUTS + offsets, offsets < NUM_OUTPUTS, other=-1)
    accepted = tl.sum((ids >= 0).to(tl.int32), 0)
    stop = tl.load(stop_pos + row)
    accepted = tl.where(stop >= 0, tl.minimum(accepted, stop + 1), accepted)
    computed = start + (accepted if IS_DECODING else query_length)
    valid = ((accepted > 0) & (boundary > start) & (boundary <= computed)
             & (boundary <= start + query_length) & (boundary + LOOKAHEAD <= computed + 1))
    if not IS_DECODING:
        # Prefill exposes one final state, not every intermediate token state.
        valid = valid & (boundary == computed)
    tl.store(mask + index, valid)


def decode_save_mask(plan: torch.Tensor, history: torch.Tensor, query_lengths: torch.Tensor,
                     tokens: torch.Tensor, stop_pos: torch.Tensor, *, is_decoding: bool,
                     lookahead: int) -> torch.Tensor:
    """Validate candidate states on device, including stop-truncated MTP
    acceptance."""
    num_copies = plan.size(1)
    num_outputs = tokens.numel() // history.numel()
    mask = torch.empty(num_copies, dtype=torch.bool, device=plan.device)
    _decode_save_mask_kernel[(num_copies,)](
        plan, history, query_lengths, tokens, stop_pos, mask,
        NUM_COPIES=num_copies, NUM_OUTPUTS=num_outputs, IS_DECODING=is_decoding,
        LOOKAHEAD=lookahead, BLOCK=triton.next_power_of_2(num_outputs))
    return mask
