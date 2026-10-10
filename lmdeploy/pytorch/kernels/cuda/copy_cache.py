# Copyright (c) OpenMMLab. All rights reserved.
"""Copy scheduler-sized blocks in contiguous cache pools."""

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_cache_blocks_kernel(cache,
                              src_block_offsets,
                              dst_block_offsets,
                              cache_stride_outer,
                              cache_stride_block,
                              bytes_per_block: tl.constexpr,
                              BLOCK_BYTES: tl.constexpr):
    tile_id = tl.program_id(0)
    outer_id = tl.program_id(1)
    pair_id = tl.program_id(2)

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
                      pages_per_block: int) -> None:
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
        cache_bytes.stride(0),
        cache_bytes.stride(1) * pages_per_block,
        bytes_per_block=bytes_per_block,
        BLOCK_BYTES=block_bytes,
        num_warps=num_warps,
    )


@triton.jit
def _restore_state_rows_kernel(cache, before, slots, rows, rejected,
                               stride_layer, stride_slot, stride_row,
                               BATCH: tl.constexpr, WIDTH: tl.constexpr,
                               ROW_BYTES: tl.constexpr, BLOCK: tl.constexpr):
    tiles: tl.constexpr = tl.cdiv(ROW_BYTES, BLOCK)
    row_id = tl.program_id(0) // tiles
    layer = row_id // (BATCH * WIDTH)
    batch = row_id // WIDTH % BATCH
    column = row_id % WIDTH
    slot = tl.load(slots + batch).to(tl.int64)
    restore = tl.load(rejected + batch * WIDTH + column)
    if slot >= 0 and restore:
        row = tl.load(rows + batch * WIDTH + column).to(tl.int64)
        offsets = (tl.program_id(0) % tiles) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < ROW_BYTES
        value = tl.load(before + row_id.to(tl.int64) * ROW_BYTES + offsets, mask=mask)
        dst = layer.to(tl.int64) * stride_layer + slot * stride_slot + row * stride_row
        tl.store(cache + dst + offsets, value, mask=mask)


def restore_state_rows(cache: torch.Tensor, before: torch.Tensor,
                       slots: torch.Tensor, rows: torch.Tensor,
                       rejected: torch.Tensor) -> None:
    """Restore rejected V4 rows, ignoring padded slots without host selection.

    Cache layout is [layers, slots, rows, ...], with contiguous row payloads.
    Snapshots are contiguous [layers, batch, width, ...]. Byte copies also
    support FP8 without conversion or loss of bit patterns.
    """
    batch, width = rows.shape
    if not batch or not width or not cache.numel():
        return
    cache_bytes = cache.view(torch.uint8)
    before_bytes = before.contiguous().view(torch.uint8)
    row_bytes = before_bytes.numel() // (cache.size(0) * batch * width)
    block = min(1024, triton.next_power_of_2(row_bytes))
    _restore_state_rows_kernel[(triton.cdiv(row_bytes, block) * cache.size(0) * batch * width,)](
        cache_bytes, before_bytes, slots.contiguous(), rows.contiguous(), rejected.contiguous(),
        *cache_bytes.stride()[:3], batch, width, row_bytes, block)
