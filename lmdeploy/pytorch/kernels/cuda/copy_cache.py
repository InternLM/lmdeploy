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
