# Copyright (c) OpenMMLab. All rights reserved.

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_packed_cache_kernel(cache,
                              src_block_offsets,
                              dst_block_offsets,
                              cache_stride_row,
                              cache_stride_block,
                              words_per_block: tl.constexpr,
                              BLOCK_WORDS: tl.constexpr):
    tile_id = tl.program_id(0)
    row_id = tl.program_id(1)
    pair_id = tl.program_id(2)

    src_block = tl.load(src_block_offsets + pair_id).to(tl.int64)
    dst_block = tl.load(dst_block_offsets + pair_id).to(tl.int64)
    word_offsets = tile_id * BLOCK_WORDS + tl.arange(0, BLOCK_WORDS)
    mask = word_offsets < words_per_block
    row_offset = row_id.to(tl.int64) * cache_stride_row
    src_offsets = row_offset + src_block * cache_stride_block + word_offsets
    dst_offsets = row_offset + dst_block * cache_stride_block + word_offsets
    values = tl.load(cache + src_offsets, mask=mask)
    tl.store(cache + dst_offsets, values, mask=mask)


def copy_packed_cache(packed_cache: torch.Tensor,
                      src_block_offsets: torch.Tensor,
                      dst_block_offsets: torch.Tensor,
                      pages_per_block: int) -> None:
    """Copy complete packed logical blocks with one Triton launch.

    Cache layout, logical/page geometry, and copy-plan semantics are internal backend invariants established before this
    trusted launcher is called.
    """
    num_pairs = src_block_offsets.numel()
    if num_pairs == 0:
        return

    physical_pages = packed_cache.size(-2)
    num_logical_blocks = physical_pages // pages_per_block
    words_per_block = packed_cache.size(-1) * pages_per_block // torch.int64.itemsize
    block_words = min(1024, triton.next_power_of_2(words_per_block))
    num_warps = 8 if block_words >= 1024 else 4
    num_rows = packed_cache.numel() // (physical_pages * packed_cache.size(-1))
    packed_words = packed_cache.view(torch.int64).reshape(num_rows, num_logical_blocks, words_per_block)
    num_tiles = triton.cdiv(words_per_block, block_words)
    grid = (num_tiles, num_rows, num_pairs)
    _copy_packed_cache_kernel[grid](
        packed_words,
        src_block_offsets,
        dst_block_offsets,
        packed_words.stride(0),
        packed_words.stride(1),
        words_per_block=words_per_block,
        BLOCK_WORDS=block_words,
        num_warps=num_warps,
    )
