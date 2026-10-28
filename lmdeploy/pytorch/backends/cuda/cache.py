# Copyright (c) OpenMMLab. All rights reserved.
"""CUDA cache layouts and local cache primitives."""

import torch

from ...engine.cache_engine.layout import CacheAllocation
from ..cache import BlockCacheCopy
from ..default.cache import DefaultCacheBackend


class CudaBlockCacheCopy(BlockCacheCopy):
    """Triton logical-block copy across contiguous allocation pools."""

    def __init__(self, allocation: CacheAllocation, num_logical_blocks: int,
                 pages_per_block: int):
        super().__init__(allocation, num_logical_blocks, pages_per_block)
        if any(not pool.tensor.is_contiguous() for pool in self.pools):
            raise ValueError('CUDA block-cache copy requires contiguous allocation pools.')

    def copy(self, src_block_offsets: torch.Tensor, dst_block_offsets: torch.Tensor) -> None:
        """Copy every owning pool with one launch per pool."""
        from ...kernels.cuda.copy_cache import copy_cache_blocks

        for pool in self.pools:
            copy_cache_blocks(pool.tensor,
                              pool.entry_axis,
                              src_block_offsets,
                              dst_block_offsets,
                              self.pages_per_block)


class CudaCacheBackend(DefaultCacheBackend):
    """Select default layouts and CUDA-local cache primitives."""

    @classmethod
    def build_block_copy(cls, allocation: CacheAllocation, num_logical_blocks: int,
                         pages_per_block: int):
        """Build the Triton allocation-aware copy primitive."""
        return CudaBlockCacheCopy(allocation, num_logical_blocks, pages_per_block)

    @staticmethod
    def get_decode_save_mask(plan: torch.Tensor, history: torch.Tensor, query_lengths: torch.Tensor,
                             tokens: torch.Tensor, stop_pos: torch.Tensor, *, is_decoding: bool,
                             lookahead: int) -> torch.Tensor:
        """Validate candidates without reading acceptance back to the host."""
        from ...kernels.cuda.copy_cache import decode_save_mask
        return decode_save_mask(plan, history, query_lengths, tokens, stop_pos,
                                is_decoding=is_decoding, lookahead=lookahead)

    @staticmethod
    def copy_decode_slots(slot_tensors: tuple[tuple[torch.Tensor, int], ...],
                          plan: torch.Tensor, copy_mask: torch.Tensor) -> None:
        """Freeze selected checkpoint pools without allocating slot-sized
        temporaries."""
        from ...kernels.cuda.copy_cache import copy_cache_blocks
        for tensor, slot_axis in slot_tensors:
            copy_cache_blocks(tensor, slot_axis, plan[2], plan[3], 1, copy_mask)
