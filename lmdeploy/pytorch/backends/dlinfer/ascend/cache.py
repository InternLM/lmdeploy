# Copyright (c) OpenMMLab. All rights reserved.
"""Ascend cache layouts and local cache primitives."""

import torch

from ....engine.cache_engine.layout import CacheAllocation
from ...default.cache import TorchBlockCacheCopy
from ..cache import DlinferCacheBackend


class AscendBlockCacheCopy(TorchBlockCacheCopy):
    """Block-view copy primitive for torch_npu allocation pools.

    torch_npu's ``index_copy_`` materializes a pool-sized temporary even for
    contiguous pools, which OOMs on production-size pools; plain ``copy_``
    between per-block views stays bounded by a single block. Chunk sizing
    and workspace handling are inherited from ``TorchBlockCacheCopy``.
    """

    def copy(self, src_block_offsets: torch.Tensor, dst_block_offsets: torch.Tensor) -> None:
        """Gather then scatter complete logical blocks for every pool."""
        num_blocks = src_block_offsets.numel()
        if num_blocks == 0:
            return

        if self._workspaces is None:
            workspaces = []
            for logical_pool, entry_axis in self._logical_pools:
                workspace_shape = list(logical_pool.shape)
                workspace_shape[entry_axis] = self.blocks_per_chunk
                workspaces.append(logical_pool.new_empty(workspace_shape))
            self._workspaces = tuple(workspaces)

        for (logical_pool, entry_axis), workspace in zip(self._logical_pools, self._workspaces):
            for start in range(0, num_blocks, self.blocks_per_chunk):
                end = min(start + self.blocks_per_chunk, num_blocks)
                chunk_blocks = end - start
                chunk_workspace = workspace.narrow(entry_axis, 0, chunk_blocks)
                self._gather_blocks(logical_pool, entry_axis, src_block_offsets[start:end],
                                    chunk_workspace)
                self._scatter_blocks(logical_pool, entry_axis, dst_block_offsets[start:end],
                                     chunk_workspace)

    @staticmethod
    def _gather_blocks(pool: torch.Tensor, entry_axis: int, block_offsets: torch.Tensor,
                       workspace: torch.Tensor) -> None:
        """Copy blocks out of ``pool`` into consecutive workspace slots."""
        for i, offset in enumerate(block_offsets.tolist()):
            workspace.select(entry_axis, i).copy_(pool.select(entry_axis, offset))

    @staticmethod
    def _scatter_blocks(pool: torch.Tensor, entry_axis: int, block_offsets: torch.Tensor,
                        workspace: torch.Tensor) -> None:
        """Copy consecutive workspace slots back into ``pool`` blocks."""
        for i, offset in enumerate(block_offsets.tolist()):
            pool.select(entry_axis, offset).copy_(workspace.select(entry_axis, i))


class AscendCacheBackend(DlinferCacheBackend):
    """Dlinfer contiguous layouts with Ascend-local block-copy primitives."""

    @classmethod
    def build_block_copy(cls, allocation: CacheAllocation, num_logical_blocks: int,
                         pages_per_block: int):
        """Build the block-view allocation-aware copy primitive."""
        return AscendBlockCacheCopy.build(allocation, num_logical_blocks, pages_per_block)
