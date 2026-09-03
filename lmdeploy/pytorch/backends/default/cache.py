# Copyright (c) OpenMMLab. All rights reserved.

from itertools import groupby
from operator import index as as_index

import torch

from ...engine.cache_engine.layout import (
    CacheAllocation,
    CompositeBlockCacheLayout,
    ContiguousBlockCacheLayout,
    PackedBlockCacheLayout,
    PackedStateCacheLayout,
    RowBlockCacheLayout,
)
from ..cache import BlockCacheCopy, CacheBackend


class TorchBlockCacheCopy(BlockCacheCopy):
    """Bounded tensor fallback for allocation-owned logical-block copies."""

    _TARGET_WORKSPACE_BYTES = 64 * 1024**2

    def __init__(self, allocation: CacheAllocation, num_logical_blocks: int,
                 pages_per_block: int, blocks_per_chunk: int):
        super().__init__(allocation, num_logical_blocks, pages_per_block)
        self.blocks_per_chunk = blocks_per_chunk
        self._logical_pools = tuple(
            (pool.tensor.unflatten(pool.entry_axis,
                                   (self.num_logical_blocks, self.pages_per_block)), pool.entry_axis)
            for pool in self.pools)
        self._workspaces: tuple[torch.Tensor, ...] | None = None

    @classmethod
    def build(cls, allocation: CacheAllocation, num_logical_blocks: int,
              pages_per_block: int) -> 'TorchBlockCacheCopy':
        """Choose a chunk size under one aggregate workspace budget."""
        num_logical_blocks = as_index(num_logical_blocks)
        pages_per_block = as_index(pages_per_block)
        if num_logical_blocks == 0:
            blocks_per_chunk = 1
        else:
            bytes_per_block = sum(pool.nbytes // num_logical_blocks for pool in allocation.pools)
            if bytes_per_block == 0:
                blocks_per_chunk = num_logical_blocks
            else:
                blocks_per_chunk = max(1, cls._TARGET_WORKSPACE_BYTES // bytes_per_block)
                blocks_per_chunk = min(blocks_per_chunk, num_logical_blocks)
        return cls(allocation, num_logical_blocks, pages_per_block, blocks_per_chunk)

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
                src_chunk = src_block_offsets[start:end]
                dst_chunk = dst_block_offsets[start:end]
                torch.index_select(logical_pool, entry_axis, src_chunk, out=chunk_workspace)
                logical_pool.index_copy_(entry_axis, dst_chunk, chunk_workspace)


class DefaultCacheBackend(CacheBackend):
    """Build the default cache storage layouts."""

    @classmethod
    def build_block_layout(cls, tensor_specs, num_layers: int):
        """Select the default block-cache packing."""
        tensor_specs = tuple(tensor_specs)
        if not tensor_specs:
            return PackedBlockCacheLayout(tensor_specs, num_layers=num_layers)

        def layout_kind(spec):
            if spec.per_row_contiguous:
                return 'contiguous'
            if spec.consumer_rows is not None:
                return 'rows'
            return 'packed'

        layouts = []
        for kind, group in groupby(tensor_specs, key=layout_kind):
            group = tuple(group)
            if kind == 'contiguous':
                layout = ContiguousBlockCacheLayout(group, num_layers=num_layers)
            elif kind == 'rows':
                layout = RowBlockCacheLayout(group)
            else:
                layout = PackedBlockCacheLayout(group, num_layers=num_layers)
            layouts.append(layout)

        if len(layouts) == 1:
            return layouts[0]
        return CompositeBlockCacheLayout(tuple(layouts))

    @classmethod
    def build_state_layout(cls, tensor_specs):
        """Select the default packed state-cache layout."""
        return PackedStateCacheLayout(tuple(tensor_specs))

    @classmethod
    def build_block_copy(cls, allocation: CacheAllocation, num_logical_blocks: int,
                         pages_per_block: int):
        """Build the portable allocation-aware copy fallback."""
        return TorchBlockCacheCopy.build(allocation, num_logical_blocks, pages_per_block)
