# Copyright (c) OpenMMLab. All rights reserved.

from collections.abc import Sequence

import torch

from ..cache_block_copy import CacheBlockCopyBuildSpec, CacheBlockCopyImpl

# Bound the total persistent gather workspace across all packed pools. One
# logical block remains the irreducible unit, so an unusually large block may
# exceed this target.
_TARGET_WORKSPACE_BYTES = 64 * 1024**2


class DefaultCacheBlockCopyImpl(CacheBlockCopyImpl):
    """Batched torch fallback for copying opaque packed logical blocks."""

    def __init__(self,
                 packed_caches: Sequence[torch.Tensor],
                 num_logical_blocks: int,
                 pages_per_block: int,
                 blocks_per_chunk: int):
        packed_caches = tuple(packed_caches)
        self.blocks_per_chunk = blocks_per_chunk
        self._logical_caches = tuple(
            packed_cache.unflatten(-2, (num_logical_blocks, pages_per_block))
            for packed_cache in packed_caches)
        self._workspaces: tuple[torch.Tensor, ...] | None = None

    def forward(self, src_block_offsets: torch.Tensor, dst_block_offsets: torch.Tensor) -> None:
        """Copy logical blocks with bounded gather/scatter batches."""
        num_blocks = src_block_offsets.numel()
        if len(self._logical_caches) == 0 or num_blocks == 0:
            return

        if self._workspaces is None:
            self._workspaces = tuple(
                torch.empty((*cache.shape[:-3], self.blocks_per_chunk, *cache.shape[-2:]),
                            dtype=cache.dtype,
                            device=cache.device) for cache in self._logical_caches)

        for logical_cache, workspace in zip(self._logical_caches, self._workspaces):
            for start in range(0, num_blocks, self.blocks_per_chunk):
                end = min(start + self.blocks_per_chunk, num_blocks)
                chunk_blocks = end - start
                src_chunk = src_block_offsets[start:end]
                dst_chunk = dst_block_offsets[start:end]
                chunk_workspace = workspace.narrow(-3, 0, chunk_blocks)
                torch.index_select(logical_cache, -3, src_chunk, out=chunk_workspace)

                index_shape = [1] * logical_cache.dim()
                index_shape[-3] = chunk_blocks
                dst_index = dst_chunk.view(index_shape).expand_as(chunk_workspace)
                logical_cache.scatter_(-3, dst_index, chunk_workspace)


def build_cache_block_copy(spec: CacheBlockCopyBuildSpec) -> CacheBlockCopyImpl:
    """Build the batched torch logical-block copy fallback."""
    if spec.num_logical_blocks == 0:
        blocks_per_chunk = 1
    else:
        bytes_per_block = sum(cache.numel() * cache.element_size() // spec.num_logical_blocks
                              for cache in spec.packed_caches)
        if bytes_per_block == 0:
            blocks_per_chunk = spec.num_logical_blocks
        else:
            blocks_per_chunk = max(1, _TARGET_WORKSPACE_BYTES // bytes_per_block)
            blocks_per_chunk = min(blocks_per_chunk, spec.num_logical_blocks)
    return DefaultCacheBlockCopyImpl(packed_caches=spec.packed_caches,
                                     num_logical_blocks=spec.num_logical_blocks,
                                     pages_per_block=spec.pages_per_block,
                                     blocks_per_chunk=blocks_per_chunk)
