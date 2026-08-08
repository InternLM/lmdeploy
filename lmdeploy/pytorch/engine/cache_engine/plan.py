# Copyright (c) OpenMMLab. All rights reserved.
"""Finalized block-cache geometry, access metadata, and layout."""

from dataclasses import dataclass

import torch

from .layout import BlockCacheLayout, CacheAllocation
from .schema import CacheResource, layer_maps_from_resources


@dataclass(frozen=True)
class BlockCachePlan:
    """Reuse one finalized block-cache layout across sizing and allocation."""

    resources: tuple[CacheResource, ...]
    layout: BlockCacheLayout
    kernel_blocks_per_logical_block: int

    def __post_init__(self):
        if self.kernel_blocks_per_logical_block <= 0:
            raise ValueError('kernel blocks per logical block must be positive.')

    @property
    def cache_names(self) -> tuple[str, ...]:
        """Return resource names in model-facing cache order."""
        return tuple(resource.name for resource in self.resources)

    @property
    def layer_maps(self) -> dict[str, dict[int, int]]:
        """Return global-layer to compact-row mappings."""
        return layer_maps_from_resources(self.resources)

    @property
    def legacy_cache_indices(self) -> tuple[int, ...]:
        """Return resources exposed through the legacy per-layer cache
        tuple."""
        return tuple(index for index, resource in enumerate(self.resources) if resource.layer_rows is None)

    def allocate(self, num_logical_blocks: int, device: torch.device | str) -> CacheAllocation:
        """Realize a logical block count through the selected physical
        layout."""
        num_kernel_blocks = num_logical_blocks * self.kernel_blocks_per_logical_block
        return self.layout.allocate(num_blocks=num_kernel_blocks, device=device)

    @property
    def logical_block_nbytes(self) -> int:
        """Return owning storage bytes required by one logical block."""
        return self.allocate(num_logical_blocks=1, device='meta').nbytes
