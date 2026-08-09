# Copyright (c) OpenMMLab. All rights reserved.
"""Finalized block-cache geometry, access metadata, and layout."""

from dataclasses import dataclass

import torch

from .layout import BlockCacheLayout, CacheAllocation
from .schema import CacheResource


@dataclass(frozen=True)
class BlockCachePlan:
    """Reuse one finalized block-cache layout across sizing and allocation."""

    resources: tuple[CacheResource, ...]
    layout: BlockCacheLayout
    kernel_blocks_per_logical_block: int

    def __post_init__(self):
        if self.kernel_blocks_per_logical_block <= 0:
            raise ValueError('kernel blocks per logical block must be positive.')

        row_kind_by_name = {}
        row_ids_by_name = {}
        for resource in self.resources:
            if resource.consumer_rows is not None:
                row_kind = 'consumer'
                row_ids = resource.consumer_rows
            elif resource.layer_rows is not None:
                row_kind = 'layer'
                row_ids = resource.layer_rows.layer_ids
            else:
                row_kind = 'plain'
                row_ids = ()

            existing_kind = row_kind_by_name.get(resource.name)
            if existing_kind is not None:
                if row_kind == 'plain' or existing_kind != row_kind:
                    raise ValueError(
                        f'Block cache {resource.name} cannot mix {existing_kind} and {row_kind} resources.')
            else:
                row_kind_by_name[resource.name] = row_kind

            seen_rows = row_ids_by_name.setdefault(resource.name, set())
            overlap = seen_rows.intersection(row_ids)
            if overlap:
                row = min(overlap)
                raise ValueError(f'Block cache {resource.name} row {row} belongs to multiple resources.')
            seen_rows.update(row_ids)

        for name, row_kind in row_kind_by_name.items():
            if row_kind != 'consumer':
                continue
            rows = row_ids_by_name[name]
            if rows != set(range(len(rows))):
                raise ValueError(f'Block cache {name} consumer rows must be contiguous from zero.')

    @property
    def cache_names(self) -> tuple[str, ...]:
        """Return resource names in model-facing cache order."""
        return tuple(resource.name for resource in self.resources)

    @property
    def legacy_cache_indices(self) -> tuple[int, ...]:
        """Return resources exposed through the legacy per-layer cache
        tuple."""
        return tuple(index for index, resource in enumerate(self.resources) if not resource.has_rows)

    def allocate(self, num_logical_blocks: int, device: torch.device | str) -> CacheAllocation:
        """Realize a logical block count through the selected physical
        layout."""
        num_kernel_blocks = num_logical_blocks * self.kernel_blocks_per_logical_block
        return self.layout.allocate(num_blocks=num_kernel_blocks, device=device)

    @property
    def logical_block_nbytes(self) -> int:
        """Return owning storage bytes required by one logical block."""
        return self.allocate(num_logical_blocks=1, device='meta').nbytes
