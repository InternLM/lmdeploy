# Copyright (c) OpenMMLab. All rights reserved.
"""Contiguous cache layouts for dlinfer device backends."""

from dataclasses import dataclass

import torch

from ...engine.cache_engine.layout import CacheAllocation, CachePool
from ...engine.cache_engine.schema import CacheResource
from ..default.cache import DefaultCacheBackend


@dataclass(frozen=True)
class DlinferBlockCacheLayout:
    """Allocate every block resource as one contiguous owning tensor."""

    resources: tuple[CacheResource, ...]
    num_layers: int

    def allocate(self, num_blocks: int, device: torch.device | str) -> CacheAllocation:
        """Realize contiguous resource tensors for a block count and device."""
        if len(self.resources) == 0:
            empty = torch.empty((self.num_layers, num_blocks, 0), dtype=torch.uint8, device=device)
            return CacheAllocation(pools=(CachePool(empty, entry_axis=1), ), caches=())

        pools = []
        caches = []
        for resource in self.resources:
            num_rows = resource.num_rows if resource.has_rows else self.num_layers
            cache = torch.zeros((num_rows, num_blocks, *resource.desc.shape),
                                dtype=resource.desc.dtype,
                                device=device)
            pools.append(CachePool(cache, entry_axis=1))
            caches.append(cache)
        return CacheAllocation(pools=tuple(pools), caches=tuple(caches))


@dataclass(frozen=True)
class DlinferStateCacheLayout:
    """Allocate every state resource with contiguous per-layer state slots."""

    resources: tuple[CacheResource, ...]

    def allocate(self, num_caches: int, device: torch.device | str) -> CacheAllocation:
        """Realize independent contiguous state tensors for a slot count."""
        if len(self.resources) == 0 or num_caches == 0:
            empty = torch.empty((0, 0), dtype=torch.uint8, device=device)
            return CacheAllocation(pools=(CachePool(empty, entry_axis=0), ), caches=())

        pools = []
        caches = []
        for resource in self.resources:
            if resource.layer_rows is None:
                cache_shape = (num_caches, *resource.desc.shape)
                entry_axis = 0
            else:
                assert resource.desc.shape[0] == resource.num_rows
                cache_shape = (resource.num_rows, num_caches, *resource.desc.shape[1:])
                entry_axis = 1
            cache = torch.zeros(cache_shape, dtype=resource.desc.dtype, device=device)
            pools.append(CachePool(cache, entry_axis=entry_axis))
            caches.append(cache)
        return CacheAllocation(pools=tuple(pools), caches=tuple(caches))


class DlinferCacheBackend(DefaultCacheBackend):
    """Build native dlinfer cache layouts.

    The presence of this provider is the feature-detection boundary for dlinfer versions that can skip their legacy
    CacheEngine monkey patches.
    """

    @classmethod
    def build_block_layout(cls, resources, num_layers: int):
        """Select independent contiguous block-resource tensors."""
        return DlinferBlockCacheLayout(tuple(resources), num_layers=num_layers)

    @classmethod
    def build_state_layout(cls, resources):
        """Select independent contiguous state-resource tensors."""
        return DlinferStateCacheLayout(tuple(resources))
