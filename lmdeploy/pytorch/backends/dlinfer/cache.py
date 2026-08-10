# Copyright (c) OpenMMLab. All rights reserved.
"""Contiguous cache layouts for dlinfer device backends."""

from dataclasses import dataclass

import torch

from ...engine.cache_engine.layout import CacheAllocation, CachePool
from ...engine.cache_engine.schema import CacheTensorSpec
from ..default.cache import DefaultCacheBackend


@dataclass(frozen=True)
class DlinferBlockCacheLayout:
    """Allocate every block-cache spec as one contiguous owning tensor."""

    tensor_specs: tuple[CacheTensorSpec, ...]
    num_layers: int

    def allocate(self, num_blocks: int, device: torch.device | str) -> CacheAllocation:
        """Realize contiguous cache tensors for a block count and device."""
        if len(self.tensor_specs) == 0:
            empty = torch.empty((self.num_layers, num_blocks, 0), dtype=torch.uint8, device=device)
            return CacheAllocation(pools=(CachePool(empty, entry_axis=1), ), tensor_views=())

        pools = []
        tensor_views = []
        for spec in self.tensor_specs:
            num_rows = spec.num_rows if spec.has_rows else self.num_layers
            cache = torch.zeros((num_rows, num_blocks, *spec.desc.shape),
                                dtype=spec.desc.dtype,
                                device=device)
            pools.append(CachePool(cache, entry_axis=1))
            tensor_views.append(cache)
        return CacheAllocation(pools=tuple(pools), tensor_views=tuple(tensor_views))


@dataclass(frozen=True)
class DlinferStateCacheLayout:
    """Allocate every state-cache spec with contiguous per-layer slots."""

    tensor_specs: tuple[CacheTensorSpec, ...]

    def allocate(self, num_caches: int, device: torch.device | str) -> CacheAllocation:
        """Realize independent contiguous state tensors for a slot count."""
        if len(self.tensor_specs) == 0 or num_caches == 0:
            empty = torch.empty((0, 0), dtype=torch.uint8, device=device)
            return CacheAllocation(pools=(CachePool(empty, entry_axis=0), ), tensor_views=())

        pools = []
        tensor_views = []
        for spec in self.tensor_specs:
            if spec.layer_rows is None:
                cache_shape = (num_caches, *spec.desc.shape)
                entry_axis = 0
            else:
                assert spec.desc.shape[0] == spec.num_rows
                cache_shape = (spec.num_rows, num_caches, *spec.desc.shape[1:])
                entry_axis = 1
            cache = torch.zeros(cache_shape, dtype=spec.desc.dtype, device=device)
            pools.append(CachePool(cache, entry_axis=entry_axis))
            tensor_views.append(cache)
        return CacheAllocation(pools=tuple(pools), tensor_views=tuple(tensor_views))


class DlinferCacheBackend(DefaultCacheBackend):
    """Build native dlinfer cache layouts.

    The presence of this provider is the feature-detection boundary for dlinfer versions that can skip their legacy
    CacheEngine monkey patches.
    """

    @classmethod
    def build_block_layout(cls, tensor_specs, num_layers: int):
        """Select independent contiguous block-cache tensors."""
        return DlinferBlockCacheLayout(tuple(tensor_specs), num_layers=num_layers)

    @classmethod
    def build_state_layout(cls, tensor_specs):
        """Select independent contiguous state-cache tensors."""
        return DlinferStateCacheLayout(tuple(tensor_specs))
