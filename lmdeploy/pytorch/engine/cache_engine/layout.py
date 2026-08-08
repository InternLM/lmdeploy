# Copyright (c) OpenMMLab. All rights reserved.
"""Default physical cache allocation and owning-pool metadata."""

from dataclasses import dataclass
from typing import Protocol

import torch

from .schema import CacheResource


class BlockCacheLayout(Protocol):
    """Realize a physical block-cache allocation."""

    def allocate(self, num_blocks: int, device: torch.device | str) -> 'CacheAllocation':
        """Allocate physical kernel blocks on one device."""
        ...


@dataclass(frozen=True)
class CachePool:
    """Own one storage tensor and identify its cache-entry axis."""

    tensor: torch.Tensor
    entry_axis: int

    def __post_init__(self):
        if self.entry_axis < 0 or self.entry_axis >= self.tensor.dim():
            raise ValueError(f'entry_axis {self.entry_axis} is invalid for a {self.tensor.dim()}D cache pool.')

    @property
    def nbytes(self) -> int:
        """Return the number of owning storage bytes."""
        return self.tensor.numel() * self.tensor.element_size()


@dataclass(frozen=True)
class CacheAllocation:
    """Own cache pools and the typed resource views derived from them."""

    pools: tuple[CachePool, ...]
    caches: tuple[torch.Tensor, ...]

    @property
    def nbytes(self) -> int:
        """Count owning pools without double-counting resource views."""
        return sum(pool.nbytes for pool in self.pools)

    @property
    def legacy_pool(self) -> torch.Tensor | list[torch.Tensor]:
        """Return the temporary tensor-or-list owning-pool facade."""
        pool_tensors = [pool.tensor for pool in self.pools]
        if len(pool_tensors) == 1:
            return pool_tensors[0]
        return pool_tensors

    def as_legacy(self) -> tuple[torch.Tensor | list[torch.Tensor], list[torch.Tensor]]:
        """Return the temporary two-value allocation facade."""
        return self.legacy_pool, list(self.caches)

    def __iter__(self):
        """Preserve legacy ``mem_pool, caches = allocate_caches()`` use."""
        return iter(self.as_legacy())


@dataclass(frozen=True)
class PackedBlockCacheLayout:
    """Pack uniform-layer block resources into one owning pool."""

    resources: tuple[CacheResource, ...]
    num_layers: int

    def allocate(self, num_blocks: int, device: torch.device | str) -> CacheAllocation:
        """Realize the layout for a block count and device."""
        pool_size = sum(resource.desc.aligned_size for resource in self.resources)
        pool_tensor = torch.zeros((self.num_layers, num_blocks, pool_size), dtype=torch.uint8, device=device)

        caches = []
        offset = 0
        for resource in self.resources:
            desc = resource.desc
            cache = pool_tensor[:, :, offset:offset + desc.size].view(desc.dtype)
            cache = cache.view((self.num_layers, num_blocks, *desc.shape))
            caches.append(cache)
            offset += desc.aligned_size

        return CacheAllocation(pools=(CachePool(pool_tensor, entry_axis=1), ), caches=tuple(caches))


@dataclass(frozen=True)
class LayerRowBlockCacheLayout:
    """Give every layer-scoped block resource its own compact-row pool."""

    resources: tuple[CacheResource, ...]

    def __post_init__(self):
        if any(resource.layer_rows is None for resource in self.resources):
            raise ValueError('Layer-row block layouts require explicit layer rows for every resource.')

    def allocate(self, num_blocks: int, device: torch.device | str) -> CacheAllocation:
        """Realize the layout for a block count and device."""
        pools = []
        caches = []
        for resource in self.resources:
            desc = resource.desc
            pool_tensor = torch.zeros((resource.num_rows, num_blocks, desc.aligned_size),
                                      dtype=torch.uint8,
                                      device=device)
            cache = pool_tensor[:, :, :desc.size].view(desc.dtype)
            cache = cache.view((resource.num_rows, num_blocks, *desc.shape))
            pools.append(CachePool(pool_tensor, entry_axis=1))
            caches.append(cache)

        return CacheAllocation(pools=tuple(pools), caches=tuple(caches))


@dataclass(frozen=True)
class ContiguousBlockCacheLayout:
    """Give every resource an independent contiguous typed tensor."""

    resources: tuple[CacheResource, ...]
    num_layers: int

    def allocate(self, num_blocks: int, device: torch.device | str) -> CacheAllocation:
        """Allocate contiguous resource tensors for one block count."""
        pools = []
        caches = []
        for resource in self.resources:
            num_rows = self.num_layers if resource.layer_rows is None else resource.num_rows
            cache = torch.zeros((num_rows, num_blocks, *resource.desc.shape),
                                dtype=resource.desc.dtype,
                                device=device)
            pools.append(CachePool(cache, entry_axis=1))
            caches.append(cache)

        return CacheAllocation(pools=tuple(pools), caches=tuple(caches))


@dataclass(frozen=True)
class CompositeBlockCacheLayout:
    """Combine ordered child layouts into one allocation."""

    layouts: tuple[BlockCacheLayout, ...]

    def allocate(self, num_blocks: int, device: torch.device | str) -> CacheAllocation:
        """Allocate child layouts and preserve their resource order."""
        allocations = [layout.allocate(num_blocks, device) for layout in self.layouts]
        pools = tuple(pool for allocation in allocations for pool in allocation.pools)
        caches = tuple(cache for allocation in allocations for cache in allocation.caches)
        return CacheAllocation(pools=pools, caches=caches)


@dataclass(frozen=True)
class PackedStateCacheLayout:
    """Pack state resources behind one state-slot entry axis."""

    resources: tuple[CacheResource, ...]

    def allocate(self, num_caches: int, device: torch.device | str) -> CacheAllocation:
        """Realize the layout for a state-slot count and device."""
        if len(self.resources) == 0 or num_caches == 0:
            pool_tensor = torch.empty((0, 0), dtype=torch.uint8, device=device)
            return CacheAllocation(pools=(CachePool(pool_tensor, entry_axis=0), ), caches=())

        pool_size = sum(resource.desc.aligned_size for resource in self.resources)
        pool_tensor = torch.zeros((num_caches, pool_size), dtype=torch.uint8, device=device)

        caches = []
        offset = 0
        for resource in self.resources:
            desc = resource.desc
            cache = pool_tensor[:, offset:offset + desc.size].view(desc.dtype)
            cache = cache.view((num_caches, *desc.shape))
            if resource.layer_rows is not None:
                dims = list(range(cache.dim()))
                cache = cache.permute(1, 0, *dims[2:])
            caches.append(cache)
            offset += desc.aligned_size

        return CacheAllocation(pools=(CachePool(pool_tensor, entry_axis=0), ), caches=tuple(caches))
