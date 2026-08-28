# Copyright (c) OpenMMLab. All rights reserved.
"""Default physical cache allocation and owning-pool metadata."""

from dataclasses import dataclass
from typing import Protocol

import torch

from .schema import CacheTensorSpec


class BlockCacheLayout(Protocol):
    """Realize a physical block-cache allocation."""

    def allocate(self, num_blocks: int, device: torch.device | str) -> 'CacheAllocation':
        """Allocate physical kernel blocks on one device."""
        ...


@dataclass(frozen=True)
class CachePool:
    """Own one storage tensor used for cache movement and accounting."""

    tensor: torch.Tensor
    # Axis indexing independently movable entries: physical kernel pages for
    # block-cache pools or state slots for state-cache pools. Every other axis
    # forms the payload moved with one entry.
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
    """Own cache pools and retain typed tensor views in cache-spec order."""

    pools: tuple[CachePool, ...]
    tensor_views: tuple[torch.Tensor, ...]

    @property
    def nbytes(self) -> int:
        """Count owning pools without double-counting cache views."""
        return sum(pool.nbytes for pool in self.pools)


@dataclass(frozen=True)
class PackedBlockCacheLayout:
    """Pack uniform-layer block-cache tensors into one owning pool."""

    tensor_specs: tuple[CacheTensorSpec, ...]
    num_layers: int

    def allocate(self, num_blocks: int, device: torch.device | str) -> CacheAllocation:
        """Realize the layout for a block count and device."""
        pool_size = sum(spec.desc.aligned_size for spec in self.tensor_specs)
        pool_tensor = torch.zeros((self.num_layers, num_blocks, pool_size), dtype=torch.uint8, device=device)

        tensor_views = []
        offset = 0
        for spec in self.tensor_specs:
            desc = spec.desc
            cache = pool_tensor[:, :, offset:offset + desc.size].view(desc.dtype)
            cache = cache.view((self.num_layers, num_blocks, *desc.shape))
            tensor_views.append(cache)
            offset += desc.aligned_size

        return CacheAllocation(pools=(CachePool(pool_tensor, entry_axis=1), ),
                               tensor_views=tuple(tensor_views))


@dataclass(frozen=True)
class RowBlockCacheLayout:
    """Give every row-scoped block-cache tensor its own compact-row pool."""

    tensor_specs: tuple[CacheTensorSpec, ...]

    def __post_init__(self):
        if any(spec.consumer_rows is None for spec in self.tensor_specs):
            raise ValueError('Row block layouts require consumer rows for every tensor spec.')

    def allocate(self, num_blocks: int, device: torch.device | str) -> CacheAllocation:
        """Realize the layout for a block count and device."""
        pools = []
        tensor_views = []
        for spec in self.tensor_specs:
            desc = spec.desc
            pool_tensor = torch.zeros((spec.num_rows, num_blocks, desc.aligned_size),
                                      dtype=torch.uint8,
                                      device=device)
            cache = pool_tensor[:, :, :desc.size].view(desc.dtype)
            cache = cache.view((spec.num_rows, num_blocks, *desc.shape))
            pools.append(CachePool(pool_tensor, entry_axis=1))
            tensor_views.append(cache)

        return CacheAllocation(pools=tuple(pools), tensor_views=tuple(tensor_views))


@dataclass(frozen=True)
class ContiguousBlockCacheLayout:
    """Give every cache-tensor spec an independent contiguous tensor."""

    tensor_specs: tuple[CacheTensorSpec, ...]
    num_layers: int

    def allocate(self, num_blocks: int, device: torch.device | str) -> CacheAllocation:
        """Allocate contiguous cache tensors for one block count."""
        if not self.tensor_specs:
            empty = torch.empty((self.num_layers, num_blocks, 0), dtype=torch.uint8, device=device)
            return CacheAllocation(pools=(CachePool(empty, entry_axis=1), ), tensor_views=())

        pools = []
        tensor_views = []
        for spec in self.tensor_specs:
            num_rows = len(spec.consumer_rows) if spec.consumer_rows is not None else self.num_layers
            cache = torch.zeros((num_rows, num_blocks, *spec.desc.shape),
                                dtype=spec.desc.dtype,
                                device=device)
            pools.append(CachePool(cache, entry_axis=1))
            tensor_views.append(cache)

        return CacheAllocation(pools=tuple(pools), tensor_views=tuple(tensor_views))


@dataclass(frozen=True)
class CompositeBlockCacheLayout:
    """Combine ordered child layouts into one allocation."""

    layouts: tuple[BlockCacheLayout, ...]

    def allocate(self, num_blocks: int, device: torch.device | str) -> CacheAllocation:
        """Allocate child layouts and preserve their cache-tensor order."""
        allocations = [layout.allocate(num_blocks, device) for layout in self.layouts]
        pools = tuple(pool for allocation in allocations for pool in allocation.pools)
        tensor_views = tuple(cache for allocation in allocations for cache in allocation.tensor_views)
        return CacheAllocation(pools=pools, tensor_views=tensor_views)


@dataclass(frozen=True)
class PackedStateCacheLayout:
    """Pack state-cache tensors behind one state-slot entry axis."""

    tensor_specs: tuple[CacheTensorSpec, ...]

    def allocate(self, num_caches: int, device: torch.device | str) -> CacheAllocation:
        """Realize the layout for a state-slot count and device."""
        if len(self.tensor_specs) == 0 or num_caches == 0:
            return CacheAllocation(pools=(), tensor_views=())

        pool_size = sum(spec.desc.aligned_size for spec in self.tensor_specs)
        pool_tensor = torch.zeros((num_caches, pool_size), dtype=torch.uint8, device=device)

        tensor_views = []
        offset = 0
        for spec in self.tensor_specs:
            desc = spec.desc
            cache = pool_tensor[:, offset:offset + desc.size].view(desc.dtype)
            cache = cache.view((num_caches, *desc.shape))
            if spec.layer_rows is not None:
                dims = list(range(cache.dim()))
                cache = cache.permute(1, 0, *dims[2:])
            tensor_views.append(cache)
            offset += desc.aligned_size

        return CacheAllocation(pools=(CachePool(pool_tensor, entry_axis=0), ),
                               tensor_views=tuple(tensor_views))


@dataclass(frozen=True)
class ContiguousStateCacheLayout:
    """Give every state-cache spec an independent contiguous tensor."""

    tensor_specs: tuple[CacheTensorSpec, ...]

    def allocate(self, num_caches: int, device: torch.device | str) -> CacheAllocation:
        """Allocate contiguous state tensors for one slot count."""
        if not self.tensor_specs or num_caches == 0:
            return CacheAllocation(pools=(), tensor_views=())

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
