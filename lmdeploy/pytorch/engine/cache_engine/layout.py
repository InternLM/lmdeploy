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
    """Own cache pools and retain the typed cache views derived from them."""

    pools: tuple[CachePool, ...]
    cache_tensors: tuple[torch.Tensor, ...]

    @property
    def nbytes(self) -> int:
        """Count owning pools without double-counting cache views."""
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
        return self.legacy_pool, list(self.cache_tensors)

    def __iter__(self):
        """Preserve legacy ``mem_pool, caches = allocate_caches()`` use."""
        return iter(self.as_legacy())


def _unpack_cache_allocation(
    result: CacheAllocation | tuple[torch.Tensor | list[torch.Tensor], list[torch.Tensor]],
) -> tuple[CacheAllocation | None, torch.Tensor | list[torch.Tensor], list[torch.Tensor]]:
    """Normalize native or legacy allocation while preserving its facade."""
    if isinstance(result, CacheAllocation):
        legacy_pool, caches = result.as_legacy()
        return result, legacy_pool, caches
    legacy_pool, caches = result
    return None, legacy_pool, caches


@dataclass(frozen=True)
class PackedBlockCacheLayout:
    """Pack uniform-layer block-cache tensors into one owning pool."""

    tensor_specs: tuple[CacheTensorSpec, ...]
    num_layers: int

    def allocate(self, num_blocks: int, device: torch.device | str) -> CacheAllocation:
        """Realize the layout for a block count and device."""
        pool_size = sum(spec.desc.aligned_size for spec in self.tensor_specs)
        pool_tensor = torch.zeros((self.num_layers, num_blocks, pool_size), dtype=torch.uint8, device=device)

        cache_tensors = []
        offset = 0
        for spec in self.tensor_specs:
            desc = spec.desc
            cache = pool_tensor[:, :, offset:offset + desc.size].view(desc.dtype)
            cache = cache.view((self.num_layers, num_blocks, *desc.shape))
            cache_tensors.append(cache)
            offset += desc.aligned_size

        return CacheAllocation(pools=(CachePool(pool_tensor, entry_axis=1), ),
                               cache_tensors=tuple(cache_tensors))


@dataclass(frozen=True)
class RowBlockCacheLayout:
    """Give every row-scoped block-cache tensor its own compact-row pool."""

    tensor_specs: tuple[CacheTensorSpec, ...]

    def __post_init__(self):
        if any(not spec.has_rows for spec in self.tensor_specs):
            raise ValueError('Row block layouts require explicit rows for every tensor spec.')

    def allocate(self, num_blocks: int, device: torch.device | str) -> CacheAllocation:
        """Realize the layout for a block count and device."""
        pools = []
        cache_tensors = []
        for spec in self.tensor_specs:
            desc = spec.desc
            pool_tensor = torch.zeros((spec.num_rows, num_blocks, desc.aligned_size),
                                      dtype=torch.uint8,
                                      device=device)
            cache = pool_tensor[:, :, :desc.size].view(desc.dtype)
            cache = cache.view((spec.num_rows, num_blocks, *desc.shape))
            pools.append(CachePool(pool_tensor, entry_axis=1))
            cache_tensors.append(cache)

        return CacheAllocation(pools=tuple(pools), cache_tensors=tuple(cache_tensors))


@dataclass(frozen=True)
class ContiguousBlockCacheLayout:
    """Give every cache-tensor spec an independent contiguous tensor."""

    tensor_specs: tuple[CacheTensorSpec, ...]
    num_layers: int

    def allocate(self, num_blocks: int, device: torch.device | str) -> CacheAllocation:
        """Allocate contiguous cache tensors for one block count."""
        pools = []
        cache_tensors = []
        for spec in self.tensor_specs:
            num_rows = spec.num_rows if spec.has_rows else self.num_layers
            cache = torch.zeros((num_rows, num_blocks, *spec.desc.shape),
                                dtype=spec.desc.dtype,
                                device=device)
            pools.append(CachePool(cache, entry_axis=1))
            cache_tensors.append(cache)

        return CacheAllocation(pools=tuple(pools), cache_tensors=tuple(cache_tensors))


@dataclass(frozen=True)
class CompositeBlockCacheLayout:
    """Combine ordered child layouts into one allocation."""

    layouts: tuple[BlockCacheLayout, ...]

    def allocate(self, num_blocks: int, device: torch.device | str) -> CacheAllocation:
        """Allocate child layouts and preserve their cache-tensor order."""
        allocations = [layout.allocate(num_blocks, device) for layout in self.layouts]
        pools = tuple(pool for allocation in allocations for pool in allocation.pools)
        cache_tensors = tuple(cache for allocation in allocations for cache in allocation.cache_tensors)
        return CacheAllocation(pools=pools, cache_tensors=cache_tensors)


@dataclass(frozen=True)
class PackedStateCacheLayout:
    """Pack state-cache tensors behind one state-slot entry axis."""

    tensor_specs: tuple[CacheTensorSpec, ...]

    def allocate(self, num_caches: int, device: torch.device | str) -> CacheAllocation:
        """Realize the layout for a state-slot count and device."""
        if len(self.tensor_specs) == 0 or num_caches == 0:
            pool_tensor = torch.empty((0, 0), dtype=torch.uint8, device=device)
            return CacheAllocation(pools=(CachePool(pool_tensor, entry_axis=0), ), cache_tensors=())

        pool_size = sum(spec.desc.aligned_size for spec in self.tensor_specs)
        pool_tensor = torch.zeros((num_caches, pool_size), dtype=torch.uint8, device=device)

        cache_tensors = []
        offset = 0
        for spec in self.tensor_specs:
            desc = spec.desc
            cache = pool_tensor[:, offset:offset + desc.size].view(desc.dtype)
            cache = cache.view((num_caches, *desc.shape))
            if spec.layer_rows is not None:
                dims = list(range(cache.dim()))
                cache = cache.permute(1, 0, *dims[2:])
            cache_tensors.append(cache)
            offset += desc.aligned_size

        return CacheAllocation(pools=(CachePool(pool_tensor, entry_axis=0), ),
                               cache_tensors=tuple(cache_tensors))
