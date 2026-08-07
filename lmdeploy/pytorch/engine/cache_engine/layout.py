# Copyright (c) OpenMMLab. All rights reserved.
"""Default physical cache allocation and owning-pool metadata."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .schema import CacheResource


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


def allocate_packed_block_caches(resources: Sequence[CacheResource],
                                 num_layers: int,
                                 num_blocks: int,
                                 device: torch.device | str) -> CacheAllocation:
    """Pack all block-cache resources behind layer and block axes."""
    pool_size = sum(resource.desc.aligned_size for resource in resources)
    pool_tensor = torch.zeros((num_layers, num_blocks, pool_size), dtype=torch.uint8, device=device)

    caches = []
    offset = 0
    for resource in resources:
        desc = resource.desc
        cache = pool_tensor[:, :, offset:offset + desc.size].view(desc.dtype)
        cache = cache.view((num_layers, num_blocks, *desc.shape))
        caches.append(cache)
        offset += desc.aligned_size

    return CacheAllocation(pools=(CachePool(pool_tensor, entry_axis=1), ), caches=tuple(caches))


def allocate_layer_row_block_caches(resources: Sequence[CacheResource],
                                    num_blocks: int,
                                    device: torch.device | str) -> CacheAllocation:
    """Allocate one compact layer-row pool for each block resource."""
    pools = []
    caches = []
    for resource in resources:
        desc = resource.desc
        pool_tensor = torch.zeros((resource.num_rows, num_blocks, desc.aligned_size),
                                  dtype=torch.uint8,
                                  device=device)
        cache = pool_tensor[:, :, :desc.size].view(desc.dtype)
        cache = cache.view((resource.num_rows, num_blocks, *desc.shape))
        pools.append(CachePool(pool_tensor, entry_axis=1))
        caches.append(cache)

    return CacheAllocation(pools=tuple(pools), caches=tuple(caches))


def allocate_packed_state_caches(resources: Sequence[CacheResource],
                                 num_caches: int,
                                 device: torch.device | str) -> CacheAllocation:
    """Pack state resources behind one state-slot entry axis."""
    if len(resources) == 0 or num_caches == 0:
        pool_tensor = torch.empty((0, 0), dtype=torch.uint8, device=device)
        return CacheAllocation(pools=(CachePool(pool_tensor, entry_axis=0), ), caches=())

    pool_size = sum(resource.desc.aligned_size for resource in resources)
    pool_tensor = torch.zeros((num_caches, pool_size), dtype=torch.uint8, device=device)

    caches = []
    offset = 0
    for resource in resources:
        desc = resource.desc
        cache = pool_tensor[:, offset:offset + desc.size].view(desc.dtype)
        cache = cache.view((num_caches, *desc.shape))
        if resource.layer_rows is not None:
            dims = list(range(cache.dim()))
            cache = cache.permute(1, 0, *dims[2:])
        caches.append(cache)
        offset += desc.aligned_size

    return CacheAllocation(pools=(CachePool(pool_tensor, entry_axis=0), ), caches=tuple(caches))
