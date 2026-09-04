# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from operator import index as as_index
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..engine.cache_engine.layout import CacheAllocation, CachePool


class BlockCacheCopy(ABC):
    """Copy logical blocks across one stable cache allocation."""

    def __init__(self, allocation: 'CacheAllocation', num_logical_blocks: int,
                 pages_per_block: int):
        num_logical_blocks = as_index(num_logical_blocks)
        pages_per_block = as_index(pages_per_block)
        if num_logical_blocks < 0:
            raise ValueError('num_logical_blocks must be non-negative.')
        if pages_per_block <= 0:
            raise ValueError('pages_per_block must be positive.')

        pools: tuple[CachePool, ...] = tuple(allocation.pools)
        if not pools:
            raise ValueError('Block-cache copy requires at least one allocation pool.')
        device = pools[0].tensor.device
        expected_pages = num_logical_blocks * pages_per_block
        for pool_id, pool in enumerate(pools):
            if pool.tensor.device != device:
                raise ValueError('Block-cache allocation pools must use one device.')
            if pool.tensor.size(pool.entry_axis) != expected_pages:
                raise ValueError(
                    f'Block-cache pool {pool_id} has {pool.tensor.size(pool.entry_axis)} physical pages; '
                    f'expected {expected_pages}.')

        self.pools = pools
        self.num_logical_blocks = num_logical_blocks
        self.pages_per_block = pages_per_block
        self.device = device

    @abstractmethod
    def copy(self, src_block_offsets: torch.Tensor, dst_block_offsets: torch.Tensor) -> None:
        """Copy scheduler-sized blocks on the current stream."""
        raise NotImplementedError


class CacheBackend(ABC):
    """Build backend-specific cache layouts and local primitives."""

    @classmethod
    @abstractmethod
    def build_block_layout(cls, tensor_specs, num_layers: int):
        """Select the physical layout for block-cache tensor specs."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_state_layout(cls, tensor_specs):
        """Select the physical layout for state-cache tensor specs."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_block_copy(cls, allocation: 'CacheAllocation', num_logical_blocks: int,
                         pages_per_block: int) -> BlockCacheCopy:
        """Build the local logical-block copy primitive."""
        raise NotImplementedError
