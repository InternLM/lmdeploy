# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from lmdeploy.pytorch.config import StateCacheSpec
from lmdeploy.pytorch.engine.cache_engine.layout import (
    CachePool,
    allocate_layer_row_block_caches,
    allocate_packed_block_caches,
    allocate_packed_state_caches,
)
from lmdeploy.pytorch.engine.cache_engine.schema import (
    CacheDesc,
    CacheResource,
    LayerRowMap,
    build_state_cache_resources,
)


def test_cache_pool_validates_and_counts_owning_storage():
    tensor = torch.empty((2, 3, 16), dtype=torch.uint8)
    pool = CachePool(tensor=tensor, entry_axis=1)

    assert pool.nbytes == tensor.numel()
    with pytest.raises(ValueError, match='entry_axis'):
        CachePool(tensor=tensor, entry_axis=3)


def test_packed_block_allocation_owns_one_block_axis_pool():
    resources = (
        CacheResource('first', CacheDesc(shape=[3], dtype=torch.float32, alignment=16)),
        CacheResource('second', CacheDesc(shape=[2], dtype=torch.float16, alignment=8)),
    )

    allocation = allocate_packed_block_caches(resources, num_layers=2, num_blocks=3, device='cpu')

    assert len(allocation.pools) == 1
    assert allocation.pools[0].entry_axis == 1
    assert tuple(allocation.pools[0].tensor.shape) == (2, 3, 24)
    assert torch.count_nonzero(allocation.pools[0].tensor) == 0
    assert allocation.nbytes == 2 * 3 * 24
    assert [tuple(cache.shape) for cache in allocation.caches] == [(2, 3, 3), (2, 3, 2)]
    assert [cache.storage_offset() for cache in allocation.caches] == [0, 8]


def test_layer_row_block_allocation_owns_one_pool_per_resource():
    resources = (
        CacheResource('first',
                      CacheDesc(shape=[3], dtype=torch.float32, alignment=16),
                      layer_rows=LayerRowMap.build('first', [1, 9])),
        CacheResource('second',
                      CacheDesc(shape=[2], dtype=torch.float16, alignment=8),
                      layer_rows=LayerRowMap.build('second', [7])),
    )

    allocation = allocate_layer_row_block_caches(resources, num_blocks=3, device='cpu')

    assert [pool.entry_axis for pool in allocation.pools] == [1, 1]
    assert [tuple(pool.tensor.shape) for pool in allocation.pools] == [(2, 3, 16), (1, 3, 8)]
    assert all(torch.count_nonzero(pool.tensor) == 0 for pool in allocation.pools)
    assert allocation.nbytes == 120
    assert [tuple(cache.shape) for cache in allocation.caches] == [(2, 3, 3), (1, 3, 2)]


def test_packed_state_allocation_owns_state_slot_axis():
    resources = build_state_cache_resources(
        (),
        state_specs=[
            StateCacheSpec('layered', (5, ), torch.float32, layer_ids=[3, 9], alignment=64),
            StateCacheSpec('shared', (3, ), torch.float16, alignment=16),
        ],
    )

    allocation = allocate_packed_state_caches(resources, num_caches=4, device='cpu')

    assert allocation.pools[0].entry_axis == 0
    assert tuple(allocation.pools[0].tensor.shape) == (4, 80)
    assert torch.count_nonzero(allocation.pools[0].tensor) == 0
    assert allocation.nbytes == 320
    assert [tuple(cache.shape) for cache in allocation.caches] == [(2, 4, 5), (4, 3)]


def test_empty_state_allocation_keeps_one_empty_owning_pool():
    allocation = allocate_packed_state_caches((), num_caches=4, device='cpu')

    assert len(allocation.pools) == 1
    assert allocation.pools[0].entry_axis == 0
    assert tuple(allocation.pools[0].tensor.shape) == (0, 0)
    assert allocation.caches == ()
