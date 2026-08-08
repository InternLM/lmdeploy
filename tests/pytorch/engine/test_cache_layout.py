# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from lmdeploy.pytorch.backends.default import DefaultOpsBackend
from lmdeploy.pytorch.backends.default.cache import DefaultCacheBackend
from lmdeploy.pytorch.backends.dlinfer.cache import (
    DlinferBlockCacheLayout,
    DlinferCacheBackend,
    DlinferStateCacheLayout,
)
from lmdeploy.pytorch.backends.dlinfer.op_backend import DlinferOpsBackend
from lmdeploy.pytorch.config import StateCacheSpec
from lmdeploy.pytorch.engine.cache_engine.layout import (
    CachePool,
    LayerRowBlockCacheLayout,
    PackedBlockCacheLayout,
    PackedStateCacheLayout,
)
from lmdeploy.pytorch.engine.cache_engine.plan import BlockCachePlan
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


def test_block_cache_plan_owns_geometry_layout_and_access_metadata():
    resources = (
        CacheResource('first',
                      CacheDesc(shape=[3], dtype=torch.float32, alignment=16),
                      layer_rows=LayerRowMap.build('first', [1, 9])),
        CacheResource('second',
                      CacheDesc(shape=[2], dtype=torch.float16, alignment=8),
                      layer_rows=LayerRowMap.build('second', [7])),
    )
    allocations = []

    class RecordingLayout:

        def allocate(self, num_blocks, device):
            allocations.append((num_blocks, str(device)))
            return LayerRowBlockCacheLayout(resources).allocate(num_blocks, device)

    plan = BlockCachePlan(resources=resources,
                          layout=RecordingLayout(),
                          kernel_blocks_per_logical_block=2)

    allocation = plan.allocate(num_logical_blocks=3, device='cpu')
    block_nbytes = plan.logical_block_nbytes

    assert allocations == [(6, 'cpu'), (2, 'meta')]
    assert [tuple(cache.shape) for cache in allocation.caches] == [(2, 6, 3), (1, 6, 2)]
    assert plan.cache_names == ('first', 'second')
    assert plan.layer_maps == {
        'first': {
            1: 0,
            9: 1,
        },
        'second': {
            7: 0,
        },
    }
    assert plan.uses_layer_rows
    assert block_nbytes == 2 * 2 * 16 + 1 * 2 * 8


def test_block_cache_plan_rejects_invalid_geometry():
    layout = PackedBlockCacheLayout((), num_layers=2)

    with pytest.raises(ValueError, match='kernel blocks per logical block'):
        BlockCachePlan(resources=(), layout=layout, kernel_blocks_per_logical_block=0)


def test_packed_block_allocation_owns_one_block_axis_pool():
    resources = (
        CacheResource('first', CacheDesc(shape=[3], dtype=torch.float32, alignment=16)),
        CacheResource('second', CacheDesc(shape=[2], dtype=torch.float16, alignment=8)),
    )

    layout = PackedBlockCacheLayout(resources, num_layers=2)
    allocation = layout.allocate(num_blocks=3, device='cpu')

    assert len(allocation.pools) == 1
    assert allocation.pools[0].entry_axis == 1
    assert tuple(allocation.pools[0].tensor.shape) == (2, 3, 24)
    assert torch.count_nonzero(allocation.pools[0].tensor) == 0
    assert allocation.nbytes == 2 * 3 * 24
    assert [tuple(cache.shape) for cache in allocation.caches] == [(2, 3, 3), (2, 3, 2)]
    assert [cache.storage_offset() for cache in allocation.caches] == [0, 8]

    legacy_pool, legacy_caches = allocation
    assert legacy_pool is allocation.pools[0].tensor
    assert legacy_caches == list(allocation.caches)


def test_layer_row_block_allocation_owns_one_pool_per_resource():
    resources = (
        CacheResource('first',
                      CacheDesc(shape=[3], dtype=torch.float32, alignment=16),
                      layer_rows=LayerRowMap.build('first', [1, 9])),
        CacheResource('second',
                      CacheDesc(shape=[2], dtype=torch.float16, alignment=8),
                      layer_rows=LayerRowMap.build('second', [7])),
    )

    layout = LayerRowBlockCacheLayout(resources)
    allocation = layout.allocate(num_blocks=3, device='cpu')

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

    layout = PackedStateCacheLayout(resources)
    allocation = layout.allocate(num_caches=4, device='cpu')

    assert allocation.pools[0].entry_axis == 0
    assert tuple(allocation.pools[0].tensor.shape) == (4, 80)
    assert torch.count_nonzero(allocation.pools[0].tensor) == 0
    assert allocation.nbytes == 320
    assert [tuple(cache.shape) for cache in allocation.caches] == [(2, 4, 5), (4, 3)]


def test_empty_state_allocation_keeps_one_empty_owning_pool():
    allocation = PackedStateCacheLayout(()).allocate(num_caches=4, device='cpu')

    assert len(allocation.pools) == 1
    assert allocation.pools[0].entry_axis == 0
    assert tuple(allocation.pools[0].tensor.shape) == (0, 0)
    assert allocation.caches == ()


def test_layer_row_block_layout_rejects_resources_without_layer_rows():
    resources = (CacheResource('plain', CacheDesc(shape=[3], dtype=torch.float32)), )

    with pytest.raises(ValueError, match='explicit layer rows'):
        LayerRowBlockCacheLayout(resources)


def test_default_cache_backend_selects_layout_from_resource_membership():
    plain_resources = (CacheResource('plain', CacheDesc(shape=[3], dtype=torch.float32)), )
    layer_resources = (
        CacheResource('layered',
                      CacheDesc(shape=[3], dtype=torch.float32),
                      layer_rows=LayerRowMap.build('layered', [1, 9])),
    )

    cache_backend = DefaultOpsBackend.get_cache_backend()
    block_layout = cache_backend.build_block_layout(plain_resources, num_layers=4)
    layer_layout = cache_backend.build_block_layout(layer_resources, num_layers=4)
    state_layout = cache_backend.build_state_layout(plain_resources)

    assert cache_backend is DefaultCacheBackend
    assert isinstance(block_layout, PackedBlockCacheLayout)
    assert isinstance(layer_layout, LayerRowBlockCacheLayout)
    assert isinstance(state_layout, PackedStateCacheLayout)


def test_dlinfer_block_layout_owns_contiguous_resource_tensors():
    resources = (
        CacheResource('plain', CacheDesc(shape=[3], dtype=torch.float32, alignment=16)),
        CacheResource('layered',
                      CacheDesc(shape=[2], dtype=torch.float16, alignment=8),
                      layer_rows=LayerRowMap.build('layered', [1, 9])),
    )

    layout = DlinferBlockCacheLayout(resources, num_layers=4)
    allocation = layout.allocate(num_blocks=3, device='cpu')
    meta_allocation = layout.allocate(num_blocks=3, device='meta')

    assert [pool.entry_axis for pool in allocation.pools] == [1, 1]
    assert [tuple(pool.tensor.shape) for pool in allocation.pools] == [(4, 3, 3), (2, 3, 2)]
    assert all(pool.tensor is cache for pool, cache in zip(allocation.pools, allocation.caches))
    assert all(cache.is_contiguous() for cache in allocation.caches)
    assert all(torch.count_nonzero(cache) == 0 for cache in allocation.caches)
    assert allocation.caches[0].stride(1) == 3
    assert allocation.caches[1].stride(1) == 2
    assert allocation.nbytes == meta_allocation.nbytes == 4 * 3 * 3 * 4 + 2 * 3 * 2 * 2


def test_dlinfer_state_layout_owns_contiguous_resource_tensors():
    resources = (
        CacheResource('plain', CacheDesc(shape=[3], dtype=torch.float16)),
        CacheResource('layered',
                      CacheDesc(shape=[2, 5], dtype=torch.float32),
                      layer_rows=LayerRowMap.build('layered', [3, 9])),
    )

    layout = DlinferStateCacheLayout(resources)
    allocation = layout.allocate(num_caches=4, device='cpu')
    meta_allocation = layout.allocate(num_caches=4, device='meta')

    assert [pool.entry_axis for pool in allocation.pools] == [0, 1]
    assert [tuple(pool.tensor.shape) for pool in allocation.pools] == [(4, 3), (2, 4, 5)]
    assert all(pool.tensor is cache for pool, cache in zip(allocation.pools, allocation.caches))
    assert all(cache.is_contiguous() for cache in allocation.caches)
    assert all(torch.count_nonzero(cache) == 0 for cache in allocation.caches)
    assert allocation.caches[1][0].is_contiguous()
    assert allocation.nbytes == meta_allocation.nbytes == 4 * 3 * 2 + 2 * 4 * 5 * 4


def test_dlinfer_ops_backend_selects_native_cache_provider():
    cache_backend = DlinferOpsBackend.get_cache_backend()
    resources = (CacheResource('plain', CacheDesc(shape=[3], dtype=torch.float32)), )

    assert cache_backend is DlinferCacheBackend
    assert isinstance(cache_backend.build_block_layout(resources, num_layers=2), DlinferBlockCacheLayout)
    assert isinstance(cache_backend.build_state_layout(resources), DlinferStateCacheLayout)


def test_dlinfer_empty_layouts_keep_semantic_empty_pools():
    block_allocation = DlinferBlockCacheLayout((), num_layers=2).allocate(num_blocks=3, device='cpu')
    state_allocation = DlinferStateCacheLayout(()).allocate(num_caches=4, device='cpu')

    assert block_allocation.caches == ()
    assert block_allocation.pools[0].entry_axis == 1
    assert tuple(block_allocation.pools[0].tensor.shape) == (2, 3, 0)
    assert state_allocation.caches == ()
    assert state_allocation.pools[0].entry_axis == 0
    assert tuple(state_allocation.pools[0].tensor.shape) == (0, 0)
