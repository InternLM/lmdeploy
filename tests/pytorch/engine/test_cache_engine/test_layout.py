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
    CompositeBlockCacheLayout,
    ContiguousBlockCacheLayout,
    PackedBlockCacheLayout,
    PackedStateCacheLayout,
    RowBlockCacheLayout,
)
from lmdeploy.pytorch.engine.cache_engine.schema import (
    CacheDesc,
    CacheTensorSpec,
    LayerRowMap,
    build_state_cache_tensor_specs,
)


def test_cache_pool_validates_and_counts_owning_storage():
    tensor = torch.empty((2, 3, 16), dtype=torch.uint8)
    pool = CachePool(tensor=tensor, entry_axis=1)

    assert pool.nbytes == tensor.numel()
    with pytest.raises(ValueError, match='entry_axis'):
        CachePool(tensor=tensor, entry_axis=3)


def test_packed_block_allocation_owns_one_block_axis_pool():
    tensor_specs = (
        CacheTensorSpec('first', CacheDesc(shape=[3], dtype=torch.float32, alignment=16)),
        CacheTensorSpec('second', CacheDesc(shape=[2], dtype=torch.float16, alignment=8)),
    )

    layout = PackedBlockCacheLayout(tensor_specs, num_layers=2)
    allocation = layout.allocate(num_blocks=3, device='cpu')

    assert len(allocation.pools) == 1
    assert allocation.pools[0].entry_axis == 1
    assert tuple(allocation.pools[0].tensor.shape) == (2, 3, 24)
    assert torch.count_nonzero(allocation.pools[0].tensor) == 0
    assert allocation.nbytes == 2 * 3 * 24
    assert [tuple(cache.shape) for cache in allocation.cache_tensors] == [(2, 3, 3), (2, 3, 2)]
    assert [cache.storage_offset() for cache in allocation.cache_tensors] == [0, 8]

    legacy_pool, legacy_caches = allocation
    assert legacy_pool is allocation.pools[0].tensor
    assert legacy_caches == list(allocation.cache_tensors)


def test_layer_row_block_allocation_owns_one_pool_per_tensor_spec():
    tensor_specs = (
        CacheTensorSpec('first',
                        CacheDesc(shape=[3], dtype=torch.float32, alignment=16),
                        layer_rows=LayerRowMap.build('first', [1, 9])),
        CacheTensorSpec('second',
                        CacheDesc(shape=[2], dtype=torch.float16, alignment=8),
                        layer_rows=LayerRowMap.build('second', [7])),
    )

    layout = RowBlockCacheLayout(tensor_specs)
    allocation = layout.allocate(num_blocks=3, device='cpu')

    assert [pool.entry_axis for pool in allocation.pools] == [1, 1]
    assert [tuple(pool.tensor.shape) for pool in allocation.pools] == [(2, 3, 16), (1, 3, 8)]
    assert all(torch.count_nonzero(pool.tensor) == 0 for pool in allocation.pools)
    assert allocation.nbytes == 120
    assert [tuple(cache.shape) for cache in allocation.cache_tensors] == [(2, 3, 3), (1, 3, 2)]


def test_packed_state_allocation_owns_state_slot_axis():
    tensor_specs = build_state_cache_tensor_specs(
        (),
        state_specs=[
            StateCacheSpec('layered', (5, ), torch.float32, layer_ids=[3, 9], alignment=64),
            StateCacheSpec('shared', (3, ), torch.float16, alignment=16),
        ],
    )

    layout = PackedStateCacheLayout(tensor_specs)
    allocation = layout.allocate(num_caches=4, device='cpu')

    assert allocation.pools[0].entry_axis == 0
    assert tuple(allocation.pools[0].tensor.shape) == (4, 80)
    assert torch.count_nonzero(allocation.pools[0].tensor) == 0
    assert allocation.nbytes == 320
    assert [tuple(cache.shape) for cache in allocation.cache_tensors] == [(2, 4, 5), (4, 3)]


def test_empty_state_allocation_keeps_one_empty_owning_pool():
    allocation = PackedStateCacheLayout(()).allocate(num_caches=4, device='cpu')

    assert len(allocation.pools) == 1
    assert allocation.pools[0].entry_axis == 0
    assert tuple(allocation.pools[0].tensor.shape) == (0, 0)
    assert allocation.cache_tensors == ()


def test_row_block_layout_rejects_tensor_specs_without_rows():
    tensor_specs = (CacheTensorSpec('plain', CacheDesc(shape=[3], dtype=torch.float32)), )

    with pytest.raises(ValueError, match='explicit rows'):
        RowBlockCacheLayout(tensor_specs)


def test_default_cache_backend_selects_layout_from_tensor_spec_membership():
    plain_specs = (CacheTensorSpec('plain', CacheDesc(shape=[3], dtype=torch.float32)), )
    layer_specs = (
        CacheTensorSpec('layered',
                        CacheDesc(shape=[3], dtype=torch.float32),
                        layer_rows=LayerRowMap.build('layered', [1, 9])),
    )

    cache_backend = DefaultOpsBackend.get_cache_backend()
    block_layout = cache_backend.build_block_layout(plain_specs, num_layers=4)
    layer_layout = cache_backend.build_block_layout(layer_specs, num_layers=4)
    state_layout = cache_backend.build_state_layout(plain_specs)

    assert cache_backend is DefaultCacheBackend
    assert isinstance(block_layout, PackedBlockCacheLayout)
    assert isinstance(layer_layout, RowBlockCacheLayout)
    assert isinstance(state_layout, PackedStateCacheLayout)


def test_default_composite_layout_packs_plain_and_isolates_contiguous_tensors():
    tensor_specs = (
        CacheTensorSpec('first', CacheDesc(shape=[3], dtype=torch.float32, alignment=16)),
        CacheTensorSpec('second', CacheDesc(shape=[2], dtype=torch.float16, alignment=8)),
        CacheTensorSpec(
            'index',
            CacheDesc(shape=[5], dtype=torch.float16),
            consumer_rows=(0, 1),
            per_row_contiguous=True,
        ),
    )

    layout = DefaultCacheBackend.build_block_layout(tensor_specs, num_layers=4)
    allocation = layout.allocate(num_blocks=3, device='cpu')
    meta_allocation = layout.allocate(num_blocks=3, device='meta')

    assert isinstance(layout, CompositeBlockCacheLayout)
    assert [type(child) for child in layout.layouts] == [PackedBlockCacheLayout, ContiguousBlockCacheLayout]
    assert [pool.entry_axis for pool in allocation.pools] == [1, 1]
    assert [tuple(pool.tensor.shape) for pool in allocation.pools] == [(4, 3, 24), (2, 3, 5)]
    assert [tuple(cache.shape) for cache in allocation.cache_tensors] == [(4, 3, 3), (4, 3, 2), (2, 3, 5)]
    assert allocation.cache_tensors[2] is allocation.pools[1].tensor
    assert allocation.cache_tensors[2].is_contiguous()
    assert allocation.cache_tensors[2][0].is_contiguous()
    assert [allocation.cache_tensors[index].storage_offset() for index in (0, 1)] == [0, 8]
    assert all(torch.count_nonzero(pool.tensor) == 0 for pool in allocation.pools)
    assert allocation.nbytes == meta_allocation.nbytes == 4 * 3 * 24 + 2 * 3 * 5 * 2


def test_dlinfer_block_layout_owns_contiguous_cache_tensors():
    tensor_specs = (
        CacheTensorSpec('plain', CacheDesc(shape=[3], dtype=torch.float32, alignment=16)),
        CacheTensorSpec('layered',
                        CacheDesc(shape=[2], dtype=torch.float16, alignment=8),
                        layer_rows=LayerRowMap.build('layered', [1, 9])),
    )

    layout = DlinferBlockCacheLayout(tensor_specs, num_layers=4)
    allocation = layout.allocate(num_blocks=3, device='cpu')
    meta_allocation = layout.allocate(num_blocks=3, device='meta')

    assert [pool.entry_axis for pool in allocation.pools] == [1, 1]
    assert [tuple(pool.tensor.shape) for pool in allocation.pools] == [(4, 3, 3), (2, 3, 2)]
    assert all(pool.tensor is cache for pool, cache in zip(allocation.pools, allocation.cache_tensors))
    assert all(cache.is_contiguous() for cache in allocation.cache_tensors)
    assert all(torch.count_nonzero(cache) == 0 for cache in allocation.cache_tensors)
    assert allocation.cache_tensors[0].stride(1) == 3
    assert allocation.cache_tensors[1].stride(1) == 2
    assert allocation.nbytes == meta_allocation.nbytes == 4 * 3 * 3 * 4 + 2 * 3 * 2 * 2


def test_dlinfer_state_layout_owns_contiguous_cache_tensors():
    tensor_specs = (
        CacheTensorSpec('plain', CacheDesc(shape=[3], dtype=torch.float16)),
        CacheTensorSpec('layered',
                        CacheDesc(shape=[2, 5], dtype=torch.float32),
                        layer_rows=LayerRowMap.build('layered', [3, 9])),
    )

    layout = DlinferStateCacheLayout(tensor_specs)
    allocation = layout.allocate(num_caches=4, device='cpu')
    meta_allocation = layout.allocate(num_caches=4, device='meta')

    assert [pool.entry_axis for pool in allocation.pools] == [0, 1]
    assert [tuple(pool.tensor.shape) for pool in allocation.pools] == [(4, 3), (2, 4, 5)]
    assert all(pool.tensor is cache for pool, cache in zip(allocation.pools, allocation.cache_tensors))
    assert all(cache.is_contiguous() for cache in allocation.cache_tensors)
    assert all(torch.count_nonzero(cache) == 0 for cache in allocation.cache_tensors)
    assert allocation.cache_tensors[1][0].is_contiguous()
    assert allocation.nbytes == meta_allocation.nbytes == 4 * 3 * 2 + 2 * 4 * 5 * 4


def test_dlinfer_ops_backend_selects_native_cache_provider():
    cache_backend = DlinferOpsBackend.get_cache_backend()
    tensor_specs = (CacheTensorSpec('plain', CacheDesc(shape=[3], dtype=torch.float32)), )

    assert cache_backend is DlinferCacheBackend
    assert isinstance(cache_backend.build_block_layout(tensor_specs, num_layers=2), DlinferBlockCacheLayout)
    assert isinstance(cache_backend.build_state_layout(tensor_specs), DlinferStateCacheLayout)


def test_dlinfer_empty_layouts_keep_semantic_empty_pools():
    block_allocation = DlinferBlockCacheLayout((), num_layers=2).allocate(num_blocks=3, device='cpu')
    state_allocation = DlinferStateCacheLayout(()).allocate(num_caches=4, device='cpu')

    assert block_allocation.cache_tensors == ()
    assert block_allocation.pools[0].entry_axis == 1
    assert tuple(block_allocation.pools[0].tensor.shape) == (2, 3, 0)
    assert state_allocation.cache_tensors == ()
    assert state_allocation.pools[0].entry_axis == 0
    assert tuple(state_allocation.pools[0].tensor.shape) == (0, 0)
