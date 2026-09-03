# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from lmdeploy.pytorch.engine.cache_engine import CacheEngine
from lmdeploy.pytorch.engine.cache_engine.layout import (
    CacheAllocation,
    CachePool,
    CompositeBlockCacheLayout,
    ContiguousBlockCacheLayout,
    PackedBlockCacheLayout,
)
from lmdeploy.pytorch.engine.cache_engine.schema import CacheDesc, CacheTensorSpec


def _spec(name, *, consumer_rows=None, payload=(1, ), dtype=torch.uint8) -> CacheTensorSpec:
    desc = CacheDesc(shape=list(payload), dtype=dtype)
    return CacheTensorSpec(name=name, desc=desc, consumer_rows=consumer_rows)


def _make_cache_engine(allocation: CacheAllocation) -> CacheEngine:
    cache_engine = object.__new__(CacheEngine)
    cache_engine.gpu_allocation = allocation
    return cache_engine


def test_connector_kv_caches_registers_packed_pool_rows_once():
    specs = (
        _spec('k_cache', payload=(5, )),
        _spec('v_cache', payload=(4, )),
        _spec('k_scales_zeros', payload=(2, ), dtype=torch.float16),
    )
    allocation = PackedBlockCacheLayout(specs, num_layers=3).allocate(num_blocks=6, device='cpu')
    cache_engine = _make_cache_engine(allocation)

    connector_caches = cache_engine.connector_kv_caches

    assert list(connector_caches) == [f'cache_pool.0.row.{row}' for row in range(3)]
    pool = allocation.pools[0].tensor
    for row_index in range(3):
        row = connector_caches[f'cache_pool.0.row.{row_index}']
        assert row.data_ptr() == pool[row_index].data_ptr()
        assert row.untyped_storage().data_ptr() == pool.untyped_storage().data_ptr()
        assert tuple(row.shape) == tuple(pool.shape[1:])
        assert row.is_contiguous()

    with pytest.raises(TypeError):
        connector_caches['new'] = pool[0]


def test_connector_kv_caches_registers_independent_contiguous_pools():
    specs = (
        _spec('k_cache', payload=(3, ), dtype=torch.float16),
        _spec('v_cache', payload=(5, ), dtype=torch.float32),
    )
    allocation = ContiguousBlockCacheLayout(specs, num_layers=2).allocate(num_blocks=4, device='cpu')
    cache_engine = _make_cache_engine(allocation)

    connector_caches = cache_engine.connector_kv_caches

    assert list(connector_caches) == [
        'cache_pool.0.row.0',
        'cache_pool.0.row.1',
        'cache_pool.1.row.0',
        'cache_pool.1.row.1',
    ]
    for pool_index, pool in enumerate(allocation.pools):
        for row_index in range(2):
            row = connector_caches[f'cache_pool.{pool_index}.row.{row_index}']
            assert row.data_ptr() == pool.tensor[row_index].data_ptr()
            assert row.is_contiguous()


def test_connector_kv_caches_registers_composite_layout_in_pool_order():
    standard_specs = (
        _spec('k_cache', payload=(3, )),
        _spec('v_cache', payload=(4, )),
    )
    indexer_spec = _spec(
        'dsa_indexer_k',
        consumer_rows=(0, 1),
        payload=(7, ),
    )
    layout = CompositeBlockCacheLayout((
        PackedBlockCacheLayout(standard_specs, num_layers=3),
        ContiguousBlockCacheLayout((indexer_spec, ), num_layers=3),
    ))
    allocation = layout.allocate(num_blocks=2, device='cpu')
    cache_engine = _make_cache_engine(allocation)

    connector_caches = cache_engine.connector_kv_caches

    assert list(connector_caches) == [
        'cache_pool.0.row.0',
        'cache_pool.0.row.1',
        'cache_pool.0.row.2',
        'cache_pool.1.row.0',
        'cache_pool.1.row.1',
    ]
    assert connector_caches['cache_pool.0.row.2'].data_ptr() == allocation.pools[0].tensor[2].data_ptr()
    assert connector_caches['cache_pool.1.row.1'].data_ptr() == allocation.pools[1].tensor[1].data_ptr()


def test_connector_kv_caches_omits_empty_pools():
    allocation = PackedBlockCacheLayout((), num_layers=2).allocate(num_blocks=3, device='cpu')
    cache_engine = _make_cache_engine(allocation)

    assert cache_engine.connector_kv_caches == {}


def test_connector_kv_caches_rejects_non_row_major_pool():
    allocation = CacheAllocation(
        pools=(CachePool(torch.empty((4, 2, 3)), entry_axis=0), ),
        tensor_views=(),
    )
    cache_engine = _make_cache_engine(allocation)

    with pytest.raises(ValueError, match='entry_axis=1, got 0'):
        _ = cache_engine.connector_kv_caches
