# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.engine.cache_engine import CacheEngine
from lmdeploy.pytorch.engine.cache_engine.layout import CacheAllocation
from lmdeploy.pytorch.engine.cache_engine.schema import CacheDesc, CacheTensorSpec


def _spec(name, *, consumer_rows=None, payload=(1, ), dtype=torch.uint8) -> CacheTensorSpec:
    desc = CacheDesc(shape=list(payload), dtype=dtype)
    return CacheTensorSpec(name=name, desc=desc, consumer_rows=consumer_rows)


def _pool(rows, kernel_blocks, payload, dtype=torch.uint8) -> torch.Tensor:
    return torch.empty((rows, kernel_blocks, payload), dtype=dtype)


def _make_cache_engine(
    tensor_specs,
    tensor_views,
    pools,
    *,
    num_gpu_blocks: int,
    block_size: int,
    kernel_block_size: int,
) -> CacheEngine:
    cache_engine = object.__new__(CacheEngine)
    cache_engine.block_cache_plan = SimpleNamespace(tensor_specs=tuple(tensor_specs))
    cache_engine.gpu_allocation = CacheAllocation(
        pools=tuple(pools),
        tensor_views=tuple(tensor_views),
    )
    cache_engine.cache_config = SimpleNamespace(
        num_gpu_blocks=num_gpu_blocks,
        block_size=block_size,
        kernel_block_size=kernel_block_size,
    )
    return cache_engine


def _standard_engine(rows, num_gpu_blocks, block_size, kernel_block_size, k_payload, v_payload):
    """Pack k_cache + v_cache into one shared uint8 pool, mirroring the real
    PackedBlockCacheLayout."""
    kernel_blocks = num_gpu_blocks * (block_size // kernel_block_size)
    pool = _pool(rows, kernel_blocks, k_payload + v_payload)
    k_view = pool[:, :, :k_payload]
    v_view = pool[:, :, k_payload:]
    specs = [_spec('k_cache', payload=(k_payload, )), _spec('v_cache', payload=(v_payload, ))]
    return _make_cache_engine(
        specs, [k_view, v_view], [SimpleNamespace(tensor=pool)],
        num_gpu_blocks=num_gpu_blocks,
        block_size=block_size,
        kernel_block_size=kernel_block_size)


def test_connector_kv_caches_standard_rows_are_packed_per_layer():
    cache_engine = _standard_engine(rows=3, num_gpu_blocks=3, block_size=4,
                                    kernel_block_size=2, k_payload=5, v_payload=4)

    connector_caches = cache_engine.connector_kv_caches

    assert list(connector_caches) == [
        'standard_kv_cache.layer.0',
        'standard_kv_cache.layer.1',
        'standard_kv_cache.layer.2',
    ]
    pool = cache_engine.gpu_allocation.pools[0].tensor
    for layer_id in range(3):
        row = connector_caches[f'standard_kv_cache.layer.{layer_id}']
        assert row.data_ptr() == pool[layer_id].data_ptr()
        assert row.untyped_storage().data_ptr() == pool.untyped_storage().data_ptr()
        assert tuple(row.shape) == (6, 9)
        assert row.is_contiguous()

    with pytest.raises(TypeError):
        connector_caches['new'] = pool[0]


def test_connector_kv_caches_consumer_rows_are_ordered_raw_views():
    pool = _pool(2, 6, 7)
    specs = [_spec('indexer', consumer_rows=(0, 1), payload=(7, ))]
    cache_engine = _make_cache_engine(
        specs, [pool], [SimpleNamespace(tensor=pool)],
        num_gpu_blocks=3, block_size=4, kernel_block_size=2)

    connector_caches = cache_engine.connector_kv_caches

    assert list(connector_caches) == ['block_cache.indexer.row.0', 'block_cache.indexer.row.1']
    assert connector_caches['block_cache.indexer.row.0'].data_ptr() == pool[0].data_ptr()
    assert connector_caches['block_cache.indexer.row.1'].data_ptr() == pool[1].data_ptr()


def test_connector_kv_caches_glm_layout_has_99_packed_rows():
    standard_pool = _pool(78, 2, 7)
    standard_k = standard_pool[:, :, :3]
    standard_v = standard_pool[:, :, 3:]
    indexer_pool = _pool(21, 2, 4)
    specs = [
        _spec('k_cache', payload=(3, )),
        _spec('v_cache', payload=(4, )),
        _spec('dsa_indexer_k', consumer_rows=tuple(range(21)), payload=(4, )),
    ]
    cache_engine = _make_cache_engine(
        specs, [standard_k, standard_v, indexer_pool],
        [SimpleNamespace(tensor=standard_pool), SimpleNamespace(tensor=indexer_pool)],
        num_gpu_blocks=2, block_size=64, kernel_block_size=64)

    connector_caches = cache_engine.connector_kv_caches
    keys = list(connector_caches)

    assert len(keys) == 99
    assert keys[:78] == [f'standard_kv_cache.layer.{i}' for i in range(78)]
    assert keys[78:] == [f'block_cache.dsa_indexer_k.row.{row_id}' for row_id in range(21)]
    assert connector_caches['standard_kv_cache.layer.77'].data_ptr() == standard_pool[77].data_ptr()
    assert connector_caches['block_cache.dsa_indexer_k.row.20'].data_ptr() == indexer_pool[-1].data_ptr()


def test_connector_kv_caches_rejects_duplicate_standard_pools():
    pool_a = _pool(2, 2, 3)
    pool_b = _pool(2, 2, 3)
    specs = [_spec('k_cache', payload=(3, )), _spec('k_cache', payload=(3, ))]
    cache_engine = _make_cache_engine(
        specs, [pool_a, pool_b],
        [SimpleNamespace(tensor=pool_a), SimpleNamespace(tensor=pool_b)],
        num_gpu_blocks=2, block_size=2, kernel_block_size=2)

    with pytest.raises(ValueError, match='duplicate connector cache key'):
        _ = cache_engine.connector_kv_caches
