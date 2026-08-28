# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.engine.cache_engine import CacheEngine
from lmdeploy.pytorch.engine.cache_engine.schema import CacheDesc, CacheTensorSpec, LayerRowMap


def _spec(name, *, consumer_rows=None, layer_ids=None, payload=(1, ), dtype=torch.uint8) -> CacheTensorSpec:
    desc = CacheDesc(shape=list(payload), dtype=dtype)
    layer_rows = LayerRowMap.build(name, layer_ids) if layer_ids is not None else None
    return CacheTensorSpec(name=name, desc=desc, layer_rows=layer_rows, consumer_rows=consumer_rows)


def _make_cache_engine(
    tensor_specs,
    tensor_views,
    *,
    num_gpu_blocks: int,
    block_size: int,
    kernel_block_size: int,
) -> CacheEngine:
    cache_engine = object.__new__(CacheEngine)
    cache_engine.block_cache_plan = SimpleNamespace(tensor_specs=tuple(tensor_specs))
    cache_engine.gpu_allocation = SimpleNamespace(tensor_views=tuple(tensor_views))
    cache_engine.cache_config = SimpleNamespace(
        num_gpu_blocks=num_gpu_blocks,
        block_size=block_size,
        kernel_block_size=kernel_block_size,
    )
    return cache_engine


def test_connector_kv_caches_standard_rows_are_ordered_raw_views():
    k_view = torch.arange(3 * 6 * 5, dtype=torch.uint8).view(3, 6, 5)
    v_view = torch.arange(3 * 6 * 4, dtype=torch.uint8).view(3, 6, 4)
    specs = [_spec('k_cache', payload=(5, )), _spec('v_cache', payload=(4, ))]
    cache_engine = _make_cache_engine(specs, [k_view, v_view],
                                      num_gpu_blocks=3,
                                      block_size=4,
                                      kernel_block_size=2)

    connector_caches = cache_engine.connector_kv_caches

    assert list(connector_caches) == [
        'k_cache.layer.0',
        'k_cache.layer.1',
        'k_cache.layer.2',
        'v_cache.layer.0',
        'v_cache.layer.1',
        'v_cache.layer.2',
    ]
    for layer_id in range(3):
        row = connector_caches[f'k_cache.layer.{layer_id}']
        assert row.data_ptr() == k_view[layer_id].data_ptr()
        assert row.untyped_storage().data_ptr() == k_view.untyped_storage().data_ptr()
        assert tuple(row.shape) == (6, 5)
        assert row.is_contiguous()

    with pytest.raises(TypeError):
        connector_caches['new'] = k_view[0]


def test_connector_kv_caches_consumer_rows_are_ordered_raw_views():
    view = torch.arange(2 * 6 * 7, dtype=torch.uint8).view(2, 6, 7)
    specs = [_spec('indexer', consumer_rows=(0, 1), payload=(7, ))]
    cache_engine = _make_cache_engine(specs, [view], num_gpu_blocks=3, block_size=4, kernel_block_size=2)

    connector_caches = cache_engine.connector_kv_caches

    assert list(connector_caches) == ['block_cache.indexer.row.0', 'block_cache.indexer.row.1']
    assert connector_caches['block_cache.indexer.row.0'].data_ptr() == view[0].data_ptr()
    assert connector_caches['block_cache.indexer.row.1'].data_ptr() == view[1].data_ptr()


def test_connector_kv_caches_layer_rows_are_ordered_raw_views():
    view = torch.arange(2 * 6 * 7, dtype=torch.uint8).view(2, 6, 7)
    specs = [_spec('state', layer_ids=[1, 4], payload=(7, ))]
    cache_engine = _make_cache_engine(specs, [view], num_gpu_blocks=3, block_size=4, kernel_block_size=2)

    connector_caches = cache_engine.connector_kv_caches

    assert list(connector_caches) == ['block_cache.state.layer.1', 'block_cache.state.layer.4']
    assert connector_caches['block_cache.state.layer.1'].data_ptr() == view[0].data_ptr()
    assert connector_caches['block_cache.state.layer.4'].data_ptr() == view[1].data_ptr()


def test_connector_kv_caches_glm_layout_has_99_stable_rows():
    standard_view = torch.empty((78, 2, 3), dtype=torch.uint8)
    indexer_view = torch.empty((21, 2, 4), dtype=torch.uint8)
    specs = [
        _spec('k_cache', payload=(3, )),
        _spec('dsa_indexer_k', consumer_rows=tuple(range(21)), payload=(4, )),
    ]
    cache_engine = _make_cache_engine(specs, [standard_view, indexer_view],
                                      num_gpu_blocks=2,
                                      block_size=64,
                                      kernel_block_size=64)

    connector_caches = cache_engine.connector_kv_caches
    keys = list(connector_caches)

    assert len(keys) == 99
    assert keys[:2] == ['k_cache.layer.0', 'k_cache.layer.1']
    assert keys[77] == 'k_cache.layer.77'
    assert keys[78:] == [f'block_cache.dsa_indexer_k.row.{row_id}' for row_id in range(21)]
    assert connector_caches['k_cache.layer.77'].data_ptr() == standard_view[77].data_ptr()
    assert connector_caches['block_cache.dsa_indexer_k.row.20'].data_ptr() == indexer_view[-1].data_ptr()


def test_connector_kv_caches_rejects_view_count_mismatch():
    view = torch.empty((2, 4, 3), dtype=torch.uint8)
    specs = [_spec('k_cache', payload=(3, )), _spec('v_cache', payload=(3, ))]
    cache_engine = _make_cache_engine(specs, [view], num_gpu_blocks=2, block_size=4, kernel_block_size=2)

    with pytest.raises(ValueError, match='expects 2 packed views, got 1'):
        _ = cache_engine.connector_kv_caches


def test_connector_kv_caches_rejects_block_size_below_kernel_block_size():
    view = torch.empty((2, 4, 3), dtype=torch.uint8)
    specs = [_spec('k_cache', payload=(3, ))]
    cache_engine = _make_cache_engine(specs, [view], num_gpu_blocks=2, block_size=2, kernel_block_size=4)

    with pytest.raises(ValueError, match='block_size 2 must be greater than or equal to'):
        _ = cache_engine.connector_kv_caches


@pytest.mark.parametrize(
    ('view', 'match'),
    [
        (torch.empty((4, ), dtype=torch.uint8), 'must have shape'),
        (torch.empty((2, 5, 3), dtype=torch.uint8), 'must have shape'),
        (torch.empty((4, 2, 3), dtype=torch.uint8).permute(1, 0, 2), 'packed row must be contiguous'),
    ],
)
def test_connector_kv_caches_validates_view_layout(view, match):
    specs = [_spec('k_cache', payload=(3, ))]
    cache_engine = _make_cache_engine(specs, [view], num_gpu_blocks=2, block_size=4, kernel_block_size=2)

    with pytest.raises(ValueError, match=match):
        _ = cache_engine.connector_kv_caches


def test_connector_kv_caches_rejects_non_tensor_view():
    specs = [_spec('k_cache', payload=(3, ))]
    cache_engine = _make_cache_engine(specs, ['not-a-tensor'], num_gpu_blocks=2, block_size=4, kernel_block_size=2)

    with pytest.raises(TypeError, match='must be a torch.Tensor'):
        _ = cache_engine.connector_kv_caches


def test_connector_kv_caches_rejects_consumer_rows_beyond_view():
    view = torch.empty((2, 4, 3), dtype=torch.uint8)
    specs = [_spec('indexer', consumer_rows=(0, 1, 2), payload=(3, ))]
    cache_engine = _make_cache_engine(specs, [view], num_gpu_blocks=2, block_size=4, kernel_block_size=2)

    with pytest.raises(ValueError, match='outside its 2 view rows'):
        _ = cache_engine.connector_kv_caches


def test_connector_kv_caches_rejects_duplicate_named_cache_keys():
    view = torch.empty((2, 2, 3), dtype=torch.uint8)
    specs = [_spec('duplicate', payload=(3, )), _spec('duplicate', payload=(3, ))]
    cache_engine = _make_cache_engine(specs, [view, view], num_gpu_blocks=2, block_size=2, kernel_block_size=2)

    with pytest.raises(ValueError, match='duplicate connector cache key'):
        _ = cache_engine.connector_kv_caches
