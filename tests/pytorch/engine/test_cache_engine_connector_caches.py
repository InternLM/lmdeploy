# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.config import BlockCacheSpec
from lmdeploy.pytorch.engine.cache_engine import CacheEngine


def _make_cache_engine(
    full_gpu_cache: torch.Tensor | list[torch.Tensor],
    *,
    num_layers: int,
    num_gpu_blocks: int,
    block_size: int,
    kernel_block_size: int,
    use_standard_kv_cache: bool,
    block_cache_specs: list[BlockCacheSpec] | None = None,
):
    cache_engine = object.__new__(CacheEngine)
    cache_engine.full_gpu_cache = full_gpu_cache
    cache_engine.model_config = SimpleNamespace(
        num_layers=num_layers,
        use_standard_kv_cache=use_standard_kv_cache,
        block_cache_specs=block_cache_specs or [],
    )
    cache_engine.cache_config = SimpleNamespace(
        num_gpu_blocks=num_gpu_blocks,
        block_size=block_size,
        kernel_block_size=kernel_block_size,
    )
    return cache_engine


def test_connector_kv_caches_standard_rows_are_ordered_raw_views():
    pool = torch.arange(3 * 6 * 5, dtype=torch.uint8).view(3, 6, 5)
    cache_engine = _make_cache_engine(
        pool,
        num_layers=3,
        num_gpu_blocks=3,
        block_size=4,
        kernel_block_size=2,
        use_standard_kv_cache=True,
    )

    connector_caches = cache_engine.connector_kv_caches

    assert list(connector_caches) == [
        'standard_kv_cache.layer.0',
        'standard_kv_cache.layer.1',
        'standard_kv_cache.layer.2',
    ]
    for layer_id, row in enumerate(connector_caches.values()):
        assert row.data_ptr() == pool[layer_id].data_ptr()
        assert row.untyped_storage().data_ptr() == pool.untyped_storage().data_ptr()
        assert tuple(row.shape) == (6, 5)
        assert row.is_contiguous()

    with pytest.raises(TypeError):
        connector_caches['new'] = pool[0]


def test_connector_kv_caches_named_only_accepts_single_tensor_pool():
    pool = torch.empty((2, 6, 7), dtype=torch.uint8)
    cache_engine = _make_cache_engine(
        pool,
        num_layers=6,
        num_gpu_blocks=3,
        block_size=4,
        kernel_block_size=2,
        use_standard_kv_cache=False,
        block_cache_specs=[BlockCacheSpec('indexer', [1, 4], (7, ), torch.uint8)],
    )

    connector_caches = cache_engine.connector_kv_caches

    assert list(connector_caches) == [
        'block_cache.indexer.layer.1',
        'block_cache.indexer.layer.4',
    ]
    assert connector_caches['block_cache.indexer.layer.1'].data_ptr() == pool[0].data_ptr()
    assert connector_caches['block_cache.indexer.layer.4'].data_ptr() == pool[1].data_ptr()


def test_connector_kv_caches_glm_layout_has_99_stable_rows():
    indexer_layer_ids = [0, 1, 2, *range(6, 78, 4)]
    assert len(indexer_layer_ids) == 21

    standard_pool = torch.empty((78, 2, 3), dtype=torch.uint8)
    indexer_pool = torch.empty((21, 2, 4), dtype=torch.uint8)
    cache_engine = _make_cache_engine(
        [standard_pool, indexer_pool],
        num_layers=78,
        num_gpu_blocks=2,
        block_size=64,
        kernel_block_size=64,
        use_standard_kv_cache=True,
        block_cache_specs=[BlockCacheSpec('dsa_indexer_k', indexer_layer_ids, (64, 1, 132), torch.uint8)],
    )

    connector_caches = cache_engine.connector_kv_caches
    keys = list(connector_caches)

    assert len(keys) == 99
    assert keys[:2] == ['standard_kv_cache.layer.0', 'standard_kv_cache.layer.1']
    assert keys[77] == 'standard_kv_cache.layer.77'
    assert keys[78:] == [f'block_cache.dsa_indexer_k.layer.{layer_id}' for layer_id in indexer_layer_ids]
    assert connector_caches['standard_kv_cache.layer.77'].data_ptr() == standard_pool[77].data_ptr()
    assert connector_caches['block_cache.dsa_indexer_k.layer.74'].data_ptr() == indexer_pool[-1].data_ptr()


def test_connector_kv_caches_rejects_pool_count_mismatch():
    pool = torch.empty((2, 4, 3), dtype=torch.uint8)
    cache_engine = _make_cache_engine(
        [pool, pool],
        num_layers=2,
        num_gpu_blocks=2,
        block_size=4,
        kernel_block_size=2,
        use_standard_kv_cache=True,
    )

    with pytest.raises(ValueError, match='expects 1 packed pools, got 2'):
        _ = cache_engine.connector_kv_caches


@pytest.mark.parametrize(
    ('pool', 'error_type', 'match'),
    [
        (torch.empty((2, 4, 3), dtype=torch.float32), TypeError, 'must use torch.uint8'),
        (torch.empty((2, 4), dtype=torch.uint8), ValueError, 'must have shape'),
        (torch.empty((2, 4, 3), dtype=torch.uint8).transpose(1, 2), ValueError, 'must be contiguous'),
        (torch.empty((3, 4, 3), dtype=torch.uint8), ValueError, 'has 3 rows, expected 2'),
        (torch.empty((2, 5, 3), dtype=torch.uint8), ValueError, 'has 5 kernel blocks, expected 4'),
    ],
)
def test_connector_kv_caches_validates_raw_pool_layout(pool, error_type, match):
    cache_engine = _make_cache_engine(
        pool,
        num_layers=2,
        num_gpu_blocks=2,
        block_size=4,
        kernel_block_size=2,
        use_standard_kv_cache=True,
    )

    with pytest.raises(error_type, match=match):
        _ = cache_engine.connector_kv_caches


def test_connector_kv_caches_rejects_duplicate_named_cache_keys():
    pools = [
        torch.empty((1, 2, 3), dtype=torch.uint8),
        torch.empty((1, 2, 5), dtype=torch.uint8),
    ]
    cache_engine = _make_cache_engine(
        pools,
        num_layers=2,
        num_gpu_blocks=2,
        block_size=2,
        kernel_block_size=2,
        use_standard_kv_cache=False,
        block_cache_specs=[
            BlockCacheSpec('duplicate', [1], (3, ), torch.uint8),
            BlockCacheSpec('duplicate', [1], (5, ), torch.uint8),
        ],
    )

    with pytest.raises(ValueError, match='duplicate connector cache key'):
        _ = cache_engine.connector_kv_caches
