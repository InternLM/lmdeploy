# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from lmdeploy.pytorch.config import BlockCacheSpec, StateCacheSpec
from lmdeploy.pytorch.engine.cache_engine.schema import (
    BlockCacheGeometry,
    BlockCacheRequest,
    CacheDesc,
    CacheResource,
    LayerRowMap,
    ScopedBlockCacheRequest,
    build_block_cache_resources,
    build_block_cache_resources_from_requests,
    build_state_cache_resources,
    layer_maps_from_resources,
)


def test_cache_desc_owns_payload_and_alignment_sizes():
    desc = CacheDesc(shape=[3], dtype=torch.float16, alignment=8)

    assert desc.numel == 3
    assert desc.size == 6
    assert desc.aligned_size == 8


def test_block_cache_geometry_validates_and_converts_block_units():
    geometry = BlockCacheGeometry(block_size=128, kernel_block_size=32)

    assert geometry.kernel_blocks_per_logical_block == 4

    with pytest.raises(ValueError, match='block_size must be positive'):
        BlockCacheGeometry(block_size=0, kernel_block_size=64)
    with pytest.raises(ValueError, match='greater than or equal'):
        BlockCacheGeometry(block_size=32, kernel_block_size=64)
    with pytest.raises(ValueError, match='divisible'):
        BlockCacheGeometry(block_size=96, kernel_block_size=64)


def test_layer_row_map_preserves_declared_row_order():
    row_map = LayerRowMap.build('index', [9, 1])

    assert row_map.layer_ids == (9, 1)
    assert row_map.row_by_layer == {9: 0, 1: 1}
    assert row_map.num_rows == 2


@pytest.mark.parametrize(
    ('layer_ids', 'message'),
    [
        ([], 'must not be empty'),
        ([1, 1], 'duplicated'),
        ([-1], 'non-negative'),
    ],
)
def test_layer_row_map_rejects_invalid_membership(layer_ids, message):
    with pytest.raises(ValueError, match=message):
        LayerRowMap.build('index', layer_ids)


def test_cache_resource_collects_only_layer_scoped_maps():
    desc = CacheDesc(shape=[4], dtype=torch.float32)
    layer_rows = LayerRowMap.build('layered', [2, 0])
    resources = (
        CacheResource(name='global', desc=desc),
        CacheResource(name='layered', desc=desc, layer_rows=layer_rows),
    )

    assert resources[0].layer_map is None
    assert resources[1].num_rows == 2
    assert layer_maps_from_resources(resources) == {'layered': {2: 0, 0: 1}}


def test_build_block_cache_resources_normalizes_specs_in_declared_order():
    specs = [
        BlockCacheSpec('compressed', [3, 1], (8, ), torch.float16),
        BlockCacheSpec('index', [2], (4, ), torch.uint8, alignment=128),
    ]

    resources = build_block_cache_resources(specs)

    assert [resource.name for resource in resources] == ['compressed', 'index']
    assert [resource.desc.shape for resource in resources] == [(8, ), (4, )]
    assert [resource.layer_map for resource in resources] == [{3: 0, 1: 1}, {2: 0}]
    assert resources[1].desc.alignment == 128


def test_build_block_cache_resources_aggregates_scoped_operator_requests():
    request = BlockCacheRequest('index', (64, 1, 132), torch.uint8, per_layer_contiguous=True)
    resources = build_block_cache_resources_from_requests([
        ScopedBlockCacheRequest(request, layer_id=9),
        ScopedBlockCacheRequest(request, layer_id=1),
        ScopedBlockCacheRequest(request, layer_id=9),
    ])

    assert len(resources) == 1
    assert resources[0].name == 'index'
    assert resources[0].desc.shape == [64, 1, 132]
    assert resources[0].layer_map == {9: 0, 1: 1}
    assert resources[0].per_layer_contiguous


def test_block_cache_request_normalizes_integer_like_shape_and_alignment():
    request = BlockCacheRequest('index', [64, 3], torch.float16, alignment=np.int64(128))

    assert request.shape == (64, 3)
    assert request.alignment == 128


def test_build_block_cache_resources_rejects_provider_conflicts():
    first = BlockCacheRequest('index', (64, 128), torch.uint8)
    second = BlockCacheRequest('index', (64, 256), torch.uint8)

    with pytest.raises(ValueError, match='Conflicting block cache requests'):
        build_block_cache_resources_from_requests([
            ScopedBlockCacheRequest(first, layer_id=3),
            ScopedBlockCacheRequest(second, layer_id=3),
        ])

    with pytest.raises(ValueError, match='Heterogeneous block cache request segments'):
        build_block_cache_resources_from_requests([
            ScopedBlockCacheRequest(first, layer_id=3),
            ScopedBlockCacheRequest(second, layer_id=7),
        ])


def test_build_block_cache_resources_rejects_unimplemented_global_scope():
    request = BlockCacheRequest('shared', (8, ), torch.float32)

    with pytest.raises(ValueError, match='Global block cache requests are not supported yet'):
        build_block_cache_resources_from_requests([
            ScopedBlockCacheRequest(request, layer_id=None),
        ])


def test_build_state_cache_resources_prefers_named_specs_and_keeps_legacy_bridge():
    named = build_state_cache_resources(
        state_shapes=[((99, ), torch.float32)],
        state_specs=[StateCacheSpec('state', (5, ), torch.float16, layer_ids=[3, 1])],
    )
    legacy = build_state_cache_resources(
        state_shapes=[((3, ), torch.float32), ((2, ), torch.float16)],
    )

    assert [resource.name for resource in named] == ['state']
    assert named[0].desc.shape == (2, 5)
    assert named[0].layer_map == {3: 0, 1: 1}
    assert [resource.name for resource in legacy] == ['state_0', 'state_1']
    assert all(resource.layer_map is None for resource in legacy)
