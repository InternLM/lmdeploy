# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from lmdeploy.pytorch.config import BlockCacheSpec, StateCacheSpec
from lmdeploy.pytorch.engine.cache_engine.schema import (
    CacheDesc,
    CacheResource,
    LayerRowMap,
    build_block_cache_resources,
    build_state_cache_resources,
    layer_maps_from_resources,
)


def test_cache_desc_owns_payload_and_alignment_sizes():
    desc = CacheDesc(shape=[3], dtype=torch.float16, alignment=8)

    assert desc.numel == 3
    assert desc.size == 6
    assert desc.aligned_size == 8


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
