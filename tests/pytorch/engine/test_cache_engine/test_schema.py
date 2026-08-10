# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from lmdeploy.pytorch.config import StateCacheSpec
from lmdeploy.pytorch.engine.cache_engine.schema import (
    BlockCacheGeometry,
    BlockCacheRequest,
    CacheDesc,
    LayerRowMap,
    build_block_cache_tensor_specs_from_requests,
    build_state_cache_tensor_specs,
)


def test_cache_desc_owns_payload_and_alignment_sizes():
    desc = CacheDesc(shape=[3], dtype=torch.float16, alignment=8)

    assert desc.numel == 3
    assert desc.size == 6
    assert desc.aligned_size == 8


def test_block_cache_geometry_validates_and_converts_block_units():
    geometry = BlockCacheGeometry(logical_block_size=128, kernel_block_size=32)

    assert geometry.kernel_blocks_per_logical_block == 4

    with pytest.raises(ValueError, match='logical_block_size must be positive'):
        BlockCacheGeometry(logical_block_size=0, kernel_block_size=64)
    with pytest.raises(ValueError, match='greater than or equal'):
        BlockCacheGeometry(logical_block_size=32, kernel_block_size=64)
    with pytest.raises(ValueError, match='divisible'):
        BlockCacheGeometry(logical_block_size=96, kernel_block_size=64)


def test_layer_row_map_preserves_declared_row_order():
    row_map = LayerRowMap.build('index', [9, 1])

    assert row_map.layer_ids == (9, 1)
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


def test_build_block_cache_tensor_specs_counts_operator_request_rows():
    request = BlockCacheRequest('index', (64, 1, 132), torch.uint8, per_row_contiguous=True)
    tensor_specs = build_block_cache_tensor_specs_from_requests([request, request, request])

    assert len(tensor_specs) == 1
    assert tensor_specs[0].name == 'index'
    assert tensor_specs[0].desc.shape == [64, 1, 132]
    assert tensor_specs[0].layer_rows is None
    assert tensor_specs[0].consumer_rows == (0, 1, 2)
    assert tensor_specs[0].num_rows == 3
    assert tensor_specs[0].per_row_contiguous


def test_block_cache_request_normalizes_integer_like_shape_and_alignment():
    request = BlockCacheRequest('index', [64, 3], torch.float16, alignment=np.int64(128))

    assert request.shape == (64, 3)
    assert request.alignment == 128


def test_build_block_cache_tensor_specs_groups_heterogeneous_requests():
    first = BlockCacheRequest('index', (64, 128), torch.uint8)
    second = BlockCacheRequest('index', (64, 256), torch.uint8)

    tensor_specs = build_block_cache_tensor_specs_from_requests([first, second, first])

    assert [spec.name for spec in tensor_specs] == ['index', 'index']
    assert [spec.desc.shape for spec in tensor_specs] == [[64, 128], [64, 256]]
    assert [spec.consumer_rows for spec in tensor_specs] == [(0, 2), (1, )]


def test_build_state_cache_tensor_specs_prefers_names_and_keeps_anonymous_fallback():
    named = build_state_cache_tensor_specs(
        state_shapes=[((99, ), torch.float32)],
        state_specs=[StateCacheSpec('state', (5, ), torch.float16, layer_ids=[3, 1])],
    )
    anonymous = build_state_cache_tensor_specs(
        state_shapes=[((3, ), torch.float32), ((2, ), torch.float16)],
    )

    assert [spec.name for spec in named] == ['state']
    assert named[0].desc.shape == (2, 5)
    assert named[0].layer_rows.layer_ids == (3, 1)
    assert [spec.name for spec in anonymous] == ['state_0', 'state_1']
    assert all(spec.layer_rows is None for spec in anonymous)
