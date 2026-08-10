# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from lmdeploy.pytorch.engine.cache_engine.layout import PackedBlockCacheLayout, RowBlockCacheLayout
from lmdeploy.pytorch.engine.cache_engine.plan import BlockCachePlan
from lmdeploy.pytorch.engine.cache_engine.schema import CacheDesc, CacheTensorSpec, LayerRowMap


def test_block_cache_plan_owns_geometry_layout_and_access_metadata():
    tensor_specs = (
        CacheTensorSpec('first',
                        CacheDesc(shape=[3], dtype=torch.float32, alignment=16),
                        layer_rows=LayerRowMap.build('first', [1, 9])),
        CacheTensorSpec('second',
                        CacheDesc(shape=[2], dtype=torch.float16, alignment=8),
                        layer_rows=LayerRowMap.build('second', [7])),
    )
    allocations = []

    class RecordingLayout:

        def allocate(self, num_blocks, device):
            allocations.append((num_blocks, str(device)))
            return RowBlockCacheLayout(tensor_specs).allocate(num_blocks, device)

    plan = BlockCachePlan(tensor_specs=tensor_specs,
                          layout=RecordingLayout(),
                          kernel_blocks_per_logical_block=2)

    allocation = plan.allocate(num_logical_blocks=3, device='cpu')
    block_nbytes = plan.logical_block_nbytes

    assert allocations == [(6, 'cpu'), (2, 'meta')]
    assert [tuple(cache.shape) for cache in allocation.tensor_views] == [(2, 6, 3), (1, 6, 2)]
    assert plan.cache_names == ('first', 'second')
    assert plan.legacy_cache_indices == ()
    assert block_nbytes == 2 * 2 * 16 + 1 * 2 * 8


def test_block_cache_plan_rejects_invalid_geometry():
    layout = PackedBlockCacheLayout((), num_layers=2)

    with pytest.raises(ValueError, match='kernel blocks per logical block'):
        BlockCachePlan(tensor_specs=(), layout=layout, kernel_blocks_per_logical_block=0)


def test_block_cache_plan_validates_heterogeneous_consumer_rows():
    first = CacheTensorSpec('index', CacheDesc(shape=[3], dtype=torch.float32), consumer_rows=(0, 2))
    second = CacheTensorSpec('index', CacheDesc(shape=[5], dtype=torch.float32), consumer_rows=(1, ))
    tensor_specs = (first, second)

    plan = BlockCachePlan(tensor_specs=tensor_specs,
                          layout=RowBlockCacheLayout(tensor_specs),
                          kernel_blocks_per_logical_block=1)

    assert plan.cache_names == ('index', 'index')
    with pytest.raises(ValueError, match='row 0 belongs to multiple tensor specs'):
        duplicate = (
            CacheTensorSpec('index', CacheDesc(shape=[3], dtype=torch.float32), consumer_rows=(0, )),
            CacheTensorSpec('index', CacheDesc(shape=[5], dtype=torch.float32), consumer_rows=(0, )),
        )
        BlockCachePlan(tensor_specs=duplicate,
                       layout=RowBlockCacheLayout(duplicate),
                       kernel_blocks_per_logical_block=1)
    with pytest.raises(ValueError, match='contiguous from zero'):
        missing = (CacheTensorSpec('index', CacheDesc(shape=[3], dtype=torch.float32), consumer_rows=(1, )), )
        BlockCachePlan(tensor_specs=missing,
                       layout=RowBlockCacheLayout(missing),
                       kernel_blocks_per_logical_block=1)
