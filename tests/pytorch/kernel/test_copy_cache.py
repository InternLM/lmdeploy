# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from lmdeploy.pytorch.kernels.cuda.copy_cache import copy_cache_blocks

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize(
    ('shape', 'entry_axis', 'dtype', 'pages_per_block'),
    [
        ((2, 12, 5), 1, torch.uint8, 2),
        ((18, 3, 7), 0, torch.float16, 3),
        ((2, 3, 12, 4), 2, torch.float32, 2),
    ],
)
def test_copy_cache_blocks_handles_contiguous_pool_axes(shape, entry_axis, dtype, pages_per_block):
    num_logical_blocks = shape[entry_axis] // pages_per_block
    cache = torch.full(shape, 99, dtype=dtype, device='cuda')
    logical_cache = cache.unflatten(entry_axis, (num_logical_blocks, pages_per_block))
    src_blocks = torch.tensor([0, 2, 0], dtype=torch.long, device='cuda')
    dst_blocks = torch.tensor([3, 1, 4], dtype=torch.long, device='cuda')
    for block_id in set(src_blocks.tolist()):
        logical_cache.select(entry_axis, block_id).fill_(block_id + 1)
    expected = logical_cache.index_select(entry_axis, src_blocks).clone()

    copy_cache_blocks(cache, entry_axis, src_blocks, dst_blocks, pages_per_block)

    assert torch.equal(logical_cache.index_select(entry_axis, dst_blocks), expected)
