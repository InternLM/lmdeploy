# Copyright (c) OpenMMLab. All rights reserved.
"""Exercise the native block-copy implementation on real NPU allocations."""

import torch

from lmdeploy.pytorch.backends.ascend.cache import AscendCacheBackend
from lmdeploy.pytorch.engine.cache_engine.layout import CacheAllocation, CachePool


def test_native_cache_copy_preserves_pools_and_reuses_workspace():
    # Repeated source blocks and different pool axes match the allocation
    # contracts used by KV and additional DSA caches.
    pools = (
        CachePool(torch.arange(120, device='npu', dtype=torch.float32).reshape(2, 12, 5), entry_axis=1),
        CachePool(torch.arange(252, device='npu', dtype=torch.float32).reshape(12, 3, 7), entry_axis=0),
    )
    allocation = CacheAllocation(pools=pools, tensor_views=tuple(p.tensor for p in pools))
    copier = AscendCacheBackend.build_block_copy(allocation, num_logical_blocks=6, pages_per_block=2)
    src = torch.tensor([0, 2, 0], device='npu')
    dst = torch.tensor([3, 1, 4], device='npu')
    before = [p.tensor.cpu().unflatten(p.entry_axis, (6, 2)) for p in pools]
    copier.copy(src, dst)
    torch.npu.synchronize()
    for pool, original in zip(pools, before):
        expected = original.clone()
        expected.index_copy_(pool.entry_axis, dst.cpu(), original.index_select(pool.entry_axis, src.cpu()))
        actual = pool.tensor.cpu().unflatten(pool.entry_axis, (6, 2))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    pointers = [w.data_ptr() for w in copier._workspaces]
    copier.copy(src, dst)
    assert [w.data_ptr() for w in copier._workspaces] == pointers
