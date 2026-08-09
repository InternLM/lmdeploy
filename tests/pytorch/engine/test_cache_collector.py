# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from lmdeploy.pytorch.engine.cache_engine.collector import collect_block_cache_requests
from lmdeploy.pytorch.engine.cache_engine.schema import BlockCacheGeometry, BlockCacheRequest


class _CacheRequester(nn.Module):

    def __init__(self, *requests):
        super().__init__()
        self.requests = requests
        self.bindings = []

    def get_block_cache_requests(self, geometry):
        assert geometry.kernel_block_size == 64
        return self.requests

    def bind_block_cache_row(self, name, row):
        self.bindings.append((name, row))


def test_collect_block_cache_requests_distinguishes_absent_and_empty_requesters():
    geometry = BlockCacheGeometry(block_size=64, kernel_block_size=64)

    assert collect_block_cache_requests(nn.Linear(2, 2), geometry) is None
    assert collect_block_cache_requests(_CacheRequester(), geometry) == ()


def test_collect_block_cache_requests_binds_rows_by_resource_name():
    geometry = BlockCacheGeometry(block_size=64, kernel_block_size=64)
    index = BlockCacheRequest('index', (64, 8), torch.float16)
    scale = BlockCacheRequest('scale', (64, 1), torch.float32)
    first = _CacheRequester(index, scale)
    second = _CacheRequester(index)
    model = nn.ModuleList([first, second])

    requests = collect_block_cache_requests(model, geometry)

    assert requests == (index, scale, index)
    assert first.bindings == [('index', 0), ('scale', 0)]
    assert second.bindings == [('index', 1)]
