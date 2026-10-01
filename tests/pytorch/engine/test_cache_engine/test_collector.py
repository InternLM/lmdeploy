# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from lmdeploy.pytorch.engine.cache_engine.collector import collect_block_cache_requests
from lmdeploy.pytorch.engine.cache_engine.schema import (
    BlockCacheBinding,
    BlockCacheGeometry,
    BlockCacheRequest,
    BlockCacheRequestContext,
)


class _CacheRequester(nn.Module):

    def __init__(self, *requests):
        super().__init__()
        self.requests = requests
        self.bindings = []

    def get_block_cache_requests(self, context):
        assert context.geometry.kernel_block_size == 64
        return self.requests

    def bind_block_cache(self, binding):
        self.bindings.append(binding)


def test_collect_block_cache_requests_returns_empty_without_requests():
    geometry = BlockCacheGeometry(logical_block_size=64, kernel_block_size=64)
    context = BlockCacheRequestContext(geometry=geometry)

    assert collect_block_cache_requests(nn.Linear(2, 2), context) == ()
    assert collect_block_cache_requests(_CacheRequester(), context) == ()


def test_collect_block_cache_requests_binds_rows_by_cache_name():
    geometry = BlockCacheGeometry(logical_block_size=64, kernel_block_size=64)
    context = BlockCacheRequestContext(geometry=geometry)
    index = BlockCacheRequest('index', (64, 8), torch.float16)
    scale = BlockCacheRequest('scale', (64, 1), torch.float32)
    first = _CacheRequester(index, scale)
    second = _CacheRequester(index)
    model = nn.ModuleList([first, second])

    requests = collect_block_cache_requests(model, context)

    assert requests == (index, scale, index)
    assert first.bindings == [
        BlockCacheBinding(cache_name='index', consumer_row=0),
        BlockCacheBinding(cache_name='scale', consumer_row=0),
    ]
    assert second.bindings == [BlockCacheBinding(cache_name='index', consumer_row=1)]


def test_collect_block_cache_requests_keeps_rows_stable_across_contracts():
    geometry = BlockCacheGeometry(logical_block_size=64, kernel_block_size=64)
    context = BlockCacheRequestContext(geometry=geometry)
    narrow = BlockCacheRequest('index', (64, 8), torch.float16)
    wide = BlockCacheRequest('index', (64, 16), torch.float16)
    consumers = [_CacheRequester(narrow), _CacheRequester(wide), _CacheRequester(narrow)]

    requests = collect_block_cache_requests(nn.ModuleList(consumers), context)

    assert requests == (narrow, wide, narrow)
    assert [consumer.bindings for consumer in consumers] == [
        [BlockCacheBinding(cache_name='index', consumer_row=0)],
        [BlockCacheBinding(cache_name='index', consumer_row=1)],
        [BlockCacheBinding(cache_name='index', consumer_row=2)],
    ]
