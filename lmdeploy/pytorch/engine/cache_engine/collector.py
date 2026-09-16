# Copyright (c) OpenMMLab. All rights reserved.
"""Built-operator block-cache request collection and row binding."""

from collections import defaultdict

from torch import nn

from .schema import BlockCacheBinding, BlockCacheRequest, BlockCacheRequestContext


def collect_block_cache_requests(
    model: nn.Module,
    context: BlockCacheRequestContext,
) -> tuple[BlockCacheRequest, ...]:
    """Collect cache requests from built operators and bind compact rows."""
    collected = []

    for module in model.modules():
        get_requests = getattr(module, 'get_block_cache_requests', None)
        if get_requests is None:
            continue
        requests = tuple(get_requests(context))
        if not requests:
            continue

        bind_cache = getattr(module, 'bind_block_cache', None)
        if bind_cache is None:
            raise TypeError(f'{type(module).__name__} declares block caches but cannot bind them.')

        names = set()
        for request in requests:
            if not isinstance(request, BlockCacheRequest):
                raise TypeError(f'{type(module).__name__} returned a non-BlockCacheRequest value.')
            if request.name in names:
                raise ValueError(f'{type(module).__name__} requested block cache {request.name} more than once.')
            names.add(request.name)
            collected.append((bind_cache, request))

    next_row_by_name = defaultdict(int)
    requests = []
    for bind_cache, request in collected:
        row = next_row_by_name[request.name]
        binding = BlockCacheBinding(cache_name=request.name, consumer_row=row)
        bind_cache(binding)
        next_row_by_name[request.name] += 1
        requests.append(request)
    return tuple(requests)
