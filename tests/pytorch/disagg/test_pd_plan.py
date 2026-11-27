# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from lmdeploy.pytorch.disagg.schema import describe_pd_pools
from lmdeploy.pytorch.engine.cache_engine.layout import CachePool
from lmdeploy.pytorch.engine.cache_engine.migration import build_cache_pool_assignments


@pytest.mark.parametrize('pairs, runs', [([(0, 1), (1, 2), (2, 3)], 1), ([(0, 1), (1, 3), (3, 4)], 3)])
def test_plan_copies_paired_runs_and_keeps_other_entries(pairs, runs):
    source = torch.arange(2 * 6 * 8, dtype=torch.uint8).view(2, 6, 8)
    destination = torch.full((2, 5, 8), 255, dtype=torch.uint8)
    remote = describe_pd_pools(dict(target=[CachePool(source, 1, transfer_ranges=())]))
    local = describe_pd_pools(dict(target=[CachePool(destination, 1)]))
    plan = build_cache_pool_assignments(local, remote, pairs)
    assert len(plan) == 2 * runs
    for task in plan:
        destination.view(-1)[task.source_offset:task.source_offset + task.length].copy_(
            source.view(-1)[task.target_offset:task.target_offset + task.length])
    for src, dst in pairs:
        assert torch.equal(destination[:, dst], source[:, src])
    assert (destination[:, 0] == 255).all()
    with pytest.raises(RuntimeError, match='outside'):
        build_cache_pool_assignments(local, remote, [(6, 1)])
