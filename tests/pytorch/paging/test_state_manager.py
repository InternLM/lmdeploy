# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.config import CacheConfig
from lmdeploy.pytorch.paging.state_manager import build_state_manager


def test_reserved_state_cache_is_not_allocatable():
    cache_config = CacheConfig(max_batches=128,
                               block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0,
                               num_state_caches=129,
                               states_shapes=[((1,), torch.float16)])
    state_manager = build_state_manager(cache_config)

    seqs = [SimpleNamespace(logical_state=-1) for _ in range(128)]
    for seq in seqs:
        state_manager.allocate(seq)

    state_ids = [seq.logical_state for seq in seqs]
    assert state_ids == list(range(1, 129))
    assert state_manager.get_num_free() == 0

    with pytest.raises(RuntimeError, match='No free states.'):
        state_manager.allocate(SimpleNamespace(logical_state=-1))


def test_non_ssm_state_manager_without_state_caches():
    cache_config = CacheConfig(max_batches=128,
                               block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0)
    state_manager = build_state_manager(cache_config)

    assert state_manager.get_num_free() == 0
    with pytest.raises(RuntimeError, match='No free states.'):
        state_manager.allocate(SimpleNamespace(logical_state=-1))
