# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.config import CacheConfig
from lmdeploy.pytorch.paging.state_manager import StateManager, build_state_manager


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


def test_spare_state_cache_is_allocatable_for_runtime():
    cache_config = CacheConfig(max_batches=128,
                               block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0,
                               num_state_caches=130,
                               states_shapes=[((1,), torch.float16)])
    state_manager = build_state_manager(cache_config)

    seqs = [SimpleNamespace(logical_state=-1) for _ in range(129)]
    for seq in seqs:
        state_manager.allocate(seq)

    state_ids = [seq.logical_state for seq in seqs]
    assert state_ids == list(range(1, 130))
    assert state_manager.get_num_free_runtime() == 0
    assert state_manager.get_num_free() == 0

    with pytest.raises(RuntimeError, match='No free states.'):
        state_manager.allocate(SimpleNamespace(logical_state=-1))


def test_prefix_cache_budget_is_not_counted_as_runtime_spare():
    cache_config = CacheConfig(max_batches=1,
                               block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0,
                               num_state_caches=3,
                               prefix_cache_state_budget=1,
                               states_shapes=[((1,), torch.float16)])
    state_manager = build_state_manager(cache_config)

    seq = SimpleNamespace(logical_state=-1)
    state_manager.allocate(seq)

    assert state_manager.get_num_runtime_states() == 1
    assert state_manager.get_num_free_runtime() == 0
    assert state_manager.get_num_free_checkpoint() == 1

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


def test_state_manager_reserves_system_state_slot():
    manager = StateManager(num_states=3, num_reserved=1)

    assert manager.allocate_state() == 1
    assert manager.allocate_state() == 2
    with pytest.raises(RuntimeError, match='No free states'):
        manager.allocate_state()


def test_state_manager_checkpoint_can_borrow_idle_runtime_slots():
    manager = StateManager(num_states=5, num_reserved=1, num_runtime_states=2)

    checkpoints = [manager.allocate_checkpoint_state() for _ in range(4)]
    assert checkpoints == [1, 2, 3, 4]
    with pytest.raises(RuntimeError, match='No free states'):
        manager.allocate_checkpoint_state()

    manager.free_checkpoint_state(checkpoints[0])
    manager.free_checkpoint_state(checkpoints[1])
    assert manager.allocate_state() == checkpoints[1]
    assert manager.allocate_state() == checkpoints[0]
    with pytest.raises(RuntimeError, match='No free states'):
        manager.allocate_state()


def test_state_manager_caps_runtime_count_even_with_extra_free_slots():
    manager = StateManager(num_states=6, num_reserved=1, num_runtime_states=2)

    assert manager.num_runtime_states == 2
    assert manager.allocate_state() == 1
    assert manager.allocate_state() == 2
    assert manager.get_num_free() == 3
    assert manager.get_num_free_runtime() == 0
    with pytest.raises(RuntimeError, match='No free states'):
        manager.allocate_state()
