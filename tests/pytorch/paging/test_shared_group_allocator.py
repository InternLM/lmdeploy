# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
from lmdeploy.pytorch.messages import SequenceMeta, UpdateTokenMode
from lmdeploy.pytorch.paging.block_manager import DefaultBlockManager, SharedBlockManager, build_block_manager
from lmdeploy.pytorch.paging.block_manager.base_block_manager import LogicalAllocator
from lmdeploy.pytorch.paging.block_manager.group_allocator import GroupAllocator
from lmdeploy.pytorch.paging.eviction_helper.recompute_eviction_helper import RecomputeEvictionHelper
from lmdeploy.pytorch.paging.scheduler import Scheduler
from lmdeploy.pytorch.paging.state_manager import build_state_manager
from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy


def test_group_allocator_keeps_tail_private_and_releases_whole_group():
    group_allocator = GroupAllocator(num_gpu_blocks=8, group_size=4, reserve_padding_group=True)
    allocator = LogicalAllocator(num_cpu_blocks=0,
                                 num_gpu_blocks=8,
                                 group_allocator=group_allocator)

    first = group_allocator.acquire_group()
    first_blocks = allocator.allocate_at(np.array([4, 5], dtype=np.int64))
    second = group_allocator.acquire_group()
    second_blocks = allocator.allocate_at(np.array([8], dtype=np.int64))

    assert first.group_id == 1
    assert second.group_id == 2
    assert group_allocator.num_empty_groups == 0
    assert np.array_equal(allocator.get_physical_blocks(first_blocks), [4, 5])

    allocator.free(first_blocks)
    assert group_allocator.group_role(first.group_id) == 'empty'
    current = group_allocator.acquire_group()
    assert current.group_id == first.group_id
    with pytest.raises(RuntimeError, match='stale shared group'):
        group_allocator.release_group(first)
    group_allocator.release_group(current)
    with pytest.raises(RuntimeError, match='live logical blocks'):
        allocator.release_group(second)
    allocator.free(second_blocks)
    assert group_allocator.num_empty_groups == 2


def test_group_allocator_allocate_is_atomic_and_padding_is_protected():
    group_allocator = GroupAllocator(num_gpu_blocks=4, group_size=2, reserve_padding_group=True)
    allocator = LogicalAllocator(num_cpu_blocks=0,
                                 num_gpu_blocks=4,
                                 group_allocator=group_allocator)

    with pytest.raises(ValueError, match='KV-owned'):
        allocator.allocate_at(np.array([0], dtype=np.int64))
    logical_blocks = allocator.allocate(3)
    assert np.array_equal(allocator.get_physical_blocks(logical_blocks), [2, 3, 4])
    assert group_allocator.num_empty_groups == 0
    allocator.free(logical_blocks)
    assert group_allocator.num_empty_groups == 2


def _make_shared_scheduler(*,
                           group_size=4,
                           num_gpu_blocks=8,
                           states_shapes=None,
                           enable_prefix_caching=False,
                           num_state_caches=None):
    cache_config = CacheConfig(max_batches=2,
                               block_size=4,
                               num_cpu_blocks=0,
                               num_gpu_blocks=num_gpu_blocks,
                               num_state_caches=num_state_caches,
                               enable_prefix_caching=enable_prefix_caching,
                               enable_kv_state_cache_sharing=True,
                               arena_units_per_group=group_size,
                               states_shapes=states_shapes or [])
    scheduler_config = SchedulerConfig(max_batches=2,
                                       max_session_len=128,
                                       max_request_output_len=64)
    seq_meta = SequenceMeta(4, strategy=ARSequenceStrategy())
    return Scheduler(scheduler_config, cache_config, seq_meta=seq_meta)


def test_shared_manager_allocates_sequence_private_groups():
    scheduler = _make_shared_scheduler()
    session = scheduler.add_session(0)
    first = session.add_sequence(torch.ones(1, dtype=torch.int64))
    second = session.add_sequence(torch.ones(1, dtype=torch.int64))
    manager = scheduler.block_manager

    manager.allocate(first)
    first.update_token_ids(torch.ones(4, dtype=torch.int64))
    manager.allocate(first)
    manager.allocate(second)

    assert np.array_equal(manager.get_block_table(first), [4, 5])
    assert np.array_equal(manager.get_block_table(second), [8])
    assert manager.get_num_free_gpu_blocks() == 0
    assert manager.get_num_free_cpu_blocks() == 0

    manager.free(first)
    manager.free(second)
    assert manager.get_num_free_gpu_blocks() == 8


def test_shared_manager_discards_group_cursor_after_trie_deduplication():
    scheduler = _make_shared_scheduler(group_size=4, num_gpu_blocks=12)
    scheduler.cache_config.enable_prefix_caching = True
    scheduler.block_trie.enabled = True
    session_a = scheduler.add_session(0)
    session_b = scheduler.add_session(1)
    first = session_a.add_sequence(torch.ones(8, dtype=torch.int64))
    second = session_b.add_sequence(torch.ones(8, dtype=torch.int64))
    manager = scheduler.block_manager

    # Both requests can allocate before either trie insertion runs. The
    # second insertion then deduplicates all of its blocks and releases its
    # now-empty private group.
    manager.allocate(first)
    manager.allocate(second)
    scheduler.block_trie.allocate(first)
    scheduler.block_trie.allocate(second)

    second.update_token_ids(torch.ones(4, dtype=torch.int64), mode=UpdateTokenMode.DECODE)
    manager.allocate(second)
    assert len(manager.get_block_table(second)) == 3


def test_shared_eviction_helper_counts_open_append_slots():
    scheduler = _make_shared_scheduler(group_size=4, num_gpu_blocks=4)
    session = scheduler.add_session(0)
    seq = session.add_sequence(torch.ones(4, dtype=torch.int64))
    manager = scheduler.block_manager

    manager.allocate(seq)
    seq.update_token_ids(torch.ones(4, dtype=torch.int64), mode=UpdateTokenMode.DECODE)

    # The only KV group has three unused append slots even though no complete
    # group is empty.
    assert manager.get_num_free_gpu_blocks() == 0
    assert manager.can_allocate(seq)
    assert scheduler.eviction_helper.try_make_capacity_for(seq, (), 0)


def test_shared_state_manager_maps_logical_slots_to_protected_and_flexible_groups():
    cache_config = CacheConfig(max_batches=1,
                               block_size=4,
                               num_cpu_blocks=0,
                               num_gpu_blocks=8,
                               num_state_caches=3,
                               prefix_cache_state_budget=1,
                               states_shapes=[((1, ), torch.float16)],
                               enable_kv_state_cache_sharing=True,
                               arena_units_per_group=2)
    manager = build_block_manager(cache_config)
    state_manager = build_state_manager(cache_config, manager.group_allocator)

    runtime = type('Sequence', (), {'logical_state': -1})()
    state_manager.allocate(runtime)
    assert runtime.logical_state == 1
    assert state_manager.get_physical_state_id(0) == 1
    assert state_manager.get_physical_state_id(1) == 2

    checkpoint_state = state_manager.allocate_checkpoint_state()
    assert state_manager.get_physical_state_id(checkpoint_state) == 3
    state_manager.free_checkpoint_state(checkpoint_state)
    state_manager.free(runtime)
    assert manager.group_allocator.num_empty_groups == 4


def test_shared_checkpoint_borrow_keeps_protected_runtime_mapping():
    cache_config = CacheConfig(max_batches=2,
                               block_size=4,
                               num_cpu_blocks=0,
                               num_gpu_blocks=4,
                               num_state_caches=4,
                               states_shapes=[((1, ), torch.float16)],
                               enable_kv_state_cache_sharing=True,
                               arena_units_per_group=2)
    manager = build_block_manager(cache_config)
    state_manager = build_state_manager(cache_config, manager.group_allocator)

    checkpoint_state = state_manager.allocate_checkpoint_state()
    assert checkpoint_state == 1
    assert state_manager.get_physical_state_id(checkpoint_state) == 2
    assert manager.group_allocator.group_role(2) == 'protected'

    state_manager.free_checkpoint_state(checkpoint_state)
    runtime = type('Sequence', (), {'logical_state': -1})()
    state_manager.allocate(runtime)
    assert runtime.logical_state == checkpoint_state
    assert state_manager.get_physical_state_id(runtime.logical_state) == 2
    assert manager.group_allocator.group_role(2) == 'protected'


def test_shared_runtime_state_reuses_protected_slot_after_flexible_release():
    cache_config = CacheConfig(max_batches=2,
                               block_size=4,
                               num_cpu_blocks=0,
                               num_gpu_blocks=8,
                               num_state_caches=5,
                               prefix_cache_state_budget=2,
                               states_shapes=[((1, ), torch.float16)],
                               enable_kv_state_cache_sharing=True,
                               arena_units_per_group=2)
    manager = build_block_manager(cache_config)
    state_manager = build_state_manager(cache_config, manager.group_allocator)

    runtime_states = [state_manager.allocate_state() for _ in range(2)]
    checkpoint_states = [state_manager.allocate_checkpoint_state() for _ in range(2)]
    assert runtime_states == [1, 2]
    assert checkpoint_states == [3, 4]

    state_manager.free_state(runtime_states[0])
    state_manager.free_checkpoint_state(checkpoint_states[1])

    recycled_runtime = state_manager.allocate_state()
    assert recycled_runtime == runtime_states[0]
    assert state_manager.get_physical_state_id(recycled_runtime) == 2
    assert manager.group_allocator.group_role(2) == 'protected'


def test_shared_partial_checkpoint_owns_and_releases_a_private_group():
    scheduler = _make_shared_scheduler(
        group_size=2,
        num_gpu_blocks=8,
        states_shapes=[((1, ), torch.float32)],
        enable_prefix_caching=True,
        num_state_caches=2,
    )
    session = scheduler.add_session(0)
    seq = session.add_sequence(torch.ones(6, dtype=torch.int64))
    manager = scheduler.block_manager

    manager.allocate(seq)
    scheduler.block_trie.allocate(seq)
    state_id = scheduler.block_trie.state_checkpoints.reserve_save(seq)
    checkpoint = seq.prefix_cache.pending_save.node.state_checkpoint
    assert state_id >= 0
    assert checkpoint.frozen_block_id >= 0
    frozen_group = int(manager.allocator.get_physical_blocks(np.array([checkpoint.frozen_block_id]))[0]) // 2

    assert scheduler.block_trie.state_checkpoints.publish_save(seq)
    session.remove_sequence(seq)
    assert manager.group_allocator.group_role(frozen_group) == 'kv'
    assert scheduler.block_trie.state_checkpoints.evict_frozen_checkpoints(1) == 1
    assert manager.group_allocator.group_role(frozen_group) == 'empty'


def test_shared_reset_cache_releases_trie_and_checkpoint_groups():
    scheduler = _make_shared_scheduler(
        group_size=2,
        num_gpu_blocks=8,
        states_shapes=[((1, ), torch.float32)],
        enable_prefix_caching=True,
        num_state_caches=3,
    )
    session = scheduler.add_session(0)
    seq = session.add_sequence(torch.ones(6, dtype=torch.int64))
    manager = scheduler.block_manager
    manager.allocate(seq)
    scheduler.block_trie.allocate(seq)
    assert scheduler.block_trie.state_checkpoints.reserve_save(seq) >= 0
    assert scheduler.block_trie.state_checkpoints.publish_save(seq)
    node = seq.prefix_cache.trie_cursor

    scheduler.end_session(session.session_id)
    scheduler.reset_cache()

    assert manager.group_allocator.num_empty_groups == 4
    assert scheduler.state_manager.get_num_allocated_checkpoint_states() == 0
    assert scheduler.block_trie._roots == {}
    assert node.parent is None


def test_shared_block_manager_rejects_cpu_and_window_modes():
    cache_config = CacheConfig(max_batches=1,
                               block_size=4,
                               num_cpu_blocks=1,
                               num_gpu_blocks=4,
                               enable_kv_state_cache_sharing=True)
    with pytest.raises(ValueError, match='num_cpu_blocks=0'):
        build_block_manager(cache_config)
    cache_config = CacheConfig(max_batches=1,
                               block_size=4,
                               num_cpu_blocks=0,
                               num_gpu_blocks=4,
                               window_size=8,
                               enable_kv_state_cache_sharing=True)
    with pytest.raises(ValueError, match='sliding-window'):
        build_block_manager(cache_config)

    cache_config.window_size = None
    manager = build_block_manager(cache_config)
    assert isinstance(manager, SharedBlockManager)
    assert manager.allocator.shared

    cache_config.enable_kv_state_cache_sharing = False
    cache_config.num_gpu_blocks = 4
    cache_config.window_size = -1
    assert isinstance(build_block_manager(cache_config), DefaultBlockManager)


def test_trie_can_evict_one_complete_shared_kv_group():
    scheduler = _make_shared_scheduler(group_size=2, num_gpu_blocks=4)
    scheduler.cache_config.enable_prefix_caching = True
    scheduler.block_trie.enabled = True
    session = scheduler.add_session(0)
    seq = session.add_sequence(torch.ones(8, dtype=torch.int64))
    manager = scheduler.block_manager
    manager.allocate(seq)
    scheduler.block_trie.allocate(seq)
    manager.free(seq)

    assert scheduler.block_trie.evict_one_kv_group() == 2
    assert manager.group_allocator.num_empty_groups == 2


def test_shared_eviction_helper_reclaims_complete_groups():
    free_blocks = [0]
    evictions = []

    block_manager = type('BlockManager', (), {
        'allocator': type('Allocator', (), {'shared': True})(),
        'get_num_free_gpu_blocks': lambda self: free_blocks[0],
    })()

    class _Trie:

        class _StateCheckpoints:

            @staticmethod
            def evict_frozen_checkpoints(limit):
                evictions.append(('frozen', limit))
                return 0

        state_checkpoints = _StateCheckpoints()

        def evict_one_kv_group(self):
            evictions.append(('group', ))
            free_blocks[0] = 4
            return 4

    helper = RecomputeEvictionHelper(block_manager=block_manager,
                                     block_trie=_Trie(),
                                     state_manager=None,
                                     load_coordinator=None,
                                     is_ssm=False)
    assert helper._try_make_block_capacity(1)
    assert evictions == [('frozen', 1), ('group', )]


def test_shared_eviction_helper_does_not_over_evict_open_append_capacity():
    free_blocks = [0]
    evictions = []

    block_manager = type('BlockManager', (), {
        'allocator': type('Allocator', (), {'shared': True})(),
        'get_num_free_gpu_blocks': lambda self: free_blocks[0],
        'can_allocate': lambda self, seq, prealloc_size: free_blocks[0] >= 4,
    })()

    class _Trie:

        class _StateCheckpoints:

            @staticmethod
            def evict_frozen_checkpoints(limit):
                evictions.append(('frozen', limit))
                return 0

        state_checkpoints = _StateCheckpoints()

        def evict_one_kv_group(self):
            evictions.append(('group', ))
            free_blocks[0] += 4
            return 4

    helper = RecomputeEvictionHelper(block_manager=block_manager,
                                     block_trie=_Trie(),
                                     state_manager=None,
                                     load_coordinator=None,
                                     is_ssm=False)
    assert helper._try_make_block_capacity(5, seq=object())
    assert evictions == [('frozen', 1), ('group', )]
