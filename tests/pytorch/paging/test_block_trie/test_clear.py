# Copyright (c) OpenMMLab. All rights reserved.
import pytest


@pytest.mark.parametrize('step', [8, 32, 35])
@pytest.mark.parametrize('published', [False, True])
def test_clear_releases_checkpoints_and_all_kv(ssm_scheduler, step, published):
    scheduler = ssm_scheduler
    trie = scheduler.block_trie
    manager = scheduler.block_manager
    initial_free = manager.get_num_free_gpu_blocks()
    initial_logical_free = manager.allocator.get_num_free_blocks()
    initial_states = scheduler.state_manager.get_num_free()
    nodes = []
    for session_id, adapter in enumerate([None, 'adapter']):
        seq = scheduler.add_session(session_id).add_sequence([1] * step, adapter_name=adapter)
        manager.allocate(seq)
        trie.allocate(seq)
        assert trie.state_checkpoints.reserve_save(seq) >= 0
        nodes.append(seq.prefix_cache.pending_save.node)
        if published:
            assert trie.state_checkpoints.publish_save(seq)
        # Keep unpublished reservations to exercise clear's ownership rather
        # than the usual end-session discard path.
        seq.prefix_cache.pending_save.clear()
        scheduler.end_session(session_id)

    trie.clear()
    trie.clear()  # Idempotent; no duplicate frees.

    assert manager.get_num_free_gpu_blocks() == initial_free
    assert manager.allocator.get_num_free_blocks() == initial_logical_free
    assert scheduler.state_manager.get_num_free() == initial_states
    assert not trie.leaves
    assert not list(trie._checkpoint_index.unique_nodes())
    assert all(node.state_checkpoint is None for node in nodes)
    for session_id, adapter in enumerate([None, 'adapter']):
        seq = scheduler.add_session(session_id).add_sequence([1] * step + [2], adapter_name=adapter)
        trie.match(seq)
        assert seq.num_history_ids == 0
        assert not seq.prefix_cache.restore.is_selected


def test_clear_rejects_live_kv_refs_without_partial_release(scheduler):
    trie = scheduler.block_trie
    seq = scheduler.add_session(0).add_sequence([1] * 32)
    scheduler.block_manager.allocate(seq)
    trie.allocate(seq)
    free = scheduler.block_manager.get_num_free_gpu_blocks()

    with pytest.raises(RuntimeError, match='referenced'):
        trie.clear()
    assert scheduler.block_manager.get_num_free_gpu_blocks() == free
    scheduler.end_session(0)
    trie.clear()
    assert scheduler.block_manager.get_num_free_gpu_blocks() == scheduler.cache_config.num_gpu_blocks


def test_clear_rejects_checkpoint_copy_pins(ssm_scheduler):
    scheduler = ssm_scheduler
    trie = scheduler.block_trie
    seq = scheduler.add_session(0).add_sequence([1] * 32)
    scheduler.block_manager.allocate(seq)
    trie.allocate(seq)
    trie.state_checkpoints.reserve_save(seq)
    trie.state_checkpoints.publish_save(seq, pin_save=True)
    scheduler.end_session(0)
    free = scheduler.block_manager.get_num_free_gpu_blocks()

    with pytest.raises(RuntimeError, match='pinned'):
        trie.clear()
    assert scheduler.block_manager.get_num_free_gpu_blocks() == free
    trie.state_checkpoints.unpin_save(seq)
    trie.clear()
    assert scheduler.state_manager.get_num_allocated_checkpoint_states() == 0
