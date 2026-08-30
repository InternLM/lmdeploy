# Copyright (c) OpenMMLab. All rights reserved.

import torch

from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
from lmdeploy.pytorch.engine.inputs_maker import _make_state_prefix_cache_save_plan
from lmdeploy.pytorch.messages import MessageStatus, SequenceMeta, UpdateTokenMode
from lmdeploy.pytorch.paging.scheduler import Scheduler


def _make_ssm_scheduler(max_batch_size: int = 1, prefix_cache_state_budget: int = 0, num_gpu_blocks: int = 16):
    from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy
    block_size = 16
    cache_config = CacheConfig(max_batches=max_batch_size,
                               block_size=block_size,
                               num_cpu_blocks=4,
                               num_gpu_blocks=num_gpu_blocks,
                               enable_prefix_caching=True,
                               num_state_caches=max_batch_size + 1 + prefix_cache_state_budget,
                               prefix_cache_state_budget=prefix_cache_state_budget,
                               states_shapes=[((1, ), torch.float32)])
    scheduler_config = SchedulerConfig(max_batches=max_batch_size,
                                       max_session_len=128,
                                       max_request_output_len=64,
                                       eviction_type='recompute')
    seq_meta = SequenceMeta(block_size, strategy=ARSequenceStrategy())
    return Scheduler(scheduler_config=scheduler_config, cache_config=cache_config, seq_meta=seq_meta)


def _add_published_ssm_checkpoint(scheduler: Scheduler, token_ids: list[int]):
    session = scheduler.add_session(len(scheduler.sessions))
    seq = session.add_sequence(token_ids)
    scheduler.block_manager.allocate(seq)
    scheduler.block_trie.allocate(seq)
    state_idx = scheduler.block_trie.state_checkpoints.reserve_save(seq)
    assert state_idx >= 0
    assert scheduler.block_trie.state_checkpoints.publish_save(seq)
    node = seq.prefix_cache.trie_cursor
    session.remove_sequence(seq)
    return node, state_idx


def test_ssm_runtime_state_reclaims_borrowed_checkpoint_slot():
    scheduler = _make_ssm_scheduler(max_batch_size=1, prefix_cache_state_budget=0)
    block_size = scheduler.seq_meta.block_size
    node, state_idx = _add_published_ssm_checkpoint(scheduler, [1] * block_size * 2)
    seq = scheduler.add_session(100).add_sequence([2] * block_size * 2)

    output = scheduler.schedule(is_prefill=True)

    assert output.running == [seq]
    assert seq.logical_state == state_idx
    assert node.state_checkpoint is None
    assert scheduler.state_manager.get_num_runtime_states() == 1
    assert scheduler.state_manager.get_num_allocated_checkpoint_states() == 0


def test_ssm_long_chunked_request_schedules_with_only_runtime_state_slot():
    scheduler = _make_ssm_scheduler(max_batch_size=1, prefix_cache_state_budget=0)
    scheduler.cache_config.max_prefill_token_num = scheduler.seq_meta.block_size * 2
    block_size = scheduler.seq_meta.block_size
    token_ids = [1] * block_size + [2] * block_size + [3] * block_size
    seq = scheduler.add_session(100).add_sequence(token_ids)

    output = scheduler.schedule(is_prefill=True)

    assert output.running == [seq]
    assert seq.logical_state >= 0
    assert scheduler.state_manager.get_num_runtime_states() == 1
    assert scheduler.state_manager.get_num_allocated_checkpoint_states() == 0
    assert scheduler.block_trie.state_checkpoints.reserve_save(seq, step=block_size * 2) == -1


def test_ssm_running_request_reuses_own_runtime_state_without_spare_slot():
    scheduler = _make_ssm_scheduler(max_batch_size=1, prefix_cache_state_budget=0)
    block_size = scheduler.seq_meta.block_size
    seq = scheduler.add_session(100).add_sequence([1] * block_size)

    output = scheduler.schedule(is_prefill=True)
    assert output.running == [seq]
    assert scheduler.state_manager.get_num_free_runtime() == 0
    seq.state.activate()

    seq.update_token_ids([2] * block_size, mode=UpdateTokenMode.DECODE)
    valid_mask = scheduler.schedule_running([seq], num_required_tokens=0, prealloc_size=0)

    assert valid_mask == [True]
    assert seq.status == MessageStatus.RUNNING
    assert seq.logical_state >= 0
    assert seq.num_blocks == 2
    assert scheduler.state_manager.get_num_runtime_states() == 1
    assert scheduler.state_manager.get_num_free_runtime() == 0


def test_ssm_runtime_state_waits_when_only_checkpoint_slot_is_pinned():
    scheduler = _make_ssm_scheduler(max_batch_size=1, prefix_cache_state_budget=0)
    block_size = scheduler.seq_meta.block_size
    node, state_idx = _add_published_ssm_checkpoint(scheduler, [1] * block_size * 2)
    node.state_checkpoint.pin_count = 1
    seq = scheduler.add_session(100).add_sequence([2] * block_size * 2)

    output = scheduler.schedule(is_prefill=True)

    assert output.running == []
    assert seq.status == MessageStatus.WAITING
    assert seq.logical_state == -1
    assert node.state_checkpoint.slot == state_idx
    assert node.state_checkpoint.published


def test_ssm_same_batch_duplicate_checkpoint_save_has_unique_dst_offsets():
    scheduler = _make_ssm_scheduler(max_batch_size=2, prefix_cache_state_budget=2)
    block_size = scheduler.seq_meta.block_size
    token_ids = [1] * block_size * 2

    seq_a = scheduler.add_session(100).add_sequence(token_ids)
    seq_b = scheduler.add_session(101).add_sequence(token_ids)

    output = scheduler.schedule(is_prefill=True)
    assert output.running == [seq_a, seq_b]
    assert seq_a.logical_state >= 0
    assert seq_b.logical_state >= 0
    assert seq_a.logical_state != seq_b.logical_state
    assert seq_a.prefix_cache.trie_cursor is seq_b.prefix_cache.trie_cursor

    save_state_offsets = [
        scheduler.block_trie.state_checkpoints.reserve_save(seq) for seq in output.running
    ]
    save_plan = _make_state_prefix_cache_save_plan(output.running, save_state_offsets)
    assert save_plan is not None
    save_src_offsets, save_dst_offsets = save_plan

    assert save_src_offsets == (seq_a.logical_state, )
    assert save_dst_offsets == (save_state_offsets[0], )
    assert save_state_offsets[0] >= 0
    assert save_state_offsets[1] == -1
    assert len(save_dst_offsets) == len(set(save_dst_offsets))


def test_ssm_end_session_discards_pending_checkpoint_reservation():
    scheduler = _make_ssm_scheduler(max_batch_size=1, prefix_cache_state_budget=1)
    block_size = scheduler.seq_meta.block_size
    session = scheduler.add_session(100)
    seq = session.add_sequence([1] * block_size * 2)
    scheduler.block_manager.allocate(seq)
    scheduler.block_trie.allocate(seq)
    scheduler.state_manager.allocate(seq)

    state_idx = scheduler.block_trie.state_checkpoints.reserve_save(seq)
    node = seq.prefix_cache.pending_save.node
    assert state_idx >= 0
    assert node is not None
    assert scheduler.state_manager.get_num_allocated_checkpoint_states() == 1

    scheduler.end_session(100)

    assert 100 not in scheduler.sessions
    assert node.state_checkpoint is None
    assert scheduler.state_manager.get_num_runtime_states() == 0
    assert scheduler.state_manager.get_num_allocated_checkpoint_states() == 0


def test_ssm_end_session_unpins_restore_checkpoint():
    scheduler = _make_ssm_scheduler(max_batch_size=1, prefix_cache_state_budget=1)
    block_size = scheduler.seq_meta.block_size
    node, state_idx = _add_published_ssm_checkpoint(scheduler, [1] * block_size * 2)
    seq = scheduler.add_session(100).add_sequence([1] * block_size * 2 + [2])

    scheduler.block_trie.match(seq)
    assert seq.prefix_cache.restore.slot == state_idx
    assert scheduler.block_trie.state_checkpoints.pin_restore(seq)
    assert node.state_checkpoint.pin_count == 1

    scheduler.end_session(100)

    assert 100 not in scheduler.sessions
    assert node.state_checkpoint.slot == state_idx
    assert node.state_checkpoint.published
    assert node.state_checkpoint.pin_count == 0


def test_ssm_failed_restore_schedule_rolls_back_match():
    scheduler = _make_ssm_scheduler(max_batch_size=1, prefix_cache_state_budget=0)
    block_size = scheduler.seq_meta.block_size
    node, state_idx = _add_published_ssm_checkpoint(scheduler, [1] * block_size * 2)
    node.state_checkpoint.pin_count = 1
    seq = scheduler.add_session(100).add_sequence([1] * block_size * 2 + [2])

    output = scheduler.schedule(is_prefill=True)

    assert output.running == []
    assert seq.status == MessageStatus.WAITING
    assert seq.num_history_ids == 0
    assert len(seq.logical_blocks) == 0
    assert seq.cached_tokens == 0
    assert seq.prefix_cache.trie_cursor is None
    assert seq.prefix_cache.restore.slot == -1
    assert seq.prefix_cache.restore.node is None
    assert node.state_checkpoint.slot == state_idx
    assert node.state_checkpoint.published
    assert scheduler.block_trie.stats.num_query_tokens == 0
    assert scheduler.block_trie.stats.num_hit_tokens == 0

    node.state_checkpoint.pin_count = 0
    output = scheduler.schedule(is_prefill=True)

    assert output.running == [seq]
    assert seq.status == MessageStatus.READY
    assert seq.num_history_ids == 0
    assert seq.prefix_cache.restore.slot == -1
    assert seq.logical_state == state_idx
    assert node.state_checkpoint is None
    assert scheduler.block_trie.stats.num_query_tokens == 0
    assert scheduler.block_trie.stats.num_hit_tokens == 0


def test_ssm_scheduler_preserves_matched_checkpoint_when_evicting_for_runtime_state():
    scheduler = _make_ssm_scheduler(max_batch_size=1, prefix_cache_state_budget=1)
    block_size = scheduler.seq_meta.block_size
    node_a, state_idx_a = _add_published_ssm_checkpoint(scheduler, [1] * block_size * 2)
    node_b, state_idx_b = _add_published_ssm_checkpoint(scheduler, [2] * block_size * 2)
    seq = scheduler.add_session(100).add_sequence([1] * block_size * 2 + [3])

    output = scheduler.schedule(is_prefill=True)

    assert output.running == [seq]
    assert seq.num_history_ids == block_size * 2
    assert seq.cached_tokens == block_size * 2
    assert seq.prefix_cache.restore.slot == state_idx_a
    assert seq.prefix_cache.restore.node is node_a
    assert seq.prefix_cache.restore.pinned
    assert seq.logical_state == state_idx_b
    assert node_a.state_checkpoint.slot == state_idx_a
    assert node_a.state_checkpoint.published
    assert node_a.state_checkpoint.pin_count == 1
    assert node_b.state_checkpoint is None
    assert scheduler.block_trie.stats.num_hit_tokens == block_size * 2

    assert scheduler.block_trie.state_checkpoints.unpin_restore(seq)


def test_ssm_scheduler_evicts_stopped_runtime_state_with_free_checkpoint_slot():
    scheduler = _make_ssm_scheduler(max_batch_size=1, prefix_cache_state_budget=1)
    block_size = scheduler.seq_meta.block_size
    seq_a = scheduler.add_session(100).add_sequence([1] * block_size)

    output = scheduler.schedule(is_prefill=True)
    assert output.running == [seq_a]
    assert seq_a.logical_state >= 0
    assert scheduler.state_manager.get_num_free() == 1
    assert scheduler.state_manager.get_num_free_runtime() == 0

    seq_a.state.stop()
    seq_b = scheduler.add_session(101).add_sequence([2] * block_size)

    output = scheduler.schedule(is_prefill=True)

    assert output.running == [seq_b]
    assert seq_b.logical_state >= 0
    assert seq_a.logical_state == -1
    assert seq_a.status == MessageStatus.STOPPED


def test_ssm_scheduler_rolls_back_prefix_match_for_prefill_gate_without_pinning_restore_state():
    scheduler = _make_ssm_scheduler(max_batch_size=1, prefix_cache_state_budget=1)
    scheduler.cache_config.max_prefill_token_num = scheduler.seq_meta.block_size
    block_size = scheduler.seq_meta.block_size
    node, state_idx = _add_published_ssm_checkpoint(scheduler, [1] * block_size * 2)
    scheduler.block_trie.stats.reset()

    still_long = scheduler.add_session(100).add_sequence([1] * block_size * 2 + [3] * (block_size + 1))

    output = scheduler.schedule(is_prefill=True, allow_long_prefill=False)

    assert output.running == []
    assert still_long.status == MessageStatus.WAITING
    assert still_long.num_history_ids == 0
    assert still_long.cached_tokens == 0
    assert still_long.prefix_cache.trie_cursor is None
    assert still_long.prefix_cache.restore.slot == -1
    assert still_long.prefix_cache.restore.node is None
    assert not still_long.prefix_cache.restore.pinned
    assert node.state_checkpoint.slot == state_idx
    assert node.state_checkpoint.pin_count == 0
    assert scheduler.block_trie.stats.num_query_tokens == 0
    assert scheduler.block_trie.stats.num_hit_tokens == 0


def test_ssm_scheduler_rejects_prefix_match_for_prefill_gate_after_pinned_restore_rollback():
    scheduler = _make_ssm_scheduler(max_batch_size=1, prefix_cache_state_budget=1, num_gpu_blocks=2)
    scheduler.cache_config.max_prefill_token_num = scheduler.seq_meta.block_size
    block_size = scheduler.seq_meta.block_size
    node, state_idx = _add_published_ssm_checkpoint(scheduler, [1] * block_size * 2)
    scheduler.block_trie.stats.reset()

    cache_hit_tail = scheduler.add_session(100).add_sequence([1] * block_size * 2 + [3])

    output = scheduler.schedule(is_prefill=True, allow_long_prefill=False)

    assert output.running == []
    assert cache_hit_tail.status == MessageStatus.WAITING
    assert cache_hit_tail.num_history_ids == 0
    assert cache_hit_tail.num_token_ids == block_size * 2 + 1
    assert cache_hit_tail.num_blocks == 0
    assert cache_hit_tail.kv_token_limit is None
    assert cache_hit_tail.logical_state == -1
    assert cache_hit_tail.cached_tokens == 0
    assert cache_hit_tail.prefix_cache.trie_cursor is None
    assert cache_hit_tail.prefix_cache.restore.slot == -1
    assert cache_hit_tail.prefix_cache.restore.node is None
    assert not cache_hit_tail.prefix_cache.restore.pinned
    assert node.state_checkpoint.slot == state_idx
    assert node.state_checkpoint.published
    assert node.state_checkpoint.pin_count == 0
    assert scheduler.block_trie.stats.num_query_tokens == 0
    assert scheduler.block_trie.stats.num_hit_tokens == 0


def test_ssm_scheduler_rejects_prefix_match_for_prefill_gate_after_runtime_state_rollback(monkeypatch):
    scheduler = _make_ssm_scheduler(max_batch_size=1, prefix_cache_state_budget=1, num_gpu_blocks=4)
    scheduler.cache_config.max_prefill_token_num = scheduler.seq_meta.block_size
    block_size = scheduler.seq_meta.block_size
    node, state_idx = _add_published_ssm_checkpoint(scheduler, [1] * block_size * 2)
    ensure_results = iter([False, True])

    def _ensure_runtime_state_available_once_then_succeed():
        return next(ensure_results)

    monkeypatch.setattr(scheduler._prefill_scheduler,
                        '_ensure_runtime_state_available',
                        _ensure_runtime_state_available_once_then_succeed)
    scheduler.block_trie.stats.reset()

    cache_hit_tail = scheduler.add_session(100).add_sequence([1] * block_size * 2 + [3])

    output = scheduler.schedule(is_prefill=True, allow_long_prefill=False)

    assert output.running == []
    assert cache_hit_tail.status == MessageStatus.WAITING
    assert cache_hit_tail.num_history_ids == 0
    assert cache_hit_tail.num_token_ids == block_size * 2 + 1
    assert cache_hit_tail.num_blocks == 0
    assert cache_hit_tail.kv_token_limit is None
    assert cache_hit_tail.logical_state == -1
    assert cache_hit_tail.cached_tokens == 0
    assert cache_hit_tail.prefix_cache.trie_cursor is None
    assert cache_hit_tail.prefix_cache.restore.slot == -1
    assert cache_hit_tail.prefix_cache.restore.node is None
    assert not cache_hit_tail.prefix_cache.restore.pinned
    assert node.state_checkpoint.slot == state_idx
    assert node.state_checkpoint.published
    assert node.state_checkpoint.pin_count == 0
    assert scheduler.block_trie.stats.num_query_tokens == 0
    assert scheduler.block_trie.stats.num_hit_tokens == 0


def _make_ssm_scheduler_for_long_context_chunks(num_gpu_blocks: int = 2):
    from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy
    block_size = 4
    seq_meta = SequenceMeta(block_size, strategy=ARSequenceStrategy())
    cache_config = CacheConfig(max_batches=1,
                               block_size=block_size,
                               num_cpu_blocks=0,
                               num_gpu_blocks=num_gpu_blocks,
                               max_prefill_token_num=block_size * 2,
                               num_state_caches=2,
                               states_shapes=[((1, ), torch.float32)])
    scheduler_config = SchedulerConfig(max_batches=1,
                                       max_session_len=64,
                                       max_request_output_len=64,
                                       eviction_type='recompute')
    scheduler = Scheduler(scheduler_config=scheduler_config, cache_config=cache_config, seq_meta=seq_meta)
    return scheduler, block_size


def test_schedule_prefill_reapplies_chunk_limit_after_ssm_state_rollback():
    scheduler, block_size = _make_ssm_scheduler_for_long_context_chunks(num_gpu_blocks=2)
    long_seq = scheduler.add_session(100).add_sequence([1] * (block_size * 4))

    ensure_results = iter([False, True])

    def _ensure_runtime_state_available_once_then_succeed():
        return next(ensure_results)

    scheduler._prefill_scheduler._ensure_runtime_state_available = (
        _ensure_runtime_state_available_once_then_succeed)

    output = scheduler.schedule(is_prefill=True, prealloc_size=1)

    assert output.running == [long_seq]
    assert long_seq.status == MessageStatus.READY
    assert long_seq.kv_token_limit == block_size * 2
    assert long_seq.num_blocks == 2
