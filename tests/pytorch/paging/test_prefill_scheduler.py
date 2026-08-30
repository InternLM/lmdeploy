# Copyright (c) OpenMMLab. All rights reserved.
import time
from unittest.mock import Mock

import torch

import lmdeploy.pytorch.paging.prefill_scheduler as prefill_scheduler_module
from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
from lmdeploy.pytorch.messages import MessageStatus, SequenceMeta, UpdateTokenMode
from lmdeploy.pytorch.paging.scheduler import Scheduler


def test_scheduler_publishes_cached_tokens_for_accepted_prefix_hit():
    from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy
    block_size = 16
    seq_meta = SequenceMeta(block_size, strategy=ARSequenceStrategy())
    cache_config = CacheConfig(max_batches=1,
                               block_size=block_size,
                               num_cpu_blocks=0,
                               num_gpu_blocks=8,
                               enable_prefix_caching=True)
    scheduler_config = SchedulerConfig(max_batches=1,
                                       max_session_len=128,
                                       max_request_output_len=64,
                                       eviction_type='recompute')
    scheduler = Scheduler(scheduler_config=scheduler_config, cache_config=cache_config, seq_meta=seq_meta)

    cached = scheduler.add_session(0).add_sequence([1] * block_size + [2] * block_size + [3])
    scheduler.schedule(is_prefill=True)
    cached.state.stop()

    seq = scheduler.add_session(1).add_sequence([1] * block_size + [2] * block_size + [4])
    output = scheduler.schedule(is_prefill=True)

    assert output.running == [seq]
    assert seq.num_history_ids == block_size * 2
    assert seq.cached_tokens == block_size * 2

    seq.update_token_ids(torch.tensor([5]))

    assert seq.cached_tokens == 0
    assert seq.prefix_cache.match_start_step == -1


def test_scheduler_ar_spec_prefix_hit_recomputes_overlap_block():
    from lmdeploy.pytorch.strategies.ar_spec.sequence import ARSpecSequenceStrategy
    block_size = 16
    seq_meta = SequenceMeta(block_size, strategy=ARSpecSequenceStrategy())
    cache_config = CacheConfig(max_batches=1,
                               block_size=block_size,
                               num_cpu_blocks=0,
                               num_gpu_blocks=8,
                               enable_prefix_caching=True)
    scheduler_config = SchedulerConfig(max_batches=1,
                                       max_session_len=128,
                                       max_request_output_len=64,
                                       eviction_type='recompute')
    scheduler = Scheduler(scheduler_config=scheduler_config, cache_config=cache_config, seq_meta=seq_meta)

    token_ids = [1] * block_size + [2] * block_size + [3] * block_size + [4]
    cached = scheduler.add_session(0).add_sequence(token_ids)
    scheduler.block_manager.allocate(cached)
    scheduler.block_trie.allocate(cached)
    cached_blocks = cached.logical_blocks.get_real_blocks().copy()
    cached.state.stop()

    seq = scheduler.add_session(1).add_sequence(token_ids)
    scheduler.block_trie.stats.reset()

    output = scheduler.schedule(is_prefill=True)

    assert output.running == [seq]
    assert seq.prefix_cache.recompute_overlap.recompute_blocks == 1
    assert seq.num_history_ids == block_size * 2
    assert seq.cached_tokens == block_size * 2
    assert seq.logical_blocks[2] != cached_blocks[2]
    assert seq.prefix_cache.recompute_overlap.fresh_block_range is None
    assert scheduler.block_trie.stats.num_query_tokens == len(token_ids)
    assert scheduler.block_trie.stats.num_hit_tokens == block_size * 2


def test_scheduler_prefix_match_rollback_clears_recompute_overlap_window(monkeypatch):
    from lmdeploy.pytorch.strategies.ar_spec.sequence import ARSpecSequenceStrategy
    block_size = 16
    seq_meta = SequenceMeta(block_size, strategy=ARSpecSequenceStrategy())
    cache_config = CacheConfig(max_batches=1,
                               block_size=block_size,
                               num_cpu_blocks=0,
                               num_gpu_blocks=8,
                               enable_prefix_caching=True)
    scheduler_config = SchedulerConfig(max_batches=1,
                                       max_session_len=128,
                                       max_request_output_len=64,
                                       eviction_type='recompute')
    scheduler = Scheduler(scheduler_config=scheduler_config, cache_config=cache_config, seq_meta=seq_meta)

    token_ids = [1] * block_size + [2] * block_size + [3] * block_size + [4]
    cached = scheduler.add_session(0).add_sequence(token_ids)
    scheduler.block_manager.allocate(cached)
    scheduler.block_trie.allocate(cached)
    cached.state.stop()

    seq = scheduler.add_session(1).add_sequence(token_ids)
    monkeypatch.setattr(scheduler.eviction_helper, 'evict_for_seq', Mock(return_value=False))
    scheduler.block_trie.stats.reset()

    output = scheduler.schedule(is_prefill=True)

    assert output.running == []
    assert seq.num_history_ids == 0
    assert seq.num_token_ids == len(token_ids)
    assert seq.cached_tokens == 0
    assert seq.prefix_cache.recompute_overlap.fresh_block_range is None
    assert scheduler.block_trie.stats.num_query_tokens == 0
    assert scheduler.block_trie.stats.num_hit_tokens == 0


def test_scheduler_recomputes_prefill_budget_after_prefix_hit():
    from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy
    block_size = 16
    seq_meta = SequenceMeta(block_size, strategy=ARSequenceStrategy())
    cache_config = CacheConfig(max_batches=2,
                               block_size=block_size,
                               num_cpu_blocks=0,
                               num_gpu_blocks=8,
                               max_prefill_token_num=block_size,
                               enable_prefix_caching=True)
    scheduler_config = SchedulerConfig(max_batches=2,
                                       max_session_len=128,
                                       max_request_output_len=64,
                                       eviction_type='recompute')
    scheduler = Scheduler(scheduler_config=scheduler_config, cache_config=cache_config, seq_meta=seq_meta)

    cached = scheduler.add_session(0).add_sequence([1] * block_size + [2])
    scheduler.schedule(is_prefill=True)
    cached.state.stop()

    cache_hit_tail = scheduler.add_session(1).add_sequence([1] * block_size + [3])
    short = scheduler.add_session(2).add_sequence([4])

    output = scheduler.schedule(is_prefill=True)

    assert output.running == [cache_hit_tail, short]
    assert cache_hit_tail.num_history_ids == block_size
    assert cache_hit_tail.num_token_ids == 1
    assert short.status == MessageStatus.READY


def _make_prefix_cache_scheduler(max_batches: int = 2, max_prefill_token_num: int = 16):
    from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy
    block_size = 16
    seq_meta = SequenceMeta(block_size, strategy=ARSequenceStrategy())
    cache_config = CacheConfig(max_batches=max_batches,
                               block_size=block_size,
                               num_cpu_blocks=0,
                               num_gpu_blocks=8,
                               max_prefill_token_num=max_prefill_token_num,
                               enable_prefix_caching=True)
    scheduler_config = SchedulerConfig(max_batches=max_batches,
                                       max_session_len=128,
                                       max_request_output_len=64,
                                       eviction_type='recompute')
    scheduler = Scheduler(scheduler_config=scheduler_config, cache_config=cache_config, seq_meta=seq_meta)
    return scheduler, block_size


def test_scheduler_short_turn_uses_prefix_hit_to_admit_long_looking_sibling():
    scheduler, block_size = _make_prefix_cache_scheduler(max_batches=2, max_prefill_token_num=16)

    cached = scheduler.add_session(0).add_sequence([1] * block_size)
    scheduler.schedule(is_prefill=True)
    cached.state.stop()

    short = scheduler.add_session(1).add_sequence([4])
    cache_hit_tail = scheduler.add_session(2).add_sequence([1] * block_size + [3])

    output = scheduler.schedule(is_prefill=True, allow_long_prefill=False)

    assert output.running == [short, cache_hit_tail]
    assert cache_hit_tail.num_history_ids == block_size
    assert cache_hit_tail.num_token_ids == 1
    assert cache_hit_tail.cached_tokens == block_size


def test_scheduler_budget_gate_uses_prefix_hit_to_admit_sibling():
    scheduler, block_size = _make_prefix_cache_scheduler(max_batches=2, max_prefill_token_num=16)

    cached = scheduler.add_session(0).add_sequence([1] * block_size)
    scheduler.schedule(is_prefill=True)
    cached.state.stop()

    almost_full = scheduler.add_session(1).add_sequence([4] * (block_size - 1))
    cache_hit_tail = scheduler.add_session(2).add_sequence([1] * block_size + [3])

    output = scheduler.schedule(is_prefill=True)

    assert output.running == [almost_full, cache_hit_tail]
    assert cache_hit_tail.num_history_ids == block_size
    assert cache_hit_tail.num_token_ids == 1


def test_scheduler_reorder_cache_stays_order_only_after_prefix_hit():
    scheduler, block_size = _make_prefix_cache_scheduler(max_batches=2, max_prefill_token_num=16)

    cached = scheduler.add_session(0).add_sequence([1] * block_size)
    scheduler.schedule(is_prefill=True)
    cached.state.stop()

    cache_hit_tail = scheduler.add_session(1).add_sequence([1] * block_size + [3])
    normal = scheduler.add_session(2).add_sequence([4] * (block_size - 1))

    output = scheduler.schedule(is_prefill=True, prefer_long_prefill=True)

    assert output.running == [cache_hit_tail, normal]
    assert cache_hit_tail.num_history_ids == block_size
    assert cache_hit_tail.num_token_ids == 1
    assert cache_hit_tail.cached_tokens == block_size
    assert normal.status == MessageStatus.READY


def test_scheduler_resource_rejection_rolls_back_tentative_prefix_match(monkeypatch):
    scheduler, block_size = _make_prefix_cache_scheduler(max_batches=1)

    cached = scheduler.add_session(0).add_sequence([1] * block_size + [2])
    scheduler.schedule(is_prefill=True)
    cached.state.stop()
    cached_block = cached.logical_blocks.get_real_blocks()[:1]
    ref_count = scheduler.block_manager.allocator.get_ref_count(cached_block).copy()
    scheduler.block_trie.stats.reset()

    seq = scheduler.add_session(1).add_sequence([1] * block_size + [3])
    evict_for_seq = Mock(return_value=False)
    monkeypatch.setattr(scheduler.eviction_helper, 'evict_for_seq', evict_for_seq)

    output = scheduler.schedule(is_prefill=True)

    assert output.running == []
    assert seq.status == MessageStatus.WAITING
    assert seq.num_history_ids == 0
    assert seq.num_blocks == 0
    assert seq.kv_token_limit is None
    assert seq.cached_tokens == 0
    assert seq.prefix_cache.trie_cursor is None
    assert seq.prefix_cache.match_start_step == -1
    assert evict_for_seq.call_count == 1
    assert scheduler.block_manager.allocator.get_ref_count(cached_block).tolist() == ref_count.tolist()
    assert scheduler.block_trie.stats.num_query_tokens == 0
    assert scheduler.block_trie.stats.num_hit_tokens == 0


def test_scheduler_rolls_back_prefix_match_for_prefill_gate_when_tail_still_exceeds_budget():
    scheduler, block_size = _make_prefix_cache_scheduler(max_batches=2, max_prefill_token_num=16)

    cached = scheduler.add_session(0).add_sequence([1] * block_size)
    scheduler.schedule(is_prefill=True)
    cached.state.stop()

    full = scheduler.add_session(1).add_sequence([4] * block_size)
    cache_hit_tail = scheduler.add_session(2).add_sequence([1] * block_size + [3])

    output = scheduler.schedule(is_prefill=True, allow_long_prefill=False)

    assert output.running == [full]
    assert cache_hit_tail.status == MessageStatus.WAITING
    assert cache_hit_tail.num_history_ids == 0
    assert cache_hit_tail.cached_tokens == 0
    assert cache_hit_tail.prefix_cache.trie_cursor is None
    assert cache_hit_tail.prefix_cache.match_start_step == -1


def test_scheduler_rolls_back_prefix_match_for_prefill_gate_that_still_needs_long_chunk():
    scheduler, block_size = _make_prefix_cache_scheduler(max_batches=1, max_prefill_token_num=16)

    cached = scheduler.add_session(0).add_sequence([1] * block_size)
    scheduler.schedule(is_prefill=True)
    cached.state.stop()
    scheduler.block_trie.stats.reset()

    still_long = scheduler.add_session(1).add_sequence([1] * block_size + [3] * (block_size + 1))

    output = scheduler.schedule(is_prefill=True, allow_long_prefill=False)

    assert output.running == []
    assert still_long.status == MessageStatus.WAITING
    assert still_long.num_history_ids == 0
    assert still_long.cached_tokens == 0
    assert still_long.prefix_cache.trie_cursor is None
    assert still_long.prefix_cache.match_start_step == -1
    assert scheduler.block_trie.stats.num_query_tokens == 0
    assert scheduler.block_trie.stats.num_hit_tokens == 0


def test_scheduler_reports_zero_cached_tokens_for_prefix_miss():
    from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy
    block_size = 16
    seq_meta = SequenceMeta(block_size, strategy=ARSequenceStrategy())
    cache_config = CacheConfig(max_batches=1,
                               block_size=block_size,
                               num_cpu_blocks=0,
                               num_gpu_blocks=8,
                               enable_prefix_caching=True)
    scheduler_config = SchedulerConfig(max_batches=1,
                                       max_session_len=128,
                                       max_request_output_len=64,
                                       eviction_type='recompute')
    scheduler = Scheduler(scheduler_config=scheduler_config, cache_config=cache_config, seq_meta=seq_meta)

    cached = scheduler.add_session(0).add_sequence([1] * block_size + [2])
    scheduler.schedule(is_prefill=True)
    cached.state.stop()

    seq = scheduler.add_session(1).add_sequence([3] * block_size + [4])
    output = scheduler.schedule(is_prefill=True)

    assert output.running == [seq]
    assert seq.num_history_ids == 0
    assert seq.cached_tokens == 0


def test_scheduler_cached_tokens_only_count_current_prompt_after_session_eviction():
    from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy
    block_size = 16
    seq_meta = SequenceMeta(block_size, strategy=ARSequenceStrategy())
    cache_config = CacheConfig(max_batches=1,
                               block_size=block_size,
                               num_cpu_blocks=0,
                               num_gpu_blocks=8,
                               enable_prefix_caching=True)
    scheduler_config = SchedulerConfig(max_batches=1,
                                       max_session_len=128,
                                       max_request_output_len=64,
                                       eviction_type='recompute')
    scheduler = Scheduler(scheduler_config=scheduler_config, cache_config=cache_config, seq_meta=seq_meta)

    session = scheduler.add_session(0)
    seq = session.add_sequence([1] * block_size + [2] * block_size + [3])
    scheduler.schedule(is_prefill=True)
    seq.update_token_ids(torch.tensor([9]), mode=UpdateTokenMode.PREFILL)
    seq.state.stop()
    seq.state.free()

    seq.update_token_ids(torch.tensor([4] * 4))
    assert seq.input_start_pos == block_size * 2 + 2
    assert seq.input_end_pos == block_size * 2 + 6
    seq.state.activate()

    output = scheduler.schedule(is_prefill=True)

    assert output.running == [seq]
    assert seq.num_history_ids == block_size * 2
    assert seq.cached_tokens == 0


def test_scheduler_excludes_recompute_eviction_prefix_hits_from_stats():
    from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy
    block_size = 16
    seq_meta = SequenceMeta(block_size, strategy=ARSequenceStrategy())
    cache_config = CacheConfig(max_batches=1,
                               block_size=block_size,
                               num_cpu_blocks=0,
                               num_gpu_blocks=4,
                               enable_prefix_caching=True)
    scheduler_config = SchedulerConfig(max_batches=1,
                                       max_session_len=128,
                                       max_request_output_len=64,
                                       eviction_type='recompute')
    scheduler = Scheduler(scheduler_config=scheduler_config, cache_config=cache_config, seq_meta=seq_meta)

    seq = scheduler.add_session(0).add_sequence([1] * block_size + [2] * block_size + [3])
    output = scheduler.schedule(is_prefill=True)
    assert output.running == [seq]

    seq.state.evict()
    pressure = scheduler.add_session(1).add_sequence([9] * block_size * 3)
    scheduler.block_trie.stats.reset()

    assert scheduler.eviction_helper.evict_for_seq(pressure, [seq], 0)
    assert seq.prefix_cache.suppress_match_stats
    pressure.session.remove_sequence(pressure)

    output = scheduler.schedule(is_prefill=True)

    assert output.running == [seq]
    assert seq.num_history_ids >= block_size
    assert seq.cached_tokens == 0
    assert not seq.prefix_cache.suppress_match_stats
    assert scheduler.block_trie.stats.num_query_tokens == 0
    assert scheduler.block_trie.stats.num_hit_tokens == 0


def _make_scheduler_for_long_context_chunks(num_gpu_blocks: int = 6):
    from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy
    block_size = 4
    seq_meta = SequenceMeta(block_size, strategy=ARSequenceStrategy())
    cache_config = CacheConfig(max_batches=2,
                               block_size=block_size,
                               num_cpu_blocks=0,
                               num_gpu_blocks=num_gpu_blocks,
                               max_prefill_token_num=block_size * 2)
    scheduler_config = SchedulerConfig(max_batches=2,
                                       max_session_len=64,
                                       max_request_output_len=64,
                                       eviction_type='recompute')
    scheduler = Scheduler(scheduler_config=scheduler_config, cache_config=cache_config, seq_meta=seq_meta)
    return scheduler, block_size


def test_schedule_prefill_allocates_only_first_long_context_chunk():
    scheduler, block_size = _make_scheduler_for_long_context_chunks(num_gpu_blocks=2)
    long_seq = scheduler.add_session(100).add_sequence([1] * (block_size * 4))

    output = scheduler.schedule(is_prefill=True, prealloc_size=1)

    assert output.running == [long_seq]
    assert long_seq.status == MessageStatus.READY
    assert long_seq.kv_token_limit == block_size * 2
    assert long_seq.num_blocks == 2
    assert scheduler.block_manager.get_num_free_gpu_blocks() == 0


def test_schedule_prefill_short_only_skips_long_waiter_without_mutation():
    scheduler, block_size = _make_scheduler_for_long_context_chunks(num_gpu_blocks=8)
    head_long = scheduler.add_session(100).add_sequence([1] * (block_size * 4))
    short_a = scheduler.add_session(101).add_sequence([2] * (block_size // 2))
    short_b = scheduler.add_session(102).add_sequence([3] * (block_size // 2))

    output = scheduler.schedule(is_prefill=True, allow_long_prefill=False)

    assert output.running == [short_a, short_b]
    assert head_long.status == MessageStatus.WAITING
    assert head_long.num_blocks == 0
    assert head_long.kv_token_limit is None
    assert short_a.status == MessageStatus.READY
    assert short_b.status == MessageStatus.READY

    short_a.session.remove_sequence(short_a)
    short_b.session.remove_sequence(short_b)
    next_output = scheduler.schedule(is_prefill=True)

    assert next_output.running == [head_long]
    assert head_long.status == MessageStatus.READY
    assert head_long.kv_token_limit == block_size * 2
    assert head_long.num_blocks == 2


def test_schedule_prefill_prefer_long_admits_oldest_long_waiter_first():
    scheduler, block_size = _make_scheduler_for_long_context_chunks(num_gpu_blocks=8)
    short_a = scheduler.add_session(100).add_sequence([1] * (block_size // 2))
    old_long = scheduler.add_session(101).add_sequence([2] * (block_size * 4))
    short_b = scheduler.add_session(102).add_sequence([3] * (block_size // 2))
    new_long = scheduler.add_session(103).add_sequence([4] * (block_size * 4))

    assert scheduler.has_waiting_long_prefill()

    output = scheduler.schedule(is_prefill=True, prefer_long_prefill=True)

    assert output.running == [old_long]
    assert old_long.status == MessageStatus.READY
    assert old_long.kv_token_limit == block_size * 2
    assert old_long.num_blocks == 2
    assert short_a.status == MessageStatus.WAITING
    assert short_a.num_blocks == 0
    assert short_b.status == MessageStatus.WAITING
    assert short_b.num_blocks == 0
    assert new_long.status == MessageStatus.WAITING
    assert new_long.num_blocks == 0
    assert new_long.kv_token_limit is None


def test_scheduler_reads_opt_ttft_env(monkeypatch):
    monkeypatch.setattr(prefill_scheduler_module._envs, 'opt_ttft_policy',
                        'fifo')
    monkeypatch.setattr(prefill_scheduler_module._envs, 'opt_ttft_aging_sec',
                        0.25)

    scheduler, _ = _make_scheduler_for_long_context_chunks(num_gpu_blocks=8)

    assert scheduler._prefill_scheduler._long_prefill_policy == 'fifo'
    assert scheduler._prefill_scheduler._long_prefill_aging_seconds_per_chunk == 0.25


def test_schedule_prefill_prefer_long_fifo_policy_keeps_oldest_huge_waiter_first():
    scheduler, block_size = _make_scheduler_for_long_context_chunks(num_gpu_blocks=8)
    scheduler._prefill_scheduler._long_prefill_policy = 'fifo'
    now = time.perf_counter()
    huge_long = scheduler.add_session(100).add_sequence([1] * (block_size * 16))
    huge_long.arrive_time = now - 1.0
    moderate_long = scheduler.add_session(101).add_sequence([2] * (block_size * 4))
    moderate_long.arrive_time = now

    output = scheduler.schedule(is_prefill=True, prefer_long_prefill=True)

    assert output.running == [huge_long]
    assert huge_long.status == MessageStatus.READY
    assert huge_long.kv_token_limit == block_size * 2
    assert huge_long.num_blocks == 2
    assert moderate_long.status == MessageStatus.WAITING
    assert moderate_long.num_blocks == 0
    assert moderate_long.kv_token_limit is None


def test_schedule_prefill_prefer_long_admits_smaller_long_waiter_first():
    scheduler, block_size = _make_scheduler_for_long_context_chunks(num_gpu_blocks=8)
    now = time.perf_counter()
    huge_long = scheduler.add_session(100).add_sequence([1] * (block_size * 16))
    huge_long.arrive_time = now - 1.0
    moderate_long = scheduler.add_session(101).add_sequence([2] * (block_size * 4))
    moderate_long.arrive_time = now
    short = scheduler.add_session(102).add_sequence([3] * (block_size // 2))

    output = scheduler.schedule(is_prefill=True, prefer_long_prefill=True)

    assert output.running == [moderate_long]
    assert moderate_long.status == MessageStatus.READY
    assert moderate_long.kv_token_limit == block_size * 2
    assert moderate_long.num_blocks == 2
    assert huge_long.status == MessageStatus.WAITING
    assert huge_long.num_blocks == 0
    assert huge_long.kv_token_limit is None
    assert short.status == MessageStatus.WAITING
    assert short.num_blocks == 0


def test_schedule_prefill_prefer_long_ages_huge_long_waiter():
    scheduler, block_size = _make_scheduler_for_long_context_chunks(num_gpu_blocks=8)
    scheduler._prefill_scheduler._long_prefill_aging_seconds_per_chunk = 0.01
    now = time.perf_counter()
    huge_long = scheduler.add_session(100).add_sequence([1] * (block_size * 16))
    huge_long.arrive_time = now - 1.0
    moderate_long = scheduler.add_session(101).add_sequence([2] * (block_size * 4))
    moderate_long.arrive_time = now

    output = scheduler.schedule(is_prefill=True, prefer_long_prefill=True)

    assert output.running == [huge_long]
    assert huge_long.status == MessageStatus.READY
    assert huge_long.kv_token_limit == block_size * 2
    assert huge_long.num_blocks == 2
    assert moderate_long.status == MessageStatus.WAITING
    assert moderate_long.num_blocks == 0
    assert moderate_long.kv_token_limit is None


def test_reserve_long_context_chunk_grows_one_chunk_at_a_time():
    scheduler, block_size = _make_scheduler_for_long_context_chunks(num_gpu_blocks=6)
    long_seq = scheduler.add_session(100).add_sequence([1] * (block_size * 5))

    output = scheduler.schedule(is_prefill=True, prealloc_size=1)
    assert output.running == [long_seq]
    assert long_seq.kv_token_limit == block_size * 2
    assert long_seq.num_blocks == 2

    scheduler.activate_seqs([long_seq])
    long_seq.set_step(block_size * 2)

    assert scheduler.reserve_long_context_chunk(long_seq, block_size * 2)
    assert long_seq.status == MessageStatus.RUNNING
    assert long_seq.kv_token_limit == block_size * 4
    assert long_seq.num_blocks == 4

    long_seq.set_step(block_size * 4)

    assert scheduler.reserve_long_context_chunk(long_seq, block_size, prealloc_size=1, is_last_chunk=True)
    assert long_seq.kv_token_limit is None
    assert long_seq.num_blocks == 6
    assert scheduler.block_manager.get_num_free_gpu_blocks() == 0


def test_reserve_long_context_chunk_failure_preserves_committed_prefix():
    scheduler, block_size = _make_scheduler_for_long_context_chunks(num_gpu_blocks=2)
    long_seq = scheduler.add_session(100).add_sequence([1] * (block_size * 4))

    output = scheduler.schedule(is_prefill=True)
    assert output.running == [long_seq]
    scheduler.activate_seqs([long_seq])
    long_seq.set_step(block_size * 2)

    assert not scheduler.reserve_long_context_chunk(long_seq, block_size * 2)
    assert long_seq.status == MessageStatus.RUNNING
    assert long_seq.kv_token_limit == block_size * 2
    assert long_seq.num_blocks == 2


def test_reserve_last_long_context_chunk_failure_restores_chunk_limit():
    scheduler, block_size = _make_scheduler_for_long_context_chunks(num_gpu_blocks=3)
    long_seq = scheduler.add_session(100).add_sequence([1] * (block_size * 4))

    output = scheduler.schedule(is_prefill=True)
    assert output.running == [long_seq]
    scheduler.activate_seqs([long_seq])
    long_seq.set_step(block_size * 2)

    assert not scheduler.reserve_long_context_chunk(long_seq,
                                                    block_size * 2,
                                                    prealloc_size=1,
                                                    is_last_chunk=True)
    assert long_seq.status == MessageStatus.RUNNING
    assert long_seq.kv_token_limit == block_size * 2
    assert long_seq.num_blocks == 2
    assert scheduler.block_manager.get_num_free_gpu_blocks() == 1


def test_scheduler_accepts_prefix_hit_that_starts_middle_long_context_chunk():
    from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy
    block_size = 16
    seq_meta = SequenceMeta(block_size, strategy=ARSequenceStrategy())
    cache_config = CacheConfig(max_batches=1,
                               block_size=block_size,
                               num_cpu_blocks=0,
                               num_gpu_blocks=8,
                               max_prefill_token_num=block_size * 2,
                               enable_prefix_caching=True)
    scheduler_config = SchedulerConfig(max_batches=1,
                                       max_session_len=128,
                                       max_request_output_len=64,
                                       eviction_type='recompute')
    scheduler = Scheduler(scheduler_config=scheduler_config, cache_config=cache_config, seq_meta=seq_meta)

    cached = scheduler.add_session(0).add_sequence([1] * block_size + [2] * block_size)
    scheduler.block_manager.allocate(cached)
    scheduler.block_trie.allocate(cached)
    cached.state.stop()

    token_ids = [1] * block_size + [2] * block_size + [3] * block_size
    token_ids += [4] * block_size + [5] * block_size
    seq = scheduler.add_session(1).add_sequence(token_ids)

    output = scheduler.schedule(is_prefill=True)

    assert output.running == [seq]
    assert seq.num_history_ids == block_size * 2
    assert seq.num_token_ids == len(token_ids) - block_size * 2
    assert seq.cached_tokens == block_size * 2
    assert scheduler.block_trie.stats.num_query_tokens == len(token_ids)
    assert scheduler.block_trie.stats.num_hit_tokens == block_size * 2
