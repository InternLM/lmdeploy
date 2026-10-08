# Copyright (c) OpenMMLab. All rights reserved.
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import lmdeploy.pytorch.paging.prefill_scheduler as prefill_scheduler_module
from lmdeploy.messages import KVTransferConfig
from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
from lmdeploy.pytorch.long_context import plan_long_context_chunk
from lmdeploy.pytorch.messages import InputEmbeddings, MessageStatus, SequenceMeta, UpdateTokenMode
from lmdeploy.pytorch.multimodal.data_type import MultiModalData
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


@pytest.mark.parametrize('no_drop', [False, True])
def test_scheduler_ar_spec_prefix_hit_obeys_block_drop_policy(no_drop):
    from lmdeploy.pytorch.config import SpecDecodeConfig
    from lmdeploy.pytorch.strategies.ar_spec import ARSpecStrategyFactory
    block_size = 16
    strategy = ARSpecStrategyFactory(SimpleNamespace(bos_token_id=0), SpecDecodeConfig(
        model='draft', method='qwen3_5_mtp', disable_prefix_cache_block_drop=no_drop)).build_sequence_strategy()
    seq_meta = SequenceMeta(block_size, strategy=strategy)
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
    expected_hit = block_size * (3 if no_drop else 2)
    assert seq.prefix_cache.recompute_overlap.recompute_blocks == (0 if no_drop else 1)
    assert seq.num_history_ids == expected_hit
    assert seq.cached_tokens == expected_hit
    assert (seq.logical_blocks[2] == cached_blocks[2]) == no_drop
    assert seq.prefix_cache.recompute_overlap.fresh_block_range is None
    assert scheduler.block_trie.stats.num_query_tokens == len(token_ids)
    assert scheduler.block_trie.stats.num_hit_tokens == expected_hit


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
    monkeypatch.setattr(scheduler.eviction_helper, 'try_make_capacity_for', Mock(return_value=False))
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


def test_scheduler_long_first_short_turn_admits_only_final_prefix_hit():
    scheduler, block_size = _make_prefix_cache_scheduler(
        max_batches=1,
        max_prefill_token_num=16,
    )

    cached = scheduler.add_session(0).add_sequence([1] * block_size)
    scheduler.schedule(is_prefill=True)
    cached.state.stop()

    short = scheduler.add_session(1).add_sequence([4])
    cache_hit_tail = scheduler.add_session(2).add_sequence(
        [1] * block_size + [3])

    output = scheduler.schedule(
        is_prefill=True,
        allow_long_prefill=False,
        prefer_long_prefill=True,
    )

    assert output.running == [cache_hit_tail]
    assert short.status == MessageStatus.WAITING
    assert cache_hit_tail.num_history_ids == block_size
    assert cache_hit_tail.num_token_ids == 1


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
    try_make_capacity = Mock(return_value=False)
    monkeypatch.setattr(scheduler.eviction_helper, 'try_make_capacity_for', try_make_capacity)

    output = scheduler.schedule(is_prefill=True)

    assert output.running == []
    assert seq.status == MessageStatus.WAITING
    assert seq.num_history_ids == 0
    assert seq.num_blocks == 0
    assert seq.kv_token_limit is None
    assert seq.cached_tokens == 0
    assert seq.prefix_cache.trie_cursor is None
    assert seq.prefix_cache.match_start_step == -1
    assert try_make_capacity.call_count == 1
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
    seq.state.release_paging_resources()

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

    assert scheduler.eviction_helper.try_make_capacity_for(pressure, [seq], 0)
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


@pytest.mark.parametrize('method', [
    None, 'qwen3_5_mtp', 'deepseek_mtp', 'hy3_mtp', 'eagle', 'eagle3', 'dflash', 'dspark'])
def test_multimodal_chunk_reservation_obeys_the_draft_input_shift(method):
    from lmdeploy.pytorch.config import SpecDecodeConfig
    from lmdeploy.pytorch.engine.inputs_maker import LongContextChunker
    from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy
    from lmdeploy.pytorch.strategies.ar_spec import ARSpecStrategyFactory

    strategy = ARSequenceStrategy() if method is None else ARSpecStrategyFactory(
        SimpleNamespace(bos_token_id=0), SpecDecodeConfig(model='draft', method=method)).build_sequence_strategy()
    cache = CacheConfig(max_batches=1, block_size=4, num_cpu_blocks=0, num_gpu_blocks=8,
                        max_prefill_token_num=3)
    scheduler = Scheduler(SchedulerConfig(max_batches=1, max_session_len=32), cache,
                          seq_meta=SequenceMeta(4, strategy=strategy))
    seq = scheduler.add_session(0).add_sequence(range(9), multimodals={
        'image': [MultiModalData(torch.zeros(2, 2), 2, 4), MultiModalData(torch.ones(2, 2), 4, 6)]})
    shifted = method not in (None, 'dflash', 'dspark')
    assert seq.prefill_input_shift == int(shifted)
    assert scheduler.schedule(is_prefill=True).running == [seq]
    chunker = LongContextChunker(3)
    chunker.set_seq(seq)
    assert chunker.max_prefill_num == (5 if shifted else 3)
    assert chunker.next_chunk_size()[0] == seq.kv_token_limit == (1 if shifted else 2)
    assert seq.num_blocks == 1
    scheduler.shutdown()


def _make_mooncake_prefill_scheduler(budget: int = 32, hybrid: bool = True):
    from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy

    cache = CacheConfig(max_batches=2, block_size=4, num_cpu_blocks=0, num_gpu_blocks=16,
                        max_prefill_token_num=budget, states_shapes=[((2,), torch.float32)] if hybrid else [],
                        num_state_caches=6,
                        mooncake_prefill_save_alignment=8, mooncake_state_save_slots=2,
                        kv_transfer_config=KVTransferConfig(kv_connector='MooncakeStoreConnector',
                                                           kv_role='kv_producer'))
    return Scheduler(SchedulerConfig(max_batches=2, max_session_len=64), cache,
                     seq_meta=SequenceMeta(4, strategy=ARSequenceStrategy()))


def test_mooncake_aligned_prefill_runs_alone_and_reserves_only_its_chunk():
    scheduler = _make_mooncake_prefill_scheduler()
    cache = scheduler.cache_config
    short = scheduler.add_session(0).add_sequence([1] * 4)
    aligned = scheduler.add_session(1).add_sequence([2] * 20)

    # Both fit the token budget, but a non-final chunk needs its own forward.
    assert scheduler.schedule(is_prefill=True).running == [short]
    assert aligned.status == MessageStatus.WAITING
    assert aligned.num_blocks == 0
    scheduler.end_session(0)
    assert scheduler.schedule(is_prefill=True).running == [aligned]
    plan = plan_long_context_chunk(aligned, cache.max_prefill_token_num, save_alignment=8)
    assert plan.chunk_end == aligned.kv_token_limit == 16
    assert aligned.num_blocks == 4

    scheduler.activate_seqs([aligned])
    aligned.set_step(16)
    tail = plan_long_context_chunk(aligned, cache.max_prefill_token_num, save_alignment=8)
    assert tail.is_last_chunk
    assert tail.chunk_size == 4
    assert scheduler.reserve_long_context_chunk(aligned, tail.chunk_size, is_last_chunk=tail.is_last_chunk)
    assert aligned.kv_token_limit is None
    assert aligned.num_blocks == 5
    scheduler.shutdown()


@pytest.mark.parametrize(('history', 'budget', 'span', 'ends'), [
    (0, 32, None, (7,)),
    (0, 32, None, (8,)),
    (0, 32, None, (16, 20)),
    (16, 32, None, (20,)),
    (7, 32, None, (8, 9)),
    (0, 20, None, (16, 32, 48, 50)),
    (0, 6, None, (6, 8, 14, 16, 22, 24)),
    (7, 4, None, (8, 12, 16, 20)),
    (0, 8, (0, 12), (12, 16, 20)),
    (0, 10, (0, 23), (23, 40)),
])
def test_mooncake_chunk_estimate_matches_prefill_boundaries(history, budget, span, ends):
    scheduler = _make_mooncake_prefill_scheduler(budget)
    multimodals = {'image': [MultiModalData(torch.tensor([1]), *span)]} if span else None
    seq = scheduler.add_session(0).add_sequence([1] * ends[-1], multimodals=multimodals)
    seq.set_step(history)
    prefill = scheduler._prefill_scheduler
    chunk_limit = prefill._long_context_chunk_limit(seq)

    info = prefill_scheduler_module._PrefillReorderer(prefill)._get_reorder_info(seq)

    assert info.estimated_long_chunks == len(ends)
    assert info.prefill_token_count == ends[0] - history
    assert info.is_nonfinal_long_prefill == (len(ends) > 1)
    assert seq.num_history_ids == history
    assert seq.num_blocks == 0
    assert seq.kv_token_limit is None
    for end in ends:
        plan = plan_long_context_chunk(seq, chunk_limit, save_alignment=8)
        assert plan.chunk_end == end
        assert plan.is_last_chunk == (end == ends[-1])
        seq.set_step(end)
    scheduler.shutdown()


@pytest.mark.parametrize(('hybrid', 'embeddings'), [(False, False), (True, True)])
def test_mooncake_chunk_estimate_respects_alignment_gates(hybrid, embeddings):
    scheduler = _make_mooncake_prefill_scheduler(budget=20, hybrid=hybrid)
    input_embeddings = [InputEmbeddings(torch.zeros(1, 1).numpy(), 0, 1)] if embeddings else None
    seq = scheduler.add_session(0).add_sequence([1] * 50, input_embeddings=input_embeddings)

    info = prefill_scheduler_module._PrefillReorderer(scheduler._prefill_scheduler)._get_reorder_info(seq)

    assert info.estimated_long_chunks == 3  # Ordinary ends: 20, 40, 50.
    assert info.prefill_token_count == 20
    scheduler.shutdown()


@pytest.mark.parametrize(('policy', 'wait_seconds', 'expected_length'), [
    ('size', 0, 40),
    ('fifo', 0, 41),
    ('size', 1, 41),
])
def test_mooncake_chunk_estimate_preserves_size_fifo_and_aging(monkeypatch, policy, wait_seconds, expected_length):
    monkeypatch.setattr(prefill_scheduler_module.time, 'perf_counter', lambda: 100.0)
    scheduler = _make_mooncake_prefill_scheduler()
    prefill = scheduler._prefill_scheduler
    prefill._long_prefill_policy = policy
    prefill._long_prefill_aging_seconds_per_chunk = 0.01
    longer = scheduler.add_session(0).add_sequence([1] * 41)
    longer.arrive_time -= wait_seconds
    shorter = scheduler.add_session(1).add_sequence([2] * 40)

    output = scheduler.schedule(is_prefill=True, prefer_long_prefill=True)

    # Old estimates tie at two chunks; actual ends are (32, 40, 41) and (32, 40).
    expected = shorter if expected_length == 40 else longer
    assert output.running == [expected]
    assert expected.kv_token_limit == 32
    scheduler.shutdown()


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
