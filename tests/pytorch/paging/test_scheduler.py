# Copyright (c) OpenMMLab. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import lmdeploy.pytorch.paging.scheduler as scheduler_module
from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
from lmdeploy.pytorch.disagg.conn.protocol import MigrationProtocol, MigrationRequest
from lmdeploy.pytorch.messages import MessageStatus, SequenceMeta
from lmdeploy.pytorch.paging.scheduler import Scheduler


class TestScheduler:

    @pytest.fixture
    def block_size(self):
        yield 16

    @pytest.fixture
    def num_cpu_blocks(self):
        yield 4

    @pytest.fixture
    def num_gpu_blocks(self):
        yield 4

    @pytest.fixture
    def max_batch_size(self):
        yield 4

    @pytest.fixture
    def cache_config(self, block_size, num_cpu_blocks, num_gpu_blocks, max_batch_size):
        yield CacheConfig(max_batches=max_batch_size,
                          block_size=block_size,
                          num_cpu_blocks=num_cpu_blocks,
                          num_gpu_blocks=num_gpu_blocks)

    @pytest.fixture
    def scheduler_config(self, max_batch_size):
        yield SchedulerConfig(max_batches=max_batch_size,
                              max_session_len=128,
                              max_request_output_len=64,
                              eviction_type='recompute')

    @pytest.fixture
    def seq_meta(self, block_size):
        from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy
        strategy = ARSequenceStrategy()
        yield SequenceMeta(block_size, strategy=strategy)

    @pytest.fixture
    def scheduler(self, cache_config, scheduler_config, seq_meta):
        yield Scheduler(scheduler_config=scheduler_config, cache_config=cache_config, seq_meta=seq_meta)

    def test_schedule_base(self, scheduler, block_size, num_gpu_blocks):
        block_manager = scheduler.block_manager
        assert scheduler.schedule_metrics.cache_usage == 0.0

        session_id = 0
        session = scheduler.add_session(session_id)
        assert session_id in scheduler.sessions
        assert scheduler.sessions[session_id] == session
        assert scheduler.get_session(session_id) is session
        assert scheduler.get_sessions() == [session]

        num_blocks = 2
        token_ids = torch.tensor([0] * block_size * num_blocks)
        seq = session.add_sequence(token_ids)

        assert seq.status == MessageStatus.WAITING
        assert seq in scheduler.waiting

        output = scheduler.schedule(is_prefill=True)
        block_tables = scheduler.get_block_tables(output.running)

        assert seq.status == MessageStatus.READY
        assert seq in output.running
        assert len(block_tables) == 1
        assert len(block_tables[0]) == num_blocks
        assert block_manager.get_num_free_gpu_blocks() == num_gpu_blocks - num_blocks
        assert scheduler.schedule_metrics.cache_usage == num_blocks / num_gpu_blocks

        assert scheduler.has_unfinished()

    def test_schedule_metrics_without_gpu_blocks(self, cache_config, scheduler_config, seq_meta):
        cache_config.num_gpu_blocks = 0
        scheduler = Scheduler(scheduler_config=scheduler_config, cache_config=cache_config, seq_meta=seq_meta)

        assert scheduler.schedule_metrics.cache_usage == 0.0

    def test_request_only_cache_usage_disabled_skips_scan(self, cache_config, scheduler_config, seq_meta, monkeypatch):
        monkeypatch.setattr(scheduler_module._envs, 'enable_request_cache_usage_metric', False)
        scheduler = Scheduler(scheduler_config=scheduler_config, cache_config=cache_config, seq_meta=seq_meta)
        get_request_cache_usage = Mock(return_value=0.5)
        monkeypatch.setattr(scheduler, '_get_request_cache_usage', get_request_cache_usage)

        assert scheduler.schedule_metrics.cache_usage == 0.0
        get_request_cache_usage.assert_not_called()

    def test_cache_usage_includes_prefix_only_blocks_when_request_mode_disabled(self, cache_config, scheduler_config,
                                                                                seq_meta, block_size, num_gpu_blocks,
                                                                                monkeypatch):
        monkeypatch.setattr(scheduler_module._envs, 'enable_request_cache_usage_metric', False)
        cache_config.enable_prefix_caching = True
        scheduler = Scheduler(scheduler_config=scheduler_config, cache_config=cache_config, seq_meta=seq_meta)
        seq = scheduler.add_session(0).add_sequence(torch.arange(block_size * 2))

        scheduler.schedule(is_prefill=True)
        scheduler.end_session(seq.session_id)

        assert scheduler.block_manager.get_num_free_gpu_blocks() == num_gpu_blocks - 2
        assert scheduler.schedule_metrics.cache_usage == 2 / num_gpu_blocks

    def test_cache_usage_excludes_prefix_only_blocks_in_request_mode(self, cache_config, scheduler_config, seq_meta,
                                                                     block_size, num_gpu_blocks, monkeypatch):
        monkeypatch.setattr(scheduler_module._envs, 'enable_request_cache_usage_metric', True)
        cache_config.enable_prefix_caching = True
        scheduler = Scheduler(scheduler_config=scheduler_config, cache_config=cache_config, seq_meta=seq_meta)
        token_ids = torch.arange(block_size * 2)

        first = scheduler.add_session(0).add_sequence(token_ids)
        scheduler.schedule(is_prefill=True)
        assert scheduler.schedule_metrics.cache_usage == 2 / num_gpu_blocks

        scheduler.end_session(first.session_id)
        assert scheduler.block_manager.get_num_free_gpu_blocks() == num_gpu_blocks - 2
        assert scheduler.schedule_metrics.cache_usage == 0.0

        second = scheduler.add_session(1).add_sequence(token_ids)
        scheduler.schedule(is_prefill=True)
        assert second.num_blocks == 2
        assert scheduler.schedule_metrics.cache_usage == 2 / num_gpu_blocks

        third = scheduler.add_session(2).add_sequence(token_ids)
        scheduler.schedule(is_prefill=True)
        assert third.num_blocks == 2
        # Both requests share the same two physical blocks.
        # Metric collection groups them from match metadata and does not need
        # to traverse either sequence's trie cursor.
        with monkeypatch.context() as patch:
            patch.setattr(second.prefix_cache, 'trie_cursor', Mock(spec=[]))
            patch.setattr(third.prefix_cache, 'trie_cursor', Mock(spec=[]))
            assert scheduler.schedule_metrics.cache_usage == 2 / num_gpu_blocks

        scheduler.end_session(second.session_id)
        scheduler.end_session(third.session_id)
        recomputed = scheduler.add_session(3).add_sequence(token_ids)
        recomputed.prefix_cache.recompute_overlap.recompute_blocks = 1
        scheduler.schedule(is_prefill=True)
        assert recomputed.prefix_cache.recompute_overlap.trie_block_map
        # A private recompute block replaces one trie block for this request.
        assert scheduler.schedule_metrics.cache_usage == 2 / num_gpu_blocks

    def test_request_only_cache_usage_excludes_stopped_sequences(self, cache_config, scheduler_config, seq_meta,
                                                                 block_size, num_gpu_blocks, monkeypatch):
        monkeypatch.setattr(scheduler_module._envs, 'enable_request_cache_usage_metric', True)
        cache_config.enable_prefix_caching = True
        scheduler = Scheduler(scheduler_config=scheduler_config, cache_config=cache_config, seq_meta=seq_meta)
        seq = scheduler.add_session(0).add_sequence(torch.arange(block_size * 2))

        scheduler.schedule(is_prefill=True)
        scheduler.stop_session(seq.session_id)

        assert seq.status == MessageStatus.STOPPED
        assert scheduler.block_manager.get_num_free_gpu_blocks() == num_gpu_blocks - 2
        assert scheduler.schedule_metrics.cache_usage == 0.0

    def test_update(self, scheduler, block_size, num_gpu_blocks):
        block_manager = scheduler.block_manager
        session_id1 = 0
        session1 = scheduler.add_session(session_id1)
        token_ids1 = torch.tensor([0] * block_size * 1)
        seq1 = session1.add_sequence(token_ids1)

        session_id2 = 1
        session2 = scheduler.add_session(session_id2)
        token_ids2 = torch.tensor([0] * block_size * 2)
        seq2 = session2.add_sequence(token_ids2)
        token_ids3 = torch.tensor([0] * block_size * 3)
        seq3 = session2.add_sequence(token_ids3)

        scheduler.schedule(is_prefill=True)
        assert seq1.status == MessageStatus.READY
        assert seq2.status == MessageStatus.READY
        assert seq3.status == MessageStatus.WAITING

        # stop seq
        seq1.state.stop()
        assert scheduler.num_ready() == 1
        assert seq1 in scheduler.hanging

        # end seq
        seq1.session.remove_sequence(seq1)
        assert session_id1 in scheduler.sessions
        assert seq1 not in scheduler.hanging
        assert block_manager.get_num_free_gpu_blocks() == num_gpu_blocks - 2

        # stop session
        scheduler.stop_session(session_id2)
        assert scheduler.num_ready() == 0
        assert len(scheduler.waiting) == 0
        assert len(scheduler.hanging) == 2

        # end session
        scheduler.end_session(session_id2)
        assert session_id2 not in scheduler.sessions
        assert len(scheduler.hanging) == 0
        assert block_manager.get_num_free_gpu_blocks() == num_gpu_blocks

    def test_evict(self, scheduler, block_size, num_gpu_blocks, num_cpu_blocks):
        block_manager = scheduler.block_manager
        session_id = 0
        session = scheduler.add_session(session_id)

        # test: add 3 seq
        token_ids1 = torch.tensor([0] * block_size * 1)
        seq1 = session.add_sequence(token_ids1)
        token_ids2 = torch.tensor([0] * block_size * 2)
        seq2 = session.add_sequence(token_ids2)
        token_ids3 = torch.tensor([0] * block_size * 3)
        seq3 = session.add_sequence(token_ids3)
        scheduler.schedule(is_prefill=True)
        # seq1: 1 running gpu
        # seq2: 2 running gpu
        # seq3: 3 waiting empty
        assert seq1.status == MessageStatus.READY
        assert seq2.status == MessageStatus.READY
        assert seq3.status == MessageStatus.WAITING
        assert block_manager.get_num_free_gpu_blocks() == num_gpu_blocks - 3

        # test: waiting alloc
        seq2.state.stop()
        assert scheduler.num_ready() == 1
        assert len(scheduler.waiting) == 1
        assert len(scheduler.hanging) == 1

        scheduler.schedule(is_prefill=True)
        # seq1: 1 running gpu
        # seq2: 2 hanging cpu
        # seq3: 3 running gpu
        assert seq1.status == MessageStatus.READY
        assert seq2.status == MessageStatus.STOPPED
        assert seq3.status == MessageStatus.READY
        assert block_manager.get_num_free_gpu_blocks() == 0

        # test: waiting append token
        seq2.state.activate()
        seq3.session.remove_sequence(seq3)
        seq2.update_token_ids(torch.tensor([1] * block_size))
        assert scheduler.num_ready() == 1
        assert len(scheduler.waiting) == 1
        assert len(scheduler.hanging) == 0

        scheduler.schedule(is_prefill=True)
        # seq1: 1 running gpu
        # seq2: 3 running gpu
        # seq3: 3 nan
        assert seq1.status == MessageStatus.READY
        assert seq2.status == MessageStatus.READY
        assert block_manager.get_num_free_gpu_blocks() == 0

    def test_decode_requires_schedule_running(self, scheduler):
        with pytest.raises(ValueError, match='schedule_running'):
            scheduler.schedule(is_prefill=False)

    def test_schedule_running_validity_uses_input_indices(self, scheduler,
                                                          monkeypatch):
        waiting = SimpleNamespace(status=MessageStatus.WAITING)
        running = SimpleNamespace(status=MessageStatus.RUNNING)
        monkeypatch.setattr(scheduler.block_manager, 'num_required_blocks',
                            lambda seq, num_tokens: 0)

        valid_mask = scheduler.schedule_running([waiting, running])

        assert valid_mask == [False, True]

    @pytest.mark.parametrize(
        ('name', 'status'),
        [
            ('waiting', MessageStatus.WAITING),
            ('hanging', MessageStatus.STOPPED),
            ('migration_waiting', MessageStatus.MIGRATION_WAITING),
            ('migration_done', MessageStatus.MIGRATION_DONE),
        ],
    )
    def test_status_sequence_views(self, scheduler, monkeypatch, name, status):
        seq = object()
        queried = []

        def get_sequences(actual_status):
            queried.append(actual_status)
            return {0: seq}

        monkeypatch.setattr(scheduler.seq_manager, 'get_sequences', get_sequences)

        assert getattr(scheduler, name) == [seq]
        assert queried == [status]

    @pytest.mark.parametrize(
        ('name', 'status', 'expected'),
        [
            ('num_waiting', MessageStatus.WAITING, 3),
            ('num_remote_loading', MessageStatus.WAITING_FOR_REMOTE_KVS, 3),
            ('num_ready', MessageStatus.READY, 3),
            ('num_running', MessageStatus.RUNNING, 3),
            ('has_waiting', MessageStatus.WAITING, True),
            ('has_remote_loading', MessageStatus.WAITING_FOR_REMOTE_KVS, True),
            ('has_ready', MessageStatus.READY, True),
            ('has_migration_waiting', MessageStatus.MIGRATION_WAITING, True),
            ('has_migration_done', MessageStatus.MIGRATION_DONE, True),
        ],
    )
    def test_status_queries(self, scheduler, monkeypatch, name, status, expected):
        queried = []

        def num_sequences(actual_status):
            queried.append(actual_status)
            return 3

        monkeypatch.setattr(scheduler.seq_manager, 'num_sequences', num_sequences)

        assert getattr(scheduler, name)() == expected
        assert queried == [status]


def test_schedule_migration_matches_current_sequence():
    from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy
    block_size = 16
    seq_meta = SequenceMeta(block_size, strategy=ARSequenceStrategy())
    cache_config = CacheConfig(max_batches=1,
                               block_size=block_size,
                               num_cpu_blocks=4,
                               num_gpu_blocks=4,
                               enable_prefix_caching=True)
    scheduler_config = SchedulerConfig(max_batches=1,
                                       max_session_len=128,
                                       max_request_output_len=64,
                                       eviction_type='recompute')
    scheduler = Scheduler(scheduler_config=scheduler_config, cache_config=cache_config, seq_meta=seq_meta)
    migration_request = MigrationRequest(protocol=MigrationProtocol.RDMA,
                                         remote_engine_id='prefill-0',
                                         remote_session_id=7,
                                         remote_token_id=8,
                                         remote_block_ids=[1])
    seq = scheduler.add_session(100).add_sequence([1] * block_size, migration_request=migration_request)

    output = scheduler.schedule_migration()

    assert output == [seq]
    assert seq.status == MessageStatus.MIGRATION_READY


def test_schedule_migration_rolls_back_match_when_capacity_fails(monkeypatch):
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

    token_ids = [1] * (block_size * 2)
    cached = scheduler.add_session(0).add_sequence(token_ids)
    scheduler.block_manager.allocate(cached)
    scheduler.block_trie.allocate(cached)
    cached_block = int(cached.logical_blocks.get_real_blocks()[0])
    cached.session.remove_sequence(cached)

    migration_request = MigrationRequest(protocol=MigrationProtocol.RDMA,
                                         remote_engine_id='prefill-0',
                                         remote_session_id=7,
                                         remote_token_id=8,
                                         remote_block_ids=[1, 2])
    seq = scheduler.add_session(100).add_sequence(token_ids, migration_request=migration_request)
    stats_before = scheduler.block_trie.stats.snapshot()
    ref_count_before = int(scheduler.block_trie.allocator.get_ref_count(cached_block))
    try_make_capacity = scheduler.eviction_helper.try_make_capacity_for
    matched_steps = []

    def reject_capacity(actual_seq, *_):
        matched_steps.append(actual_seq.num_history_ids)
        return False

    monkeypatch.setattr(scheduler.eviction_helper, 'try_make_capacity_for', reject_capacity)

    output = scheduler.schedule_migration()

    assert output == []
    assert matched_steps == [block_size]
    assert seq.status == MessageStatus.MIGRATION_WAITING
    assert seq.num_history_ids == 0
    assert seq.num_blocks == 0
    assert seq.prefix_cache.trie_cursor is None
    assert seq.prefix_cache.match_start_step == -1
    assert seq.prefix_cache.recompute_overlap.fresh_block_range is None
    assert scheduler.block_trie.stats == stats_before
    assert int(scheduler.block_trie.allocator.get_ref_count(cached_block)) == ref_count_before

    monkeypatch.setattr(scheduler.eviction_helper, 'try_make_capacity_for', try_make_capacity)
    output = scheduler.schedule_migration()

    assert output == [seq]
    assert seq.status == MessageStatus.MIGRATION_READY
    assert seq.num_history_ids == block_size
    assert seq.num_blocks == 2
    assert seq.cached_tokens == block_size
    assert scheduler.block_trie.stats.num_query_tokens == len(token_ids)
    assert scheduler.block_trie.stats.num_hit_tokens == block_size
    assert int(scheduler.block_trie.allocator.get_ref_count(cached_block)) == ref_count_before + 1


def _make_scheduler_for_decode_growth(num_gpu_blocks: int = 2):
    from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy
    block_size = 4
    seq_meta = SequenceMeta(block_size, strategy=ARSequenceStrategy())
    cache_config = CacheConfig(max_batches=2,
                               block_size=block_size,
                               num_cpu_blocks=0,
                               num_gpu_blocks=num_gpu_blocks,
                               max_prefill_token_num=block_size * 4)
    scheduler_config = SchedulerConfig(max_batches=2,
                                       max_session_len=64,
                                       max_request_output_len=64,
                                       eviction_type='recompute')
    scheduler = Scheduler(scheduler_config=scheduler_config, cache_config=cache_config, seq_meta=seq_meta)
    return scheduler, block_size


def test_schedule_running_reclaims_waiting_blocks_for_decode_growth():
    scheduler, block_size = _make_scheduler_for_decode_growth(num_gpu_blocks=2)
    decode = scheduler.add_session(100).add_sequence([1] * block_size)
    waiting = scheduler.add_session(101).add_sequence([2] * block_size)

    output = scheduler.schedule(is_prefill=True)
    assert output.running == [decode, waiting]
    scheduler.activate_seqs([decode])
    waiting.state.evict()
    assert decode.status == MessageStatus.RUNNING
    assert waiting.status == MessageStatus.WAITING
    assert scheduler.block_manager.get_num_free_gpu_blocks() == 0

    valid_mask = scheduler.schedule_running([decode], num_required_tokens=1, prealloc_size=1)

    assert valid_mask == [True]
    assert decode.status == MessageStatus.RUNNING
    assert decode.num_blocks == 2
    assert waiting.status == MessageStatus.WAITING
    assert waiting.num_blocks == 0
    assert scheduler.block_manager.get_num_free_gpu_blocks() == 0


def test_schedule_running_keeps_other_running_sequence_when_decode_growth_fails():
    scheduler, block_size = _make_scheduler_for_decode_growth(num_gpu_blocks=2)
    decode = scheduler.add_session(100).add_sequence([1] * block_size)
    long_chunk = scheduler.add_session(101).add_sequence([2] * block_size)

    output = scheduler.schedule(is_prefill=True)
    assert output.running == [decode, long_chunk]
    scheduler.activate_seqs([decode, long_chunk])
    assert decode.status == MessageStatus.RUNNING
    assert long_chunk.status == MessageStatus.RUNNING
    assert scheduler.block_manager.get_num_free_gpu_blocks() == 0

    valid_mask = scheduler.schedule_running([decode], num_required_tokens=1, prealloc_size=1)

    assert valid_mask == [False]
    assert decode.status == MessageStatus.WAITING
    assert long_chunk.status == MessageStatus.RUNNING
    assert long_chunk.num_blocks == 1
    assert scheduler.block_manager.get_num_free_gpu_blocks() == 0
