import time
from unittest.mock import Mock

import pytest
import torch

import lmdeploy.pytorch.paging.scheduler as scheduler_module
from lmdeploy.messages import KVTransferConfig
from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
from lmdeploy.pytorch.disagg.conn.protocol import MigrationProtocol, MigrationRequest
from lmdeploy.pytorch.engine.inputs_maker import _make_state_prefix_cache_save_plan
from lmdeploy.pytorch.kv_connector import (
    KVConnectorMetadata,
    KVConnectorOutput,
    KVConnectorResult,
    KVLoadResult,
    KVSaveBlockLease,
)
from lmdeploy.pytorch.kv_connector.mooncake.store.scheduler import MooncakeStoreScheduler
from lmdeploy.pytorch.messages import MessageStatus, SequenceMeta, UpdateTokenMode
from lmdeploy.pytorch.paging.scheduler import Scheduler
from lmdeploy.pytorch.paging.state_manager import StateManager


class _AsyncLookupConnector:

    def __init__(self, results, failed_ids=()):
        self.results = iter(results)
        self.failed_ids = set(failed_ids)
        self.pending_ids = set()
        self.lookup_calls = []
        self.cancelled = []
        self.finished = []
        self.allocations = []

    def on_new_request(self, request):
        pass

    def is_lookup_pending(self, request_id):
        return request_id in self.pending_ids

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        self.lookup_calls.append((request.seq_id, num_computed_tokens))
        result = next(self.results)
        if result[0] is None:
            self.pending_ids.add(request.seq_id)
        else:
            self.pending_ids.discard(request.seq_id)
        return result

    def cancel_lookup(self, request_id):
        self.pending_ids.discard(request_id)
        self.cancelled.append(request_id)

    def update_state_after_alloc(self, request, block_ids, num_external_tokens):
        self.allocations.append((request.seq_id, tuple(block_ids), num_external_tokens))

    def build_connector_meta(self, scheduler_output):
        return None

    def update_connector_output(self, connector_output):
        return KVConnectorResult(
            load_results=tuple(
                KVLoadResult(
                    request_id=request_id,
                    success=request_id not in self.failed_ids,
                )
                for request_id in (connector_output.finished_receiving or set())
            )
        )

    def request_finished(self, request):
        self.finished.append(request.seq_id)

    def finish_transfers_after_worker_drain(self):
        pass

    def shutdown(self):
        pass


def _make_async_lookup_scheduler(
    connector,
    *,
    enable_prefix_caching=True,
    max_batches=1,
    num_gpu_blocks=16,
    max_prefill_token_num=8192,
):
    from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy
    block_size = 4
    return Scheduler(
        scheduler_config=SchedulerConfig(
            max_batches=max_batches,
            max_session_len=64,
            max_request_output_len=16,
            eviction_type='recompute',
        ),
        cache_config=CacheConfig(
            max_batches=max_batches,
            block_size=block_size,
            num_cpu_blocks=0,
            num_gpu_blocks=num_gpu_blocks,
            max_prefill_token_num=max_prefill_token_num,
            enable_prefix_caching=enable_prefix_caching,
            kv_transfer_config=KVTransferConfig(
                kv_connector='MooncakeStoreConnector',
                kv_role='kv_both',
            ),
        ),
        seq_meta=SequenceMeta(block_size, strategy=ARSequenceStrategy()),
        kv_connector=connector,
    )


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
        assert len(scheduler.ready) == 1
        assert seq1 in scheduler.hanging

        # end seq
        seq1.session.remove_sequence(seq1)
        assert session_id1 in scheduler.sessions
        assert seq1 not in scheduler.ready
        assert seq1 not in scheduler.hanging
        assert block_manager.get_num_free_gpu_blocks() == num_gpu_blocks - 2

        # stop session
        scheduler.stop_session(session_id2)
        assert len(scheduler.ready) == 0
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
        assert len(scheduler.ready) == 1
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
        assert len(scheduler.ready) == 1
        assert len(scheduler.waiting) == 1
        assert len(scheduler.hanging) == 0

        scheduler.schedule(is_prefill=True)
        # seq1: 1 running gpu
        # seq2: 3 running gpu
        # seq3: 3 nan
        assert seq1.status == MessageStatus.READY
        assert seq2.status == MessageStatus.READY
        assert block_manager.get_num_free_gpu_blocks() == 0

        # test running append
        seq1.update_token_ids(torch.tensor([1] * block_size))
        seq2.update_token_ids(torch.tensor([1] * block_size))
        assert len(scheduler.ready) == 2
        scheduler.schedule(is_prefill=False)
        # seq1: 2 running gpu
        # seq2: 4 waiting cpu
        # seq3: 3 nan
        assert seq1.status == MessageStatus.READY
        assert seq2.status == MessageStatus.WAITING
        assert block_manager.get_num_free_gpu_blocks() == 2


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

    output = scheduler._schedule_migration()

    assert output == [seq]
    assert seq.status == MessageStatus.MIGRATION_READY


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


def test_scheduler_prefix_match_rollback_clears_recompute_overlap_window():
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

    seq = scheduler.add_session(1).add_sequence(token_ids)
    stats_snapshot = scheduler.block_trie.stats.snapshot()
    scheduler.block_trie.match(seq)

    assert seq.num_history_ids == block_size * 2
    assert seq.prefix_cache.recompute_overlap.fresh_block_range == range(2, 3)

    scheduler._rollback_unscheduled_prefix_match(seq, stats_snapshot)

    assert seq.num_history_ids == 0
    assert seq.num_token_ids == len(token_ids)
    assert seq.cached_tokens == 0
    assert seq.prefix_cache.recompute_overlap.fresh_block_range is None
    assert scheduler.block_trie.stats.num_query_tokens == 0
    assert scheduler.block_trie.stats.num_hit_tokens == 0


def test_async_lookup_pending_rolls_back_a_new_request_once():
    connector = _AsyncLookupConnector([(None, False), (0, False)])
    scheduler = _make_async_lookup_scheduler(connector)
    seq = scheduler.add_session(70).add_sequence(torch.arange(9))

    first = scheduler.schedule(is_prefill=True)
    assert first.running == []
    assert scheduler.last_schedule_had_pending_lookup
    assert seq.num_history_ids == 0
    assert seq.num_blocks == 0
    assert seq.prefix_cache.trie_cursor is None
    assert seq.prefix_cache.match_start_step == -1
    assert scheduler.block_trie.stats.num_query_tokens == 0

    second = scheduler.schedule(is_prefill=True)
    assert second.running == []
    assert connector.lookup_calls == [(seq.seq_id, 0)]
    assert scheduler.block_trie.stats.num_query_tokens == 0

    connector.pending_ids.clear()
    third = scheduler.schedule(is_prefill=True)
    assert third.running == [seq]
    assert connector.lookup_calls == [(seq.seq_id, 0), (seq.seq_id, 0)]


def test_async_lookup_rebases_remote_hit_after_local_trie_grows(monkeypatch):
    connector = MooncakeStoreScheduler(
        CacheConfig(
            max_batches=1,
            block_size=4,
            num_cpu_blocks=0,
            num_gpu_blocks=16,
            enable_prefix_caching=True,
            kv_transfer_config=KVTransferConfig(
                kv_connector='MooncakeStoreConnector',
                kv_role='kv_both',
            ),
        ))
    assert connector.client is not None
    # The same asynchronous lookup first reports pending, then returns its
    # absolute remote prefix boundary from the original snapshot.
    monkeypatch.setattr(connector.client, 'lookup', Mock(side_effect=(None, 16)))
    observed_results = []
    get_matched_tokens = connector.get_num_new_matched_tokens

    def _record_result(request, num_computed_tokens):
        result = get_matched_tokens(request, num_computed_tokens)
        observed_results.append((num_computed_tokens, result))
        return result

    monkeypatch.setattr(connector, 'get_num_new_matched_tokens', _record_result)
    scheduler = _make_async_lookup_scheduler(connector)
    tokens = torch.arange(17)

    cached_to_8 = scheduler.add_session(80).add_sequence(tokens[:9])
    scheduler.block_manager.allocate(cached_to_8)
    scheduler.block_trie.allocate(cached_to_8)
    cached_to_8.state.stop()

    seq = scheduler.add_session(81).add_sequence(tokens)
    first = scheduler.schedule(is_prefill=True)

    assert first.running == []
    assert seq.num_history_ids == 0
    assert observed_results == [(8, (None, False))]

    # While the remote lookup is pending, another sequence publishes the next
    # complete local block. The retried match must now advance from 8 to 12.
    cached_to_12 = scheduler.add_session(82).add_sequence(tokens[:13])
    scheduler.block_manager.allocate(cached_to_12)
    scheduler.block_trie.allocate(cached_to_12)
    cached_to_12.state.stop()

    second = scheduler.schedule(is_prefill=True)

    assert second.running == []
    assert seq.num_history_ids == 12
    assert seq.status == MessageStatus.WAITING_FOR_REMOTE_KVS
    assert observed_results == [
        (8, (None, False)),
        (12, (4, True)),
    ]

    metadata = connector.build_connector_meta(second)
    assert metadata is not None
    assert len(metadata.load_requests) == 1
    load_request = metadata.load_requests[0]
    block_table = scheduler.block_manager.get_block_table(seq)
    assert load_request.block_ids == (int(block_table[3]), )
    assert load_request.remote_block_count == 4
    scheduler.shutdown()


def test_async_lookup_pending_request_does_not_block_later_waiter():
    connector = _AsyncLookupConnector([(None, False), (0, False)])
    scheduler = _make_async_lookup_scheduler(
        connector,
        enable_prefix_caching=False,
    )
    pending = scheduler.add_session(74).add_sequence(torch.arange(9))
    schedulable = scheduler.add_session(75).add_sequence(torch.arange(9, 18))

    output = scheduler.schedule(is_prefill=True)

    assert output.running == [schedulable]
    assert pending.status == MessageStatus.WAITING
    assert schedulable.status == MessageStatus.READY
    assert pending.seq_id in connector.pending_ids
    assert connector.lookup_calls == [
        (pending.seq_id, 0),
        (schedulable.seq_id, 0),
    ]
    assert scheduler.last_schedule_had_pending_lookup


def test_async_lookup_precisely_restores_a_multiturn_local_prefix():
    connector = _AsyncLookupConnector([(None, False), (4, True)])
    scheduler = _make_async_lookup_scheduler(connector)
    tokens = torch.arange(13)
    seq = scheduler.add_session(71).add_sequence(tokens)

    seq.kv_token_limit = 4
    scheduler.block_manager.allocate(seq)
    scheduler.block_trie.allocate(seq)
    seq.set_step(4)
    seq.kv_token_limit = 5
    seq.cached_tokens = 3
    seq.model_meta = {'state': 'keep'}
    seq.prefix_cache.recompute_overlap.fresh_block_range = range(0, 1)
    seq.prefix_cache.recompute_overlap.trie_block_map[0] = seq.logical_blocks[0]
    baseline_blocks = seq.logical_blocks.get_real_blocks().copy()
    baseline_cursor = seq.prefix_cache.trie_cursor

    cached = scheduler.add_session(72).add_sequence(tokens[:9])
    scheduler.block_manager.allocate(cached)
    scheduler.block_trie.allocate(cached)
    cached.state.stop()
    matched_block = cached.logical_blocks[1]
    matched_ref_count = scheduler.block_manager.allocator.get_ref_count(
        cached.logical_blocks.get_real_blocks()[1:2]).copy()
    scheduler.block_trie.stats.reset()

    first = scheduler.schedule(is_prefill=True)
    assert first.running == []
    assert scheduler.last_schedule_had_pending_lookup
    assert seq.num_history_ids == 4
    assert seq.num_blocks == 1
    assert torch.equal(torch.from_numpy(seq.logical_blocks.get_real_blocks()),
                       torch.from_numpy(baseline_blocks))
    assert seq.prefix_cache.trie_cursor is baseline_cursor
    assert seq.prefix_cache.match_start_step == -1
    assert seq.prefix_cache.recompute_overlap.fresh_block_range == range(0, 1)
    assert seq.prefix_cache.recompute_overlap.trie_block_map == {0: baseline_blocks[0]}
    assert seq.cached_tokens == 3
    assert seq.kv_token_limit == 5
    assert seq.model_meta == {'state': 'keep'}
    assert scheduler.block_manager.allocator.get_ref_count(
        cached.logical_blocks.get_real_blocks()[1:2]).tolist() == matched_ref_count.tolist()
    assert scheduler.block_trie.stats.num_query_tokens == 0

    second = scheduler.schedule(is_prefill=True)
    assert second.running == []
    assert connector.lookup_calls == [(seq.seq_id, 8)]
    assert seq.num_history_ids == 4
    assert scheduler.block_trie.stats.num_query_tokens == 0

    connector.pending_ids.clear()
    seq.kv_token_limit = None
    seq.prefix_cache.recompute_overlap.clear_tracking()
    third = scheduler.schedule(is_prefill=True)
    assert third.running == []
    assert connector.lookup_calls == [(seq.seq_id, 8), (seq.seq_id, 8)]
    assert seq.num_history_ids == 8
    assert matched_block in seq.logical_blocks.get_real_blocks()
    assert seq.status == MessageStatus.WAITING_FOR_REMOTE_KVS
    assert connector.allocations[0][2] == 4

    scheduler.update_connector_output(
        KVConnectorOutput(finished_receiving={seq.seq_id}))
    assert seq.num_history_ids == 12
    assert seq.status == MessageStatus.WAITING

    fourth = scheduler.schedule(is_prefill=True)
    assert fourth.running == [seq]


def test_external_cached_tokens_survive_remote_ready_admission():
    connector = _AsyncLookupConnector([(8, True)])
    scheduler = _make_async_lookup_scheduler(connector)
    seq = scheduler.add_session(73).add_sequence(torch.arange(13))

    started = scheduler.schedule(is_prefill=True)
    assert started.running == []

    scheduler.update_connector_output(
        KVConnectorOutput(finished_receiving={seq.seq_id}))
    assert seq.num_history_ids == 8
    assert seq.cached_tokens == 8
    assert seq.prefix_cache.match_start_step == 0

    admitted = scheduler.schedule(is_prefill=True)

    assert admitted.running == [seq]
    assert seq.cached_tokens == 8
    assert seq.prefix_cache.match_start_step == 0


def test_remote_ready_long_prefill_respects_short_only_turn():
    connector = _AsyncLookupConnector([(8, True)])
    scheduler = _make_async_lookup_scheduler(
        connector,
        max_prefill_token_num=4,
    )
    seq = scheduler.add_session(74).add_sequence(torch.arange(17))

    started = scheduler.schedule(is_prefill=True)
    assert started.running == []
    scheduler.update_connector_output(
        KVConnectorOutput(finished_receiving={seq.seq_id}))
    assert seq.num_history_ids == 8
    assert scheduler.kv_load_coordinator.is_remote_ready(seq)

    short_turn = scheduler.schedule(is_prefill=True, allow_long_prefill=False)

    assert short_turn.running == []
    assert seq.status == MessageStatus.WAITING
    assert seq.num_history_ids == 8
    assert scheduler.kv_load_coordinator.is_remote_ready(seq)

    long_turn = scheduler.schedule(is_prefill=True)
    assert long_turn.running == [seq]
    assert seq.status == MessageStatus.READY


def test_external_cached_tokens_survive_prefill_budget_rejection():
    connector = _AsyncLookupConnector([(8, True), (8, True)])
    scheduler = _make_async_lookup_scheduler(
        connector,
        max_batches=2,
        max_prefill_token_num=8,
    )
    admitted_seq = scheduler.add_session(74).add_sequence(torch.arange(13))
    waiting_seq = scheduler.add_session(75).add_sequence(torch.arange(20, 33))

    started = scheduler.schedule(is_prefill=True)
    assert started.running == []
    scheduler.update_connector_output(
        KVConnectorOutput(
            finished_receiving={admitted_seq.seq_id, waiting_seq.seq_id}))

    admitted = scheduler.schedule(is_prefill=True)

    assert admitted.running == [admitted_seq]
    assert waiting_seq.status == MessageStatus.WAITING
    assert waiting_seq.num_history_ids == 8
    assert waiting_seq.num_blocks == 2
    assert waiting_seq.cached_tokens == 8
    assert waiting_seq.prefix_cache.match_start_step == 0
    assert scheduler.kv_load_coordinator.is_remote_ready(waiting_seq)


def test_async_load_keeps_a_private_partial_block_at_the_suffix_start():
    connector = _AsyncLookupConnector([(7, True)])
    scheduler = _make_async_lookup_scheduler(connector)
    tokens = torch.arange(13)
    cached = scheduler.add_session(76).add_sequence(tokens)
    scheduler.block_manager.allocate(cached)
    scheduler.block_trie.allocate(cached)
    cached.state.stop()

    seq = scheduler.add_session(77).add_sequence(tokens)
    seq.kv_token_limit = 5
    scheduler.block_manager.allocate(seq)
    scheduler.block_trie.allocate(seq)
    seq.set_step(5)
    seq.kv_token_limit = None
    private_block = int(scheduler.block_manager.get_block_table(seq)[1])

    output = scheduler.schedule(is_prefill=True)

    assert output.running == []
    assert seq.status == MessageStatus.WAITING_FOR_REMOTE_KVS
    assert seq.num_blocks == 3
    assert connector.lookup_calls == [(seq.seq_id, 5)]
    load_blocks = connector.allocations[0][1]
    assert load_blocks == tuple(
        int(block_id)
        for block_id in scheduler.block_manager.get_block_table(seq)[1:3]
    )
    assert load_blocks[0] == private_block


def test_async_load_requires_capacity_for_the_complete_prefill():
    connector = _AsyncLookupConnector([(8, True)])
    scheduler = _make_async_lookup_scheduler(
        connector,
        enable_prefix_caching=False,
        num_gpu_blocks=3,
    )
    seq = scheduler.add_session(76).add_sequence(torch.arange(13))

    output = scheduler.schedule(is_prefill=True)

    assert output.running == []
    assert seq.status == MessageStatus.WAITING
    assert seq.num_blocks == 0
    assert connector.allocations == []


def test_async_load_does_not_consume_model_batch_slot():
    connector = _AsyncLookupConnector([(8, True), (0, False)])
    scheduler = _make_async_lookup_scheduler(
        connector,
        enable_prefix_caching=False,
        num_gpu_blocks=5,
    )
    loading = scheduler.add_session(77).add_sequence(torch.arange(13))
    later = scheduler.add_session(78).add_sequence(torch.arange(8))

    output = scheduler.schedule(is_prefill=True)

    assert output.running == [later]
    assert loading.status == MessageStatus.WAITING_FOR_REMOTE_KVS
    assert loading.num_blocks == 2
    assert later.status == MessageStatus.READY
    assert scheduler.kv_load_coordinator.soft_reserved_blocks() == 2
    assert scheduler.block_manager.get_num_free_gpu_blocks() == 1

    scheduler.update_connector_output(
        KVConnectorOutput(finished_receiving={loading.seq_id}))
    assert loading.num_history_ids == 8
    assert loading.cached_tokens == 8


def test_multiple_async_loads_start_in_one_prefill_turn():
    connector = _AsyncLookupConnector([(8, True), (8, True)])
    scheduler = _make_async_lookup_scheduler(
        connector,
        enable_prefix_caching=False,
        max_batches=1,
        num_gpu_blocks=16,
    )
    first = scheduler.add_session(91).add_sequence(torch.arange(13))
    second = scheduler.add_session(92).add_sequence(torch.arange(20, 33))

    output = scheduler.schedule(is_prefill=True)

    assert output.running == []
    assert first.status == MessageStatus.WAITING_FOR_REMOTE_KVS
    assert second.status == MessageStatus.WAITING_FOR_REMOTE_KVS
    assert [allocation[0] for allocation in connector.allocations] == [
        first.seq_id,
        second.seq_id,
    ]


def test_soft_reservation_blocks_new_load_until_capacity_is_released():
    connector = _AsyncLookupConnector([
        (4, True),
        (12, True),
        (12, True),
    ])
    scheduler = _make_async_lookup_scheduler(
        connector,
        enable_prefix_caching=False,
        num_gpu_blocks=6,
    )
    first = scheduler.add_session(86).add_sequence(torch.arange(13))

    scheduler.schedule(is_prefill=True)
    assert first.status == MessageStatus.WAITING_FOR_REMOTE_KVS
    assert first.num_blocks == 1
    assert scheduler.kv_load_coordinator.soft_reserved_blocks() == 3

    second = scheduler.add_session(87).add_sequence(torch.arange(17))
    blocked = scheduler.schedule(is_prefill=True)
    assert blocked.running == []
    assert second.status == MessageStatus.WAITING
    assert second.num_blocks == 0
    assert [allocation[0] for allocation in connector.allocations] == [first.seq_id]

    scheduler.end_session(86)
    scheduler.update_connector_output(
        KVConnectorOutput(finished_receiving={first.seq_id}))
    assert scheduler.kv_load_coordinator.soft_reserved_blocks() == 0

    retried = scheduler.schedule(is_prefill=True)
    assert retried.running == []
    assert second.status == MessageStatus.WAITING_FOR_REMOTE_KVS
    assert second.num_blocks == 3
    assert [allocation[0] for allocation in connector.allocations] == [
        first.seq_id,
        second.seq_id,
    ]


def test_failed_async_load_preserves_local_prefix_and_releases_remote_tail():
    connector = _AsyncLookupConnector([(4, True)])
    scheduler = _make_async_lookup_scheduler(connector)
    tokens = torch.arange(13)
    cached = scheduler.add_session(79).add_sequence(tokens[:9])
    scheduler.block_manager.allocate(cached)
    scheduler.block_trie.allocate(cached)
    cached.state.stop()
    seq = scheduler.add_session(80).add_sequence(tokens)
    scheduler.schedule(is_prefill=True)
    connector.failed_ids.add(seq.seq_id)

    scheduler.update_connector_output(
        KVConnectorOutput(finished_receiving={seq.seq_id}))

    assert seq.status == MessageStatus.WAITING
    assert seq.num_history_ids == 8
    assert seq.num_blocks == 2
    assert seq.cached_tokens == 8
    assert scheduler.kv_load_coordinator.soft_reserved_blocks() == 0


def test_async_load_soft_reservation_shrinks_across_chunks():
    connector = _AsyncLookupConnector([(4, True)])
    scheduler = _make_async_lookup_scheduler(
        connector,
        enable_prefix_caching=False,
        max_prefill_token_num=4,
    )
    seq = scheduler.add_session(81).add_sequence(torch.arange(13))

    scheduler.schedule(is_prefill=True)
    assert scheduler.kv_load_coordinator.soft_reserved_blocks() == 3
    scheduler.update_connector_output(
        KVConnectorOutput(finished_receiving={seq.seq_id}))

    assert scheduler.schedule(is_prefill=True).running == [seq]
    assert scheduler.kv_load_coordinator.soft_reserved_blocks() == 2
    seq.set_step(8)
    scheduler.release_completed_prefill_reservations([seq])
    assert scheduler.kv_load_coordinator.soft_reserved_blocks() == 2

    assert scheduler.reserve_long_context_chunk(seq, chunk_size=4)
    assert scheduler.kv_load_coordinator.soft_reserved_blocks() == 1
    seq.set_step(12)
    assert scheduler.reserve_long_context_chunk(seq, chunk_size=1, is_last_chunk=True)
    assert scheduler.kv_load_coordinator.soft_reserved_blocks() == 0


def test_end_session_waits_for_active_async_load_before_freeing_blocks():
    connector = _AsyncLookupConnector([(8, True)])
    scheduler = _make_async_lookup_scheduler(
        connector,
        enable_prefix_caching=False,
    )
    seq = scheduler.add_session(82).add_sequence(torch.arange(13))
    scheduler.schedule(is_prefill=True)

    scheduler.end_session(82)
    assert 82 in scheduler.sessions
    assert seq.status == MessageStatus.WAITING_FOR_REMOTE_KVS
    assert seq.num_blocks == 2
    assert connector.finished == []

    scheduler.update_connector_output(
        KVConnectorOutput(finished_receiving={seq.seq_id}))
    assert 82 not in scheduler.sessions
    assert connector.finished == [seq.seq_id]


def test_worker_drain_finishes_an_ended_session_with_a_dropped_load_output():
    connector = _AsyncLookupConnector([(8, True)])
    scheduler = _make_async_lookup_scheduler(
        connector,
        enable_prefix_caching=False,
    )
    seq = scheduler.add_session(85).add_sequence(torch.arange(13))
    scheduler.schedule(is_prefill=True)
    scheduler.end_session(85)

    scheduler.finish_deferred_kv_transfers_after_worker_drain()

    assert 85 not in scheduler.sessions
    assert connector.finished == [seq.seq_id]
    assert scheduler.kv_load_coordinator.soft_reserved_blocks() == 0


def test_completed_async_load_is_admitted_before_an_older_waiter():
    connector = _AsyncLookupConnector([(8, True)])
    scheduler = _make_async_lookup_scheduler(
        connector,
        enable_prefix_caching=False,
        num_gpu_blocks=4,
    )
    loaded = scheduler.add_session(83).add_sequence(torch.arange(13))
    scheduler.schedule(is_prefill=True)
    scheduler.update_connector_output(
        KVConnectorOutput(finished_receiving={loaded.seq_id}))
    newcomer = scheduler.add_session(84).add_sequence(torch.arange(4))
    newcomer.arrive_time = loaded.arrive_time - 1

    output = scheduler.schedule(is_prefill=True)

    assert output.running == [loaded]
    assert newcomer.status == MessageStatus.WAITING


def test_stop_and_end_session_cancel_lookup_before_request_cleanup():
    connector = _AsyncLookupConnector([(None, False)])
    scheduler = _make_async_lookup_scheduler(
        connector,
        enable_prefix_caching=False,
    )
    seq = scheduler.add_session(73).add_sequence(torch.arange(9))
    scheduler.schedule(is_prefill=True)

    scheduler.stop_session(73)
    assert connector.cancelled == [seq.seq_id]

    scheduler.end_session(73)
    assert connector.finished == [seq.seq_id]


def test_async_save_lease_keeps_exact_blocks_alive_until_all_tp_complete():

    class _SaveMetadata(KVConnectorMetadata):

        def __init__(self, logical_block_ids):
            self.logical_block_ids = tuple(logical_block_ids)

        def get_save_block_leases(self):
            return (KVSaveBlockLease(7, self.logical_block_ids), )

    class _SaveConnector(_AsyncLookupConnector):

        def __init__(self):
            super().__init__([])
            self.metadata = None

        def build_connector_meta(self, scheduler_output):
            metadata, self.metadata = self.metadata, None
            return metadata

        def update_connector_output(self, connector_output):
            return KVConnectorResult(
                completed_save_ids=frozenset(
                    connector_output.completed_save_ids or ()),
            )

    connector = _SaveConnector()
    scheduler = _make_async_lookup_scheduler(
        connector,
        enable_prefix_caching=False,
        num_gpu_blocks=4,
    )
    seq = scheduler.add_session(88).add_sequence(torch.arange(8))
    scheduler.block_manager.allocate(seq)
    logical_blocks = seq.logical_blocks.get_real_blocks().copy()
    allocator = scheduler.block_manager.allocator
    connector.metadata = _SaveMetadata(logical_blocks)

    metadata = scheduler.build_connector_meta(
        [seq],
        connector_token_lens=(8, ),
    )

    assert metadata is not None
    assert allocator.get_ref_count(logical_blocks).tolist() == [2, 2]
    assert scheduler.has_unfinished()

    # Sequence ownership may disappear before remote I/O completes. The save
    # lease remains as the only reference and prevents physical reuse.
    scheduler.block_manager.free(seq)
    assert allocator.get_ref_count(logical_blocks).tolist() == [1, 1]
    assert scheduler.block_manager.get_num_free_gpu_blocks() == 2

    scheduler.update_connector_output(
        KVConnectorOutput(completed_save_ids={7}))
    assert allocator.get_ref_count(logical_blocks).tolist() == [0, 0]
    assert scheduler.block_manager.get_num_free_gpu_blocks() == 4
    assert not scheduler.kv_save_coordinator.has_pending()


def test_worker_drain_releases_save_leases_when_outputs_are_discarded():

    class _SaveMetadata(KVConnectorMetadata):

        def __init__(self, logical_block_ids):
            self.logical_block_ids = tuple(logical_block_ids)

        def get_save_block_leases(self):
            return (KVSaveBlockLease(9, self.logical_block_ids), )

    connector = _AsyncLookupConnector([])
    scheduler = _make_async_lookup_scheduler(
        connector,
        enable_prefix_caching=False,
        num_gpu_blocks=2,
    )
    seq = scheduler.add_session(89).add_sequence(torch.arange(8))
    scheduler.block_manager.allocate(seq)
    logical_blocks = seq.logical_blocks.get_real_blocks().copy()
    metadata = _SaveMetadata(logical_blocks)
    connector.build_connector_meta = lambda scheduler_output: metadata

    scheduler.build_connector_meta([seq], connector_token_lens=(8, ))
    scheduler.block_manager.free(seq)
    scheduler.finish_deferred_kv_transfers_after_worker_drain()

    assert scheduler.block_manager.get_num_free_gpu_blocks() == 2
    assert not scheduler.kv_save_coordinator.has_pending()


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

    monkeypatch.setattr(scheduler, '_ensure_runtime_state_available', _ensure_runtime_state_available_once_then_succeed)
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
    monkeypatch.setattr(scheduler_module._envs, 'opt_ttft_policy', 'fifo')
    monkeypatch.setattr(scheduler_module._envs, 'opt_ttft_aging_sec', 0.25)

    scheduler, _ = _make_scheduler_for_long_context_chunks(num_gpu_blocks=8)

    assert scheduler._long_prefill_policy == 'fifo'
    assert scheduler._long_prefill_aging_seconds_per_chunk == 0.25


def test_schedule_prefill_prefer_long_fifo_policy_keeps_oldest_huge_waiter_first():
    scheduler, block_size = _make_scheduler_for_long_context_chunks(num_gpu_blocks=8)
    scheduler._long_prefill_policy = 'fifo'
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
    scheduler._long_prefill_aging_seconds_per_chunk = 0.01
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


def test_schedule_prefill_reapplies_chunk_limit_after_ssm_state_rollback():
    scheduler, block_size = _make_ssm_scheduler_for_long_context_chunks(num_gpu_blocks=2)
    long_seq = scheduler.add_session(100).add_sequence([1] * (block_size * 4))

    ensure_results = iter([False, True])

    def _ensure_runtime_state_available_once_then_succeed():
        return next(ensure_results)

    scheduler._ensure_runtime_state_available = _ensure_runtime_state_available_once_then_succeed

    output = scheduler.schedule(is_prefill=True, prealloc_size=1)

    assert output.running == [long_seq]
    assert long_seq.status == MessageStatus.READY
    assert long_seq.kv_token_limit == block_size * 2
    assert long_seq.num_blocks == 2


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
