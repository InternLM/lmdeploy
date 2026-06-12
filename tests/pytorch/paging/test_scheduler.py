import pytest
import torch

from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
from lmdeploy.pytorch.disagg.conn.protocol import MigrationProtocol, MigrationRequest
from lmdeploy.pytorch.engine.inputs_maker import _compact_state_prefix_cache_save_offsets
from lmdeploy.pytorch.messages import MessageStatus, SequenceMeta
from lmdeploy.pytorch.paging.scheduler import Scheduler
from lmdeploy.pytorch.paging.state_manager import StateManager


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

        assert scheduler.has_unfinished()

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


def _make_ssm_scheduler(max_batch_size: int = 1, prefix_cache_state_budget: int = 0):
    from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy
    block_size = 16
    cache_config = CacheConfig(max_batches=max_batch_size,
                               block_size=block_size,
                               num_cpu_blocks=4,
                               num_gpu_blocks=16,
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


def _add_ready_ssm_checkpoint(scheduler: Scheduler, token_ids: list[int]):
    session = scheduler.add_session(len(scheduler.sessions))
    seq = session.add_sequence(token_ids)
    scheduler.block_manager.allocate(seq)
    scheduler.block_trie.allocate(seq)
    state_idx = scheduler.block_trie.reserve_state_checkpoint_for_seq(seq)
    assert state_idx >= 0
    assert scheduler.block_trie.commit_state_checkpoint_for_seq(seq)
    node = seq.prefix_cache.last_shared_node
    session.remove_sequence(seq)
    return node, state_idx


def test_ssm_runtime_state_reclaims_borrowed_checkpoint_slot():
    scheduler = _make_ssm_scheduler(max_batch_size=1, prefix_cache_state_budget=0)
    block_size = scheduler.seq_meta.block_size
    node, state_idx = _add_ready_ssm_checkpoint(scheduler, [1] * block_size * 2)
    seq = scheduler.add_session(100).add_sequence([2] * block_size * 2)

    output = scheduler.schedule(is_prefill=True)

    assert output.running == [seq]
    assert seq.logical_state == state_idx
    assert node.state_idx == -1
    assert not node.state_ready
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
    assert scheduler.block_trie.reserve_state_checkpoint_for_seq(seq, step=block_size * 2) == -1


def test_ssm_runtime_state_waits_when_only_checkpoint_slot_is_pinned():
    scheduler = _make_ssm_scheduler(max_batch_size=1, prefix_cache_state_budget=0)
    block_size = scheduler.seq_meta.block_size
    node, state_idx = _add_ready_ssm_checkpoint(scheduler, [1] * block_size * 2)
    node.state_ref_count = 1
    seq = scheduler.add_session(100).add_sequence([2] * block_size * 2)

    output = scheduler.schedule(is_prefill=True)

    assert output.running == []
    assert seq.status == MessageStatus.WAITING
    assert seq.logical_state == -1
    assert node.state_idx == state_idx
    assert node.state_ready


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
    assert seq_a.prefix_cache.last_shared_node is seq_b.prefix_cache.last_shared_node

    save_state_offsets = [
        scheduler.block_trie.reserve_state_checkpoint_for_seq(seq) for seq in output.running
    ]
    save_src_offsets, save_dst_offsets = _compact_state_prefix_cache_save_offsets(output.running,
                                                                                  save_state_offsets)

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

    state_idx = scheduler.block_trie.reserve_state_checkpoint_for_seq(seq)
    node = seq.prefix_cache.save_node
    assert state_idx >= 0
    assert node is not None
    assert scheduler.state_manager.get_num_allocated_checkpoint_states() == 1

    scheduler.end_session(100)

    assert 100 not in scheduler.sessions
    assert node.state_idx == -1
    assert not node.state_ready
    assert scheduler.state_manager.get_num_runtime_states() == 0
    assert scheduler.state_manager.get_num_allocated_checkpoint_states() == 0


def test_ssm_end_session_releases_acquired_restore_checkpoint():
    scheduler = _make_ssm_scheduler(max_batch_size=1, prefix_cache_state_budget=1)
    block_size = scheduler.seq_meta.block_size
    node, state_idx = _add_ready_ssm_checkpoint(scheduler, [1] * block_size * 2)
    seq = scheduler.add_session(100).add_sequence([1] * block_size * 2 + [2])

    scheduler.block_trie.match(seq)
    assert seq.prefix_cache.restore_state == state_idx
    assert scheduler.block_trie.acquire_state_checkpoint_restore_for_seq(seq)
    assert node.state_ref_count == 1

    scheduler.end_session(100)

    assert 100 not in scheduler.sessions
    assert node.state_idx == state_idx
    assert node.state_ready
    assert node.state_ref_count == 0


def test_ssm_failed_restore_schedule_rolls_back_match():
    scheduler = _make_ssm_scheduler(max_batch_size=1, prefix_cache_state_budget=0)
    block_size = scheduler.seq_meta.block_size
    node, state_idx = _add_ready_ssm_checkpoint(scheduler, [1] * block_size * 2)
    node.state_ref_count = 1
    seq = scheduler.add_session(100).add_sequence([1] * block_size * 2 + [2])

    output = scheduler.schedule(is_prefill=True)

    assert output.running == []
    assert seq.status == MessageStatus.WAITING
    assert seq.num_history_ids == 0
    assert len(seq.logical_blocks) == 0
    assert seq.prefix_cache.last_shared_node is None
    assert seq.prefix_cache.restore_state == -1
    assert seq.prefix_cache.restore_node is None
    assert node.state_idx == state_idx
    assert node.state_ready
    assert scheduler.block_trie.stats.num_query_tokens == 0
    assert scheduler.block_trie.stats.num_hit_tokens == 0

    node.state_ref_count = 0
    output = scheduler.schedule(is_prefill=True)

    assert output.running == [seq]
    assert seq.status == MessageStatus.READY
    assert seq.num_history_ids == 0
    assert seq.prefix_cache.restore_state == -1
    assert seq.logical_state == state_idx
    assert node.state_idx == -1
    assert not node.state_ready
    assert scheduler.block_trie.stats.num_query_tokens == len(seq.all_ids)
    assert scheduler.block_trie.stats.num_hit_tokens == 0


def test_ssm_scheduler_preserves_matched_checkpoint_when_evicting_for_runtime_state():
    scheduler = _make_ssm_scheduler(max_batch_size=1, prefix_cache_state_budget=1)
    block_size = scheduler.seq_meta.block_size
    node_a, state_idx_a = _add_ready_ssm_checkpoint(scheduler, [1] * block_size * 2)
    node_b, state_idx_b = _add_ready_ssm_checkpoint(scheduler, [2] * block_size * 2)
    seq = scheduler.add_session(100).add_sequence([1] * block_size * 2 + [3])

    output = scheduler.schedule(is_prefill=True)

    assert output.running == [seq]
    assert seq.num_history_ids == block_size * 2
    assert seq.prefix_cache.restore_state == state_idx_a
    assert seq.prefix_cache.restore_node is node_a
    assert seq.prefix_cache.restore_state_acquired
    assert seq.logical_state == state_idx_b
    assert node_a.state_idx == state_idx_a
    assert node_a.state_ready
    assert node_a.state_ref_count == 1
    assert node_b.state_idx == -1
    assert not node_b.state_ready
    assert scheduler.block_trie.stats.num_hit_tokens == block_size * 2

    assert scheduler.block_trie.release_state_checkpoint_restore_for_seq(seq)


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


def test_scheduler_rolls_back_prefix_hit_that_would_start_long_context_chunk_from_middle():
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
    assert seq.num_history_ids == 0
    assert seq.num_token_ids == len(token_ids)
    assert scheduler.block_trie.stats.num_query_tokens == len(token_ids)
    assert scheduler.block_trie.stats.num_hit_tokens == 0
