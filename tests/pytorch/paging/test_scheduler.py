import pytest

from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
from lmdeploy.pytorch.messages import MessageStatus
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
        yield 16

    @pytest.fixture
    def cache_config(self, block_size, num_cpu_blocks, num_gpu_blocks):
        yield CacheConfig(block_size=block_size,
                          num_cpu_blocks=num_cpu_blocks,
                          num_gpu_blocks=num_gpu_blocks,
                          max_prefill_token_num=4096,
                          shared_cache=True)

    @pytest.fixture
    def scheduler_config(self):
        yield SchedulerConfig(max_batches=4,
                              max_session_len=256,
                              eviction_type='recompute',
                              prefill_interval=4)

    @pytest.fixture
    def scheduler(self, cache_config, scheduler_config):
        yield Scheduler(scheduler_config=scheduler_config,
                        cache_config=cache_config)

    def test_attribute(self, scheduler):
        session_id = 0
        session = scheduler.add_session(session_id)
        assert session_id in scheduler.sessions
        seq0 = session.add_sequence([1, 2, 3])
        seq1 = session.add_sequence([2, 2, 3])
        seq2 = session.add_sequence([3, 2, 3])

        seq0.status = MessageStatus.RUNNING
        seq1.status = MessageStatus.WAITING
        seq2.status = MessageStatus.STOPPED

        assert len(scheduler.running) == 1
        assert scheduler.running[0] == seq0
        assert len(scheduler.waiting) == 1
        assert scheduler.waiting[0] == seq1
        assert len(scheduler.hanging) == 1
        assert scheduler.hanging[0] == seq2

    def test_schedule_base(self, scheduler, block_size, num_gpu_blocks):
        block_manager = scheduler.block_manager
        session_id = 0
        session = scheduler.add_session(session_id)

        num_blocks = 4
        token_ids = [0] * block_size * num_blocks
        seq = session.add_sequence(token_ids)
        scheduler.add_sequence(seq)
        assert seq.status == MessageStatus.WAITING

        output = scheduler.schedule(is_prefill=True)
        block_tables = scheduler.get_block_tables(output.running)

        assert seq.status == MessageStatus.RUNNING
        assert seq in output.running
        assert len(block_tables) == 1
        assert len(block_tables[0]) == num_blocks
        assert block_manager.get_num_free_gpu_blocks(
        ) == num_gpu_blocks - num_blocks

        assert scheduler.has_unfinished()

    def test_prefill(self, scheduler, block_size, num_gpu_blocks):
        block_manager = scheduler.block_manager
        rtree_manager = scheduler.rtree_manager
        session_id1 = 0
        session1 = scheduler.add_session(session_id1)
        session_id2 = 1
        session2 = scheduler.add_session(session_id2)

        token_ids1 = [0] * block_size * 4
        seq1 = session1.add_sequence(token_ids1)
        scheduler.add_sequence(seq1)

        token_ids2 = [1] * block_size * 4
        seq2 = session2.add_sequence(token_ids2)
        scheduler.add_sequence(seq2)

        token_ids3 = [2] * block_size * 9
        seq3 = session2.add_sequence(token_ids3)
        scheduler.add_sequence(seq3)

        rtree_manager.step_time = 10
        scheduler.schedule(is_prefill=True)
        assert seq1.status == MessageStatus.RUNNING
        assert seq2.status == MessageStatus.RUNNING
        assert seq3.status == MessageStatus.WAITING
        assert block_manager.get_num_free_gpu_blocks() == num_gpu_blocks - 8
        scheduler._remove_sequence(seq3)

        # test match
        token_ids4 = [0] * block_size * 2 + [1] * block_size * 2
        seq4 = session2.add_sequence(token_ids4)
        scheduler.add_sequence(seq4)

        rtree_manager.step_time = 20
        scheduler.schedule(is_prefill=True)
        assert seq1.status == MessageStatus.RUNNING
        assert seq2.status == MessageStatus.RUNNING
        assert seq4.status == MessageStatus.RUNNING
        assert block_manager.get_num_free_gpu_blocks() == num_gpu_blocks - 10

        token_ids5 = [5] * block_size * 8
        seq5 = session2.add_sequence(token_ids5)
        scheduler.add_sequence(seq5)
        rtree_manager.step_time = 30
        scheduler.schedule(is_prefill=True)
        assert seq5.status == MessageStatus.RUNNING
        assert seq1.status == MessageStatus.WAITING
        assert seq2.status == MessageStatus.RUNNING
        assert seq4.status == MessageStatus.RUNNING
        assert block_manager.get_num_free_gpu_blocks() == 0

        scheduler._remove_sequence(seq1)
        token_ids6 = [6] * block_size * 7
        seq6 = session2.add_sequence(token_ids6)
        scheduler.add_sequence(seq6)
        rtree_manager.step_time = 40
        scheduler.schedule(is_prefill=True)
        assert seq6.status == MessageStatus.RUNNING
        assert seq5.status == MessageStatus.RUNNING
        assert seq2.status == MessageStatus.WAITING
        assert seq4.status == MessageStatus.WAITING
        assert block_manager.get_num_free_gpu_blocks() == 1

    def test_decoding(self, scheduler, block_size, num_gpu_blocks):
        block_manager = scheduler.block_manager
        rtree_manager = scheduler.rtree_manager
        session_id1 = 0
        session1 = scheduler.add_session(session_id1)

        token_ids1 = [0] * block_size * 4
        seq1 = session1.add_sequence(token_ids1)
        scheduler.add_sequence(seq1)

        token_ids2 = [1] * block_size * 4
        seq2 = session1.add_sequence(token_ids2)
        scheduler.add_sequence(seq2)

        token_ids3 = [0] * block_size * 2 + [1] * 1
        seq3 = session1.add_sequence(token_ids3)
        scheduler.add_sequence(seq3)

        rtree_manager.step_time = 10
        scheduler.schedule(is_prefill=True)
        assert seq1.status == MessageStatus.RUNNING
        assert seq2.status == MessageStatus.RUNNING
        assert seq3.status == MessageStatus.RUNNING
        assert block_manager.get_num_free_gpu_blocks() == num_gpu_blocks - 9

        scheduler.schedule(is_prefill=False, prealloc_size=8)
        assert seq1.status == MessageStatus.RUNNING
        assert seq2.status == MessageStatus.RUNNING
        assert seq3.status == MessageStatus.RUNNING
        assert block_manager.get_num_free_gpu_blocks() == num_gpu_blocks - 11
