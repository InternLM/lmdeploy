import pytest
import torch

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
        yield 4

    @pytest.fixture
    def cache_config(self, block_size, num_cpu_blocks, num_gpu_blocks):
        yield CacheConfig(block_size=block_size,
                          num_cpu_blocks=num_cpu_blocks,
                          num_gpu_blocks=num_gpu_blocks)

    @pytest.fixture
    def scheduler_config(self):
        yield SchedulerConfig(max_batches=4,
                              max_session_len=128,
                              max_request_output_len=64)

    @pytest.fixture
    def scheduler(self, cache_config, scheduler_config):
        yield Scheduler(scheduler_config=scheduler_config,
                        cache_config=cache_config)

    def test_schedule_base(self, scheduler, block_size, num_gpu_blocks):
        block_manager = scheduler.block_manager
        session_id = 0
        session = scheduler.add_session(session_id)
        assert session_id in scheduler.sessions
        assert scheduler.sessions[session_id] == session

        num_blocks = 2
        token_ids = torch.tensor([0] * block_size * num_blocks)
        seq = session.add_sequence(token_ids)
        scheduler.add_sequence(seq)

        assert seq.status == MessageStatus.WAITING
        assert seq in scheduler.waiting

        output = scheduler.schedule()

        assert seq.status == MessageStatus.RUNNING
        assert seq in output.running
        assert len(output.block_tables) == 1
        assert len(output.block_tables[0]) == num_blocks
        assert block_manager.get_num_free_gpu_blocks(
        ) == num_gpu_blocks - num_blocks

        assert scheduler.has_unfinished()

    def test_update(self, scheduler, block_size, num_gpu_blocks):
        block_manager = scheduler.block_manager
        session_id1 = 0
        session1 = scheduler.add_session(session_id1)
        token_ids1 = torch.tensor([0] * block_size * 1)
        seq1 = session1.add_sequence(token_ids1)
        scheduler.add_sequence(seq1)

        session_id2 = 1
        session2 = scheduler.add_session(session_id2)
        token_ids2 = torch.tensor([0] * block_size * 2)
        seq2 = session2.add_sequence(token_ids2)
        scheduler.add_sequence(seq2)
        token_ids3 = torch.tensor([0] * block_size * 3)
        seq3 = session2.add_sequence(token_ids3)
        scheduler.add_sequence(seq3)

        scheduler.schedule()
        assert seq1.status == MessageStatus.RUNNING
        assert seq2.status == MessageStatus.RUNNING
        assert seq3.status == MessageStatus.WAITING

        # stop seq
        seq1.status = MessageStatus.STOPPED
        scheduler.update()
        assert len(scheduler.running) == 1
        assert seq1 in scheduler.hanging

        # end seq
        seq1.status = MessageStatus.ENDED
        scheduler.update()
        assert session_id1 in scheduler.sessions
        assert seq1 not in scheduler.running
        assert seq1 not in scheduler.hanging
        assert block_manager.get_num_free_gpu_blocks() == num_gpu_blocks - 2

        # stop session
        scheduler.stop_session(session_id2)
        scheduler.update()
        assert len(scheduler.running) == 0
        assert len(scheduler.waiting) == 0
        assert len(scheduler.hanging) == 2

        # end session
        scheduler.end_session(session_id2)
        scheduler.update()
        assert seq2.status == MessageStatus.ENDED
        assert seq3.status == MessageStatus.ENDED
        assert session_id2 not in scheduler.sessions
        assert len(scheduler.hanging) == 0
        assert block_manager.get_num_free_gpu_blocks() == num_gpu_blocks

    def test_swap(self, scheduler, block_size, num_gpu_blocks, num_cpu_blocks):
        block_manager = scheduler.block_manager
        session_id = 0
        session = scheduler.add_session(session_id)

        # test: add 3 seq
        token_ids1 = torch.tensor([0] * block_size * 1)
        seq1 = session.add_sequence(token_ids1)
        scheduler.add_sequence(seq1)
        token_ids2 = torch.tensor([0] * block_size * 2)
        seq2 = session.add_sequence(token_ids2)
        scheduler.add_sequence(seq2)
        token_ids3 = torch.tensor([0] * block_size * 3)
        seq3 = session.add_sequence(token_ids3)
        scheduler.add_sequence(seq3)
        scheduler.schedule()
        # seq1: 1 running gpu
        # seq2: 2 running gpu
        # seq3: 3 waiting empty
        assert seq1.status == MessageStatus.RUNNING
        assert seq2.status == MessageStatus.RUNNING
        assert seq3.status == MessageStatus.WAITING
        assert block_manager.get_num_free_gpu_blocks() == num_gpu_blocks - 3

        # test: waiting alloc
        seq2.status = MessageStatus.STOPPED
        scheduler.update()
        assert len(scheduler.running) == 1
        assert len(scheduler.waiting) == 1
        assert len(scheduler.hanging) == 1

        output = scheduler.schedule()
        # seq1: 1 running gpu
        # seq2: 2 hanging cpu
        # seq3: 3 waiting gpu
        assert seq1.status == MessageStatus.RUNNING
        assert seq2.status == MessageStatus.STOPPED
        assert seq3.status == MessageStatus.RUNNING
        assert block_manager.get_num_free_gpu_blocks() == 0
        assert block_manager.get_num_free_cpu_blocks() == num_cpu_blocks - 2
        assert len(output.swap_out_map) == 2

        # test: waiting append token
        seq2.status = MessageStatus.WAITING
        seq3.status = MessageStatus.ENDED
        seq2.update_token_ids(torch.tensor([1] * block_size))
        scheduler.update()
        assert len(scheduler.running) == 1
        assert len(scheduler.waiting) == 1
        assert len(scheduler.hanging) == 0

        output = scheduler.schedule()
        # seq1: 1 running gpu
        # seq2: 3 running gpu
        # seq3: 3 nan
        assert seq1.status == MessageStatus.RUNNING
        assert seq2.status == MessageStatus.RUNNING
        assert block_manager.get_num_free_gpu_blocks() == 0
        assert block_manager.get_num_free_cpu_blocks() == num_cpu_blocks
        assert len(output.swap_in_map) == 2

        # test running append
        seq1.update_token_ids(torch.tensor([1] * block_size))
        seq2.update_token_ids(torch.tensor([1] * block_size))
        scheduler.update()
        assert len(scheduler.running) == 2

        output = scheduler.schedule()
        # seq1: 1 waiting cpu
        # seq2: 4 running gpu
        # seq3: 3 nan
        assert seq1.status == MessageStatus.WAITING
        assert seq2.status == MessageStatus.RUNNING
        assert block_manager.get_num_free_gpu_blocks() == 0
        assert block_manager.get_num_free_cpu_blocks() == num_cpu_blocks - 1
        assert len(output.swap_out_map) == 1
