# yapf: disable
import pytest
import torch

from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
from lmdeploy.pytorch.messages import SequenceMeta
from lmdeploy.pytorch.paging.block_manager.base_block_manager import LogicalAllocator
from lmdeploy.pytorch.paging.scheduler import Scheduler

# yapf: enable


class TestAllocator:

    @pytest.fixture
    def num_gpu_blocks(self):
        yield 16

    @pytest.fixture
    def num_cpu_blocks(self):
        yield 4

    @pytest.fixture
    def allocator(self, num_cpu_blocks, num_gpu_blocks):
        yield LogicalAllocator(num_cpu_blocks, num_gpu_blocks)

    def test_alloc(self, allocator, num_cpu_blocks, num_gpu_blocks):

        # initialize
        num_blocks = num_cpu_blocks + num_gpu_blocks
        gpu_allocator = allocator.get_phy_allocator('gpu')
        cpu_allocator = allocator.get_phy_allocator('cpu')
        assert allocator.get_num_free_blocks() == num_blocks
        assert cpu_allocator.get_num_free_blocks() == num_cpu_blocks
        assert gpu_allocator.get_num_free_blocks() == num_gpu_blocks

        # test allocate
        block_size = 4
        blocks = allocator.allocate(block_size, 'gpu')
        assert len(blocks) == block_size
        assert allocator.get_num_free_blocks() == num_blocks - block_size
        assert gpu_allocator.get_num_free_blocks() == num_gpu_blocks - block_size

        # test free
        allocator.add_ref_count(blocks, 1)
        allocator.free(blocks)
        assert allocator.get_num_free_blocks() == num_blocks - block_size
        allocator.free(blocks)
        assert allocator.get_num_free_blocks() == num_blocks
        assert gpu_allocator.get_num_free_blocks() == num_gpu_blocks
        assert cpu_allocator.get_num_free_blocks() == num_cpu_blocks

    def test_full(self, allocator, num_cpu_blocks, num_gpu_blocks):

        num_blocks = num_cpu_blocks + num_gpu_blocks
        gpu_allocator = allocator.get_phy_allocator('gpu')
        cpu_allocator = allocator.get_phy_allocator('cpu')

        # no free blocks
        gpu_block_size = num_gpu_blocks
        gpu_blocks = allocator.allocate(gpu_block_size, 'gpu')
        cpu_block_size = num_cpu_blocks
        cpu_blocks = allocator.allocate(cpu_block_size, 'cpu')
        assert cpu_allocator.get_num_free_blocks() == 0
        assert gpu_allocator.get_num_free_blocks() == 0
        with pytest.raises(MemoryError):
            allocator.allocate(1, 'gpu')
        allocator.free(gpu_blocks)
        allocator.free(cpu_blocks)
        assert allocator.get_num_free_blocks() == num_blocks
        assert gpu_allocator.get_num_free_blocks() == num_gpu_blocks
        assert cpu_allocator.get_num_free_blocks() == num_cpu_blocks


class TestDefaultBlockManager:

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

    @pytest.fixture
    def block_mgr(self, scheduler):
        yield scheduler.block_manager

    def test_alloc(self, scheduler, block_mgr, num_gpu_blocks):
        sess = scheduler.add_session(0)
        block_size = sess.seq_meta.block_size

        # test alloc
        token_ids = torch.tensor([1])
        msg = sess.add_sequence(token_ids)
        assert block_mgr.can_allocate(msg)
        block_mgr.allocate(msg)
        block_table = block_mgr.get_block_table(msg)
        assert block_mgr.get_num_free_gpu_blocks() == num_gpu_blocks - 1
        assert block_table is not None
        assert len(block_table) == 1

        # test free
        block_mgr.free(msg)
        block_table = block_mgr.get_block_table(msg)
        assert block_table is None or len(block_table) == 0
        assert block_mgr.get_num_free_gpu_blocks() == num_gpu_blocks

        # alloc over limit
        token_ids = torch.zeros((num_gpu_blocks * block_size + 1, ), dtype=torch.int64)
        msg = sess.add_sequence(token_ids)
        assert not block_mgr.can_allocate(msg)

    def test_num_required_blocks(self, scheduler, block_mgr):
        from lmdeploy.pytorch.messages import InputEmbeddings
        sess = scheduler.add_session(0)
        block_size = sess.seq_meta.block_size

        token_ids = torch.tensor([1])
        msg = sess.add_sequence(token_ids)
        num_required = block_mgr.num_required_blocks(msg)
        assert num_required == 1

        embedding = InputEmbeddings(None, 0, block_size * 2)
        msg = sess.add_sequence(token_ids, input_embeddings=[embedding])
        num_required = block_mgr.num_required_blocks(msg)
        assert num_required == 1

        token_ids = torch.tensor([1] * block_size * 3)
        embedding = InputEmbeddings(None, 0, block_size * 2)
        msg = sess.add_sequence(token_ids, input_embeddings=[embedding])
        num_required = block_mgr.num_required_blocks(msg)
        assert num_required == 3

    def test_append_slot(self, scheduler, block_mgr, num_gpu_blocks):
        sess = scheduler.add_session(0)
        block_size = sess.seq_meta.block_size

        # test append
        token_ids = torch.tensor([1])
        msg = sess.add_sequence(token_ids)
        block_mgr.allocate(msg)
        block_table = block_mgr.get_block_table(msg)
        assert len(block_table) == 1

        # no new logical block
        msg.update_token_ids(torch.tensor([1] * (block_size - 1)))
        assert block_mgr.can_allocate(msg)
        block_mgr.allocate(msg)
        block_table = block_mgr.get_block_table(msg)
        assert len(block_table) == 1
        assert block_mgr.get_num_free_gpu_blocks() == num_gpu_blocks - 1

        # with new logical block
        msg.update_token_ids(torch.tensor([1]))
        block_mgr.allocate(msg)
        block_table = block_mgr.get_block_table(msg)
        assert len(block_table) == 2
        assert block_mgr.get_num_free_gpu_blocks() == num_gpu_blocks - 2

    def test_swap(self, scheduler, block_mgr, num_gpu_blocks):
        sess = scheduler.add_session(0)
        block_size = sess.seq_meta.block_size

        token_ids = torch.tensor([1] * (block_size + 1))
        msg = sess.add_sequence(token_ids)
        block_mgr.allocate(msg)

        old_phy_blocks = block_mgr.get_block_table(msg)
        success, swap_map = block_mgr.try_swap_out(msg)
        new_phy_blocks = block_mgr.get_block_table(msg)
        assert success
        assert block_mgr.get_num_free_gpu_blocks() == num_gpu_blocks
        assert block_mgr.get_num_free_cpu_blocks() == num_gpu_blocks - 2
        assert len(swap_map) == 2
        for block_id in old_phy_blocks:
            assert block_id in swap_map
        for block_id in new_phy_blocks:
            assert block_id - num_gpu_blocks in swap_map.values()

        old_phy_blocks = block_mgr.get_block_table(msg)
        success, swap_map = block_mgr.try_swap_in(msg)
        new_phy_blocks = block_mgr.get_block_table(msg)
        assert block_mgr.get_num_free_gpu_blocks() == num_gpu_blocks - 2
        assert block_mgr.get_num_free_cpu_blocks() == num_gpu_blocks
        assert len(swap_map) == 2
        for block_id in old_phy_blocks:
            assert block_id - num_gpu_blocks in swap_map
        for block_id in new_phy_blocks:
            assert block_id in swap_map.values()

        success, swap_map = block_mgr.try_swap_out(msg)
        assert success
        token_ids = torch.tensor([1] * (block_size * 4))
        msg_full = sess.add_sequence(token_ids)
        block_mgr.allocate(msg_full)
        success, swap_map = block_mgr.try_swap_out(msg)
        assert not success


class TestWindowBlockManager:

    @pytest.fixture
    def window_size(self):
        yield 32

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
    def cache_config(self, block_size, num_cpu_blocks, num_gpu_blocks, max_batch_size, window_size):
        yield CacheConfig(max_batches=max_batch_size,
                          block_size=block_size,
                          num_cpu_blocks=num_cpu_blocks,
                          num_gpu_blocks=num_gpu_blocks,
                          window_size=window_size)

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

    @pytest.fixture
    def block_mgr(self, scheduler):
        yield scheduler.block_manager

    def test_alloc(self, scheduler, block_mgr, num_gpu_blocks):
        sess = scheduler.add_session(0)
        block_size = sess.seq_meta.block_size

        # test alloc
        token_ids = torch.tensor([1])
        msg = sess.add_sequence(token_ids)
        assert block_mgr.can_allocate(msg)
        block_mgr.allocate(msg)
        block_table = block_mgr.get_block_table(msg)
        assert block_mgr.get_num_free_gpu_blocks() == num_gpu_blocks - 1
        assert block_table is not None
        assert len(block_table) == 1

        # test free
        block_mgr.free(msg)
        block_table = block_mgr.get_block_table(msg)
        assert block_table is None or len(block_table) == 0
        assert block_mgr.get_num_free_gpu_blocks() == num_gpu_blocks

        # alloc over limit
        token_ids = torch.zeros((num_gpu_blocks * block_size + 1, ), dtype=torch.int64)
        msg = sess.add_sequence(token_ids)
        assert not block_mgr.can_allocate(msg)

    def test_win_alloc(self, scheduler, block_mgr, num_gpu_blocks, window_size):
        sess = scheduler.add_session(0)

        # 2 win block
        token_ids = torch.tensor([1] * window_size)
        msg = sess.add_sequence(token_ids)
        block_mgr.allocate(msg)
        msg.update_token_ids(torch.tensor([1]))
        block_mgr.allocate(msg)
        assert block_mgr.get_num_free_gpu_blocks() == num_gpu_blocks - 3
        block_table = block_mgr.get_block_table(msg)
        assert block_table is None or len(block_table) == 3
        block_mgr.free(msg)

        # 3 win block
        token_ids = torch.tensor([1] * (window_size + 2))
        msg = sess.add_sequence(token_ids)
        block_mgr.allocate(msg)
        assert block_mgr.get_num_free_gpu_blocks() == num_gpu_blocks - 3
        msg.update_token_ids(torch.tensor([1]))
        block_mgr.allocate(msg)
        assert block_mgr.get_num_free_gpu_blocks() == num_gpu_blocks - 3
        block_table = block_mgr.get_block_table(msg)
        assert block_table is None or len(block_table) == 3
        block_mgr.free(msg)

        # not full win
        token_ids = torch.tensor([1] * (window_size - 2))
        msg = sess.add_sequence(token_ids)
        block_mgr.allocate(msg)
        assert block_mgr.get_num_free_gpu_blocks() == num_gpu_blocks - 2
        msg.update_token_ids(torch.tensor([1]))
        block_mgr.allocate(msg)
        assert block_mgr.get_num_free_gpu_blocks() == num_gpu_blocks - 2
        block_table = block_mgr.get_block_table(msg)
        assert block_table is None or len(block_table) == 2
        block_mgr.free(msg)
