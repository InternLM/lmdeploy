import pytest
import torch

from lmdeploy.pytorch.messages import SchedulerSession
from lmdeploy.pytorch.paging.block_manager import (BlockManager,
                                                   LogicalAllocator)


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
        assert gpu_allocator.get_num_free_blocks(
        ) == num_gpu_blocks - block_size

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


class TestBlockManager:

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
    def block_mgr(self, num_cpu_blocks, num_gpu_blocks):
        yield BlockManager(num_cpu_blocks, num_gpu_blocks)

    def test_alloc(self, block_mgr, block_size, num_gpu_blocks):
        sess = SchedulerSession(0, block_size)

        # test alloc
        token_ids = torch.tensor([1])
        msg = sess.add_sequence(token_ids)
        # msg.append_tokens(1, block_size)
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
        token_ids = torch.zeros((num_gpu_blocks * block_size + 1, ),
                                dtype=torch.int64)
        msg = sess.add_sequence(token_ids)
        assert not block_mgr.can_allocate(msg)

    def test_append_slot(self, block_mgr, block_size, num_gpu_blocks):
        sess = SchedulerSession(0, block_size)

        # test append
        token_ids = torch.tensor([1])
        msg = sess.add_sequence(token_ids)
        block_mgr.allocate(msg)
        block_table = block_mgr.get_block_table(msg)
        assert len(block_table) == 1

        # no new logical block
        msg.update_token_ids(torch.tensor([1] * (block_size - 1)))
        assert block_mgr.can_append_slot(msg)
        block_mgr.append_slot(msg)
        block_table = block_mgr.get_block_table(msg)
        assert len(block_table) == 1
        assert block_mgr.get_num_free_gpu_blocks() == num_gpu_blocks - 1

        # with new logical block
        msg.update_token_ids(torch.tensor([1]))
        block_mgr.append_slot(msg)
        block_table = block_mgr.get_block_table(msg)
        assert len(block_table) == 2
        assert block_mgr.get_num_free_gpu_blocks() == num_gpu_blocks - 2

    def test_fork(self, block_mgr, block_size, num_gpu_blocks):
        sess = SchedulerSession(0, block_size)

        token_ids = torch.tensor([1] * (block_size * 2 + 1))
        from_msg = sess.add_sequence(token_ids)
        block_mgr.allocate(from_msg)
        from_block_table = block_mgr.get_block_table(from_msg)
        assert len(from_block_table) == 3

        to_msg = sess.fork_sequence(torch.tensor([1]), from_msg)

        # fork
        assert block_mgr.can_fork(from_msg)
        copy_map = block_mgr.fork(from_msg, to_msg)
        block_table = block_mgr.get_block_table(to_msg)
        assert len(block_table) == 3
        assert block_mgr.get_num_free_gpu_blocks() == num_gpu_blocks - 4
        assert block_table[0] == from_block_table[0]
        assert block_table[1] == from_block_table[1]
        assert block_table[2] != from_block_table[2]
        assert len(copy_map) == 1
        assert copy_map[from_block_table[2]] == block_table[2]

        # can not fork
        assert not block_mgr.can_fork(from_msg)

    def test_swap(self, block_mgr, block_size, num_gpu_blocks):
        sess = SchedulerSession(0, block_size)

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
            assert block_id in swap_map.values()

        old_phy_blocks = block_mgr.get_block_table(msg)
        success, swap_map = block_mgr.try_swap_in(msg)
        new_phy_blocks = block_mgr.get_block_table(msg)
        assert block_mgr.get_num_free_gpu_blocks() == num_gpu_blocks - 2
        assert block_mgr.get_num_free_cpu_blocks() == num_gpu_blocks
        assert len(swap_map) == 2
        for block_id in old_phy_blocks:
            assert block_id in swap_map
        for block_id in new_phy_blocks:
            assert block_id in swap_map.values()

        success, swap_map = block_mgr.try_swap_out(msg)
        assert success
        token_ids = torch.tensor([1] * (block_size * 4))
        msg_full = sess.add_sequence(token_ids)
        block_mgr.allocate(msg_full)
        success, swap_map = block_mgr.try_swap_out(msg)
        assert not success
