import pytest
import torch

from lmdeploy.pytorch.messages import SchedulerSession
from lmdeploy.pytorch.paging.block_manager import BlockAllocator, BlockManager


class TestAllocator:

    @pytest.fixture
    def block_size(self):
        yield 16

    @pytest.fixture
    def block_num(self):
        yield 4

    @pytest.fixture
    def device(self):
        yield 'cpu'

    @pytest.fixture
    def allocator(self, block_size, block_num, device):
        yield BlockAllocator(block_size, block_num, device)

    def test_alloc(self, allocator, block_num):
        assert allocator.get_num_free_blocks() == block_num
        # test allocate
        block = allocator.allocate()
        assert allocator.get_num_free_blocks() == block_num - 1
        # test free
        block.ref_count += 1
        allocator.free(block)
        assert allocator.get_num_free_blocks() == block_num - 1
        allocator.free(block)
        assert allocator.get_num_free_blocks() == block_num
        # no free blocks
        blocks = [allocator.allocate() for _ in range(block_num)]
        with pytest.raises(MemoryError):
            allocator.allocate()
        for block in blocks:
            allocator.free(block)
        # double free
        with pytest.raises(ValueError):
            allocator.free(blocks[0])


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
    def block_mgr(self, block_size, num_cpu_blocks, num_gpu_blocks):
        yield BlockManager(block_size, num_cpu_blocks, num_gpu_blocks)

    def test_alloc(self, block_mgr, block_size, num_gpu_blocks):
        sess = SchedulerSession(0)

        # test alloc
        token_ids = torch.tensor([1])
        msg = sess.add_sequence(token_ids)
        msg.append_tokens(1, block_size)
        assert block_mgr.can_allocate(msg)
        block_mgr.allocate(msg)
        block_table = block_mgr.get_block_table(msg)
        assert block_mgr.get_num_free_gpu_blocks() == num_gpu_blocks - 1
        assert block_table is not None
        assert len(block_table) == 1

        # test free
        block_mgr.free(msg)
        block_table = block_mgr.get_block_table(msg)
        assert block_table is None
        assert block_mgr.get_num_free_gpu_blocks() == num_gpu_blocks

        # alloc over limit
        msg = sess.add_sequence(token_ids)
        msg.append_tokens(num_gpu_blocks * block_size + 1, block_size)
        assert not block_mgr.can_allocate(msg)

    def test_append_slot(self, block_mgr, block_size, num_gpu_blocks):
        sess = SchedulerSession(0)

        # test append
        token_ids = torch.tensor([1])
        msg = sess.add_sequence(token_ids)
        msg.append_tokens(1, block_size)
        block_mgr.allocate(msg)
        block_table = block_mgr.get_block_table(msg)

        # no new logical block
        msg.append_tokens(block_size - 1, block_size)
        assert block_mgr.can_append_slot(msg)
        block_mgr.append_slot(msg)
        assert len(block_table) == 1
        assert block_mgr.get_num_free_gpu_blocks() == num_gpu_blocks - 1

        # with new logical block
        msg.append_tokens(1, block_size)
        block_mgr.append_slot(msg)
        assert len(block_table) == 2
        assert block_mgr.get_num_free_gpu_blocks() == num_gpu_blocks - 2

    def test_fork(self, block_mgr, block_size, num_gpu_blocks):
        sess = SchedulerSession(0)

        token_ids = torch.tensor([1])
        from_msg = sess.add_sequence(token_ids)
        from_msg.append_tokens(block_size + 1, block_size)
        block_mgr.allocate(from_msg)
        from_block_table = block_mgr.get_block_table(from_msg)

        to_msg = sess.fork_sequence(token_ids, from_msg)
        to_msg.append_tokens(1, block_size)

        # fork
        assert block_mgr.can_fork(from_msg)
        copy_map = block_mgr.fork(from_msg, to_msg)
        block_table = block_mgr.get_block_table(to_msg)
        assert len(block_table) == 2
        assert block_mgr.get_num_free_gpu_blocks() == num_gpu_blocks - 3
        assert block_table[0] == from_block_table[0]
        assert block_table[0].ref_count == 2
        assert block_table[1] != from_block_table[1]
        assert len(copy_map) == 1
        assert copy_map[
            from_block_table[1].block_id] == block_table[1].block_id

        # can not fork
        assert block_mgr.can_fork(from_msg)

    def test_swap(self, block_mgr, block_size, num_gpu_blocks):
        sess = SchedulerSession(0)

        token_ids = torch.tensor([1])
        msg = sess.add_sequence(token_ids)
        msg.append_tokens(block_size + 1, block_size)
        block_mgr.allocate(msg)
        block_table = block_mgr.get_block_table(msg)
        gpu_block_id = [block.block_id for block in block_table]

        assert block_mgr.can_swap_out(msg)
        swap_out_map = block_mgr.swap_out(msg)
        assert block_mgr.get_num_free_gpu_blocks() == num_gpu_blocks
        assert block_mgr.get_num_free_cpu_blocks() == num_gpu_blocks - 2
        assert len(swap_out_map) == 2
        for block_id in gpu_block_id:
            assert block_id in swap_out_map
        for block in block_table:
            assert block.device == 'cpu'

        assert block_mgr.can_swap_in(msg)
        swap_in_map = block_mgr.swap_in(msg)
        assert block_mgr.get_num_free_gpu_blocks() == num_gpu_blocks - 2
        assert block_mgr.get_num_free_cpu_blocks() == num_gpu_blocks
        assert len(swap_in_map) == 2
        for block in block_table:
            assert block.device == 'gpu'

        swap_out_map = block_mgr.swap_out(msg)
        msg_full = sess.add_sequence(token_ids)
        msg_full.append_tokens(block_size * 4, block_size)
        block_mgr.allocate(msg_full)
        assert not block_mgr.can_swap_out(msg_full)
