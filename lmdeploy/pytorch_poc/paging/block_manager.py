# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm
from typing import Dict, List

from lmdeploy.pytorch_poc.block import PhysicalTokenBlock
from lmdeploy.pytorch_poc.messages import SchedulerSequence


class BlockAllocator:
    """The block allocator.

    The allocator won't allocate real memory. It is used to support
    block manager.

    Args:
        block_size (int): The num tokens of each block.
        block_num (int): Total blocks.
        device (str): The device name.
    """

    def __init__(self, block_size: int, block_num: int, device: str):
        self.block_size = block_size
        self.block_num = block_num
        self.device = device

        free_blocks: List[PhysicalTokenBlock] = [
            PhysicalTokenBlock(device, i, block_size) for i in range(block_num)
        ]
        self.free_blocks = free_blocks

    def allocate(self):
        """Allocate block from block pool."""
        if len(self.free_blocks) > 0:
            block = self.free_blocks.pop(0)
            block.ref_count += 1
            return block
        else:
            raise MemoryError(f'No free {self.device} memory blocks.')

    def free(self, block: PhysicalTokenBlock):
        """Free block to block pool."""
        if block.ref_count == 0:
            raise ValueError(f'Double free {block}.')
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block)

    def get_num_free_blocks(self):
        """Get numbers of free blocks."""
        return len(self.free_blocks)


BlockTable = List[PhysicalTokenBlock]


class BlockManager:
    """Manage the usage of blocks, generate block tables.

    Args:
        block_size (int): The num tokens of each block.
        num_gpu_blocks (int): number of gpu blocks.
        num_cpu_blocks (int): number of cpu blocks.
    """

    def __init__(self, block_size: int, num_gpu_blocks: int,
                 num_cpu_blocks: int) -> None:
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        self.num_cpu_blocks = num_cpu_blocks

        self.gpu_allocator = BlockAllocator(block_size, num_gpu_blocks, 'gpu')
        self.cpu_allocator = BlockAllocator(block_size, num_cpu_blocks, 'cpu')

        self.block_tables: Dict[int, BlockTable] = {}

    def get_block_table(self, msg: SchedulerSequence):
        """Get the block table of given msg.

        Args:
            msg (SchedulerSequence): The msg to get block table.
        """
        seq_id = msg.seq_id
        if seq_id in self.block_tables:
            return self.block_tables[seq_id]
        else:
            return None

    def can_allocate(self, msg: SchedulerSequence):
        """Return if physical block can be allocated for given message."""
        required_blocks = len(msg.logical_blocks)
        return required_blocks <= self.gpu_allocator.get_num_free_blocks()

    def allocate(self, msg: SchedulerSequence):
        """Allocate physical blocks for given message according to logical
        blocks."""
        assert msg.seq_id not in self.block_tables
        block_table: BlockTable = []
        logical_blocks = msg.logical_blocks

        for _ in logical_blocks:
            phy_block = self.gpu_allocator.allocate()
            block_table.append(phy_block)

        self.block_tables[msg.seq_id] = block_table

    def _free_block_table(self, block_table: BlockTable):
        """Free physical blocks of given block table."""
        for block in block_table:
            if block.device == 'cpu':
                self.cpu_allocator.free(block)
            elif block.device == 'gpu':
                self.gpu_allocator.free(block)
            else:
                raise ValueError(f'Can not free block {block}.')

    def free(self, msg: SchedulerSequence):
        """Free all physical blocks allocated for the session."""
        seq_id = msg.seq_id
        if seq_id not in self.block_tables:
            return

        block_table = self.block_tables[seq_id]
        self._free_block_table(block_table)
        self.block_tables.pop(seq_id)

    def can_append_slot(self, msg: SchedulerSequence):
        """Return true if the message can append new slot."""
        seq_id = msg.seq_id
        num_blocks = len(msg.logical_blocks)
        assert seq_id in self.block_tables
        block_table = self.block_tables[seq_id]
        gpu_block_table = [
            block for block in block_table if block.device == 'gpu'
        ]
        return num_blocks - len(
            gpu_block_table) <= self.gpu_allocator.get_num_free_blocks()

    def append_slot(self, msg: SchedulerSequence):
        """Append new slot to message."""
        seq_id = msg.seq_id
        logical_blocks = msg.logical_blocks

        assert seq_id in self.block_tables
        block_table = self.block_tables[seq_id]

        while len(logical_blocks) > len(block_table):
            block = self.gpu_allocator.allocate()
            block_table.append(block)

    def can_fork(self, from_msg: SchedulerSequence):
        """Return true if blocks can be folked."""
        seq_id = from_msg.seq_id
        assert seq_id in self.block_tables
        logical_blocks = from_msg.logical_blocks
        if logical_blocks[-1].is_full():
            # every block can be shared
            return True

        block_table = self.block_tables[seq_id]
        device = block_table[-1].device
        if device == 'cpu':
            allocator = self.cpu_allocator
        elif device == 'gpu':
            allocator = self.gpu_allocator
        else:
            raise ValueError(f'Unknown device {device}')
        return allocator.get_num_free_blocks() >= 1

    def fork(self, from_msg: SchedulerSequence, to_msg: SchedulerSequence):
        """Fork new message."""
        from_msg_id = from_msg.seq_id
        from_block_table = self.block_tables[from_msg_id]

        block_table: BlockTable = []
        for block in from_block_table[:-1]:
            block.ref_count += 1
            block_table.append(block)

        # process last block
        from_logical_blocks = from_msg.logical_blocks
        last_block = from_block_table[-1]
        copy_map: Dict[int, int] = dict()
        if from_logical_blocks[-1].is_full():
            last_block.ref_count += 1
            block_table.append(last_block)
        else:
            device = last_block.device
            if device == 'cpu':
                allocator = self.cpu_allocator
            elif device == 'gpu':
                allocator = self.gpu_allocator
            block = allocator.allocate()
            block_table.append(block)
            copy_map[last_block.block_id] = block.block_id

        self.block_tables[to_msg.seq_id] = block_table
        return copy_map

    def _can_swap(self, msg: SchedulerSequence, allocator: BlockAllocator):
        """Check if swap can be performed."""
        block_table = self.get_block_table(msg)
        assert block_table is not None

        num_free_blocks = allocator.get_num_free_blocks()
        return num_free_blocks > len(block_table)

    def can_swap_in(self, msg: SchedulerSequence):
        """Check if the message can be swapped in."""
        return self._can_swap(msg, self.gpu_allocator)

    def swap_in(self, msg: SchedulerSequence):
        """Swap the message into GPU."""
        block_table = self.get_block_table(msg)
        assert block_table is not None

        swap_map: Dict[int, int] = {}
        for i in range(len(block_table)):
            block = block_table[i]
            if block.device == 'cpu':
                new_block = self.gpu_allocator.allocate()
                swap_map[block.block_id] = new_block.block_id
                block_table[i] = new_block
                self.cpu_allocator.free(block)

        return swap_map

    def can_swap_out(self, msg: SchedulerSequence):
        """Check if the message can be swap out."""
        return self._can_swap(msg, self.cpu_allocator)

    def swap_out(self, msg: SchedulerSequence):
        """Swap the message out to host."""
        block_table = self.get_block_table(msg)
        assert block_table is not None

        swap_map: Dict[int, int] = {}
        for i in range(len(block_table)):
            block = block_table[i]
            if block.device == 'gpu':
                new_block = self.cpu_allocator.allocate()
                swap_map[block.block_id] = new_block.block_id
                block_table[i] = new_block
                self.gpu_allocator.free(block)

        return swap_map

    def reset(self) -> None:
        """Reset block table."""
        for block_table in self.block_tables.values():
            self._free_block_table(block_table)
        self.block_tables.clear()

    def get_num_free_gpu_blocks(self) -> int:
        """Get number of free gpu blocks."""
        return self.gpu_allocator.get_num_free_blocks()

    def get_num_free_cpu_blocks(self) -> int:
        """Get number of free cpu blocks."""
        return self.cpu_allocator.get_num_free_blocks()
