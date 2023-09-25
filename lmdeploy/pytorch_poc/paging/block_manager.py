# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm
from typing import Dict, List

from lmdeploy.pytorch_poc.block import PhysicalTokenBlock
from lmdeploy.pytorch_poc.messages import SchedulerSession


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

    def get_block_table(self, session: SchedulerSession):
        """Get the block table of given session.

        Args:
            session (SchedulerSession): The session to get block table.
        """
        session_id = session.session_id
        if session_id in self.block_tables:
            return self.block_tables[session_id]
        else:
            return None

    def can_allocate(self, session: SchedulerSession):
        """Return if physical block can be allocated for given session."""
        required_blocks = len(session.logical_blocks)
        return required_blocks <= self.gpu_allocator.get_num_free_blocks()

    def allocate(self, session: SchedulerSession):
        """Allocate physical blocks for given session according to logical
        blocks."""
        assert session.session_id not in self.block_tables
        block_table: BlockTable = []
        logical_blocks = session.logical_blocks

        for _ in logical_blocks:
            phy_block = self.gpu_allocator.allocate()
            block_table.append(phy_block)

        self.block_tables[session.session_id] = block_table

    def _free_block_table(self, block_table: BlockTable):
        """Free physical blocks of given block table."""
        for block in block_table:
            if block.device == 'cpu':
                self.cpu_allocator.free(block)
            elif block.device == 'gpu':
                self.gpu_allocator.free(block)
            else:
                raise ValueError(f'Can not free block {block}.')

    def free(self, session: SchedulerSession):
        """Free all physical blocks allocated for the session."""
        session_id = session.session_id
        if session_id not in self.block_tables:
            return

        block_table = self.block_tables[session_id]
        self._free_block_table(block_table)
        self.block_tables.pop(session_id)

    def can_append_slot(self, session: SchedulerSession):
        """Return true if the session can append new slot."""
        session_id = session.session_id
        num_blocks = len(session.logical_blocks)
        assert session_id in self.block_tables
        block_table = self.block_tables[session_id]
        return num_blocks - len(
            block_table) <= self.gpu_allocator.get_num_free_blocks()

    def append_slot(self, session: SchedulerSession):
        """Append new slot to session."""
        session_id = session.session_id
        logical_blocks = session.logical_blocks

        assert session_id in self.block_tables
        block_table = self.block_tables[session_id]

        while len(logical_blocks) > len(block_table):
            block = self.gpu_allocator.allocate()
            block_table.append(block)

    def _can_swap(self, session: SchedulerSession, allocator: BlockAllocator):
        """Check if swap can be performed."""
        block_table = self.get_block_table(session)
        assert block_table is not None

        num_free_blocks = allocator.get_num_free_blocks()
        return num_free_blocks > len(block_table)

    def can_swap_in(self, session: SchedulerSession):
        """Check if the session can be swapped in."""
        return self._can_swap(session, self.gpu_allocator)

    def swap_in(self, session: SchedulerSession):
        """Swap the session into GPU."""
        block_table = self.get_block_table(session)
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

    def can_swap_out(self, session: SchedulerSession):
        """Check if the session can be swap out."""
        return self._can_swap(session, self.cpu_allocator)

    def swap_out(self, session: SchedulerSession):
        """Swap the session out to host."""
        block_table = self.get_block_table(session)
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
