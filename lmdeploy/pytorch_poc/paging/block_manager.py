# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm
from typing import Dict, List

from lmdeploy.pytorch_poc.block import PhysicalTokenBlock
from lmdeploy.pytorch_poc.messages import SchedulerSession


class BlockAllocator:

    def __init__(self, block_size: int, block_num: int, device: str):
        self.block_size = block_size
        self.block_num = block_num
        self.device = device

        free_blocks: List[PhysicalTokenBlock] = [
            PhysicalTokenBlock(device, i, block_size) for i in range(block_num)
        ]
        self.free_blocks = free_blocks

    def allocate(self):
        if len(self.free_blocks) > 0:
            block = self.free_blocks.pop(0)
            block.ref_count += 1
            return block
        else:
            raise MemoryError(f'No free {self.device} memory blocks.')

    def free(self, block: PhysicalTokenBlock):
        if block.ref_count == 0:
            raise ValueError(f'Double free {block}.')
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block)

    def get_num_free_blocks(self):
        return len(self.free_blocks)


BlockTable = List[PhysicalTokenBlock]


class BlockManager:

    def __init__(self, block_size: int, num_gpu_blocks: int,
                 num_cpu_blocks: int) -> None:
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        self.num_cpu_blocks = num_cpu_blocks

        self.gpu_allocator = BlockAllocator(block_size, num_gpu_blocks, 'gpu')
        self.cpu_allocator = BlockAllocator(block_size, num_cpu_blocks, 'cpu')

        self.block_tables: Dict[int, BlockTable] = {}

    def get_block_table(self, session: SchedulerSession):
        session_id = session.session_id
        if session_id in self.block_tables:
            return self.block_tables[session_id]
        else:
            return None

    def can_allocate(self, session: SchedulerSession):
        required_blocks = len(session.logical_blocks)
        return required_blocks <= self.gpu_allocator.get_num_free_blocks()

    def allocate(self, session: SchedulerSession):
        assert session.session_id not in self.block_tables
        block_table: BlockTable = []
        logical_blocks = session.logical_blocks

        for _ in logical_blocks:
            phy_block = self.gpu_allocator.allocate()
            block_table.append(phy_block)

        self.block_tables[session.session_id] = block_table

    def _free_block_table(self, block_table: BlockTable):
        for block in block_table:
            if block.device == 'cpu':
                self.cpu_allocator.free(block)
            elif block.device == 'gpu':
                self.gpu_allocator.free(block)
            else:
                raise ValueError(f'Can not free block {block}.')

    def free(self, session: SchedulerSession):
        session_id = session.session_id
        if session_id not in self.block_tables:
            return

        block_table = self.block_tables[session_id]
        self._free_block_table(block_table)
        self.block_tables.pop(session_id)

    def can_append_slot(self, session: SchedulerSession):
        session_id = session.session_id
        num_blocks = len(session.logical_blocks)
        assert session_id in self.block_tables
        block_table = self.block_tables[session_id]
        return num_blocks - len(
            block_table) <= self.gpu_allocator.get_num_free_blocks()

    def append_slot(self, session: SchedulerSession):
        session_id = session.session_id
        logical_blocks = session.logical_blocks

        assert session_id in self.block_tables
        block_table = self.block_tables[session_id]

        while len(logical_blocks) > len(block_table):
            block = self.gpu_allocator.allocate()
            block_table.append(block)

    def _can_swap(self, session: SchedulerSession, allocator: BlockAllocator):
        block_table = self.get_block_table(session)
        assert block_table is not None

        num_free_blocks = allocator.get_num_free_blocks()
        return num_free_blocks > len(block_table)

    def can_swap_in(self, session: SchedulerSession):
        return self._can_swap(session, self.gpu_allocator)

    def swap_in(self, session: SchedulerSession):
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
        return self._can_swap(session, self.cpu_allocator)

    def swap_out(self, session: SchedulerSession):
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
        for block_table in self.block_tables.values():
            self._free_block_table(block_table)
        self.block_tables.clear()

    def get_num_free_gpu_blocks(self) -> int:
        return self.gpu_allocator.get_num_free_blocks()

    def get_num_free_cpu_blocks(self) -> int:
        return self.cpu_allocator.get_num_free_blocks()
