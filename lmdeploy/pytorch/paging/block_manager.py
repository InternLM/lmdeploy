# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm
from typing import Dict

import numpy as np

from ..messages import SchedulerSequence


class LogicalMemory:
    """Logical memory blocks."""

    def __init__(self, num_blocks: int) -> None:
        self._num_blocks = num_blocks

        self.phy_map: np.ndarray = np.zeros(self._num_blocks, dtype=np.int64)

    def get_physical_blocks(self, logical_address: np.ndarray):
        """get physical address."""
        return self.phy_map[logical_address]

    def num_blocks(self):
        """get num blocks."""
        return self._num_blocks


class PhysicalMemory:
    """physical memory blocks."""

    def __init__(self, num_cpu_blocks: int, num_gpu_blocks: int) -> None:
        self._num_cpu_blocks = num_cpu_blocks
        self._num_gpu_blocks = num_gpu_blocks
        self._num_blocks = num_cpu_blocks + num_gpu_blocks

        self.ref_count: np.ndarray = np.zeros((self._num_blocks, ),
                                              dtype=np.int64)
        self.swap_count: np.ndarray = np.zeros((self._num_blocks, ),
                                               dtype=np.int64)

    def num_cpu_blocks(self):
        """get num cpu blocks."""
        return self._num_cpu_blocks

    def num_gpu_blocks(self):
        """get num gpu blocks."""
        return self._num_gpu_blocks


class PhysicalAllocator:
    """The physical block allocator.

    The allocator won't allocate real memory. It is used to support block
    manager.
    """

    def __init__(self,
                 memory: PhysicalMemory,
                 num_blocks: int,
                 offset: int = 0):
        self._mem = memory
        self._num_blocks = num_blocks
        self._offset = offset

        self._free_blocks = np.arange(num_blocks, dtype=np.int64) + offset
        self._free_count = num_blocks

    def allocate(self, num_blocks: int):
        """Allocate block from block pool."""
        if self.get_num_free_blocks() >= num_blocks:
            num_used = self._num_blocks - self._free_count
            blocks = self._free_blocks[num_used:num_used + num_blocks]
            self._mem.ref_count.put(blocks, 1)
            self._free_count -= num_blocks
            return blocks
        else:
            raise MemoryError('No enough free memory blocks.')

    def free(self, blocks: np.ndarray):
        """Free block to block pool."""
        np.add.at(self._mem.ref_count, blocks, -1)
        ref_count = self.get_ref_count(blocks)
        freed_blocks = blocks[ref_count == 0]
        num_freed_blocks = len(freed_blocks)
        if num_freed_blocks > 0:
            num_used = self._num_blocks - self._free_count
            self._free_blocks[num_used -
                              num_freed_blocks:num_used] = freed_blocks
            self._free_count += num_freed_blocks
        return freed_blocks

    def get_num_free_blocks(self):
        """Get numbers of free blocks."""
        return self._free_count

    def get_ref_count(self, blocks: np.ndarray):
        """get ref count."""
        return self._mem.ref_count[blocks]


class LogicalAllocator:
    """The logical block allocator."""

    def __init__(self, num_cpu_blocks: int, num_gpu_blocks: int) -> None:
        self._log_mem = LogicalMemory(num_cpu_blocks + num_gpu_blocks)
        self._phy_mem = PhysicalMemory(num_cpu_blocks, num_gpu_blocks)

        self._cpu_mem_offset = num_gpu_blocks
        self._gpu_allocator = PhysicalAllocator(self._phy_mem, num_gpu_blocks,
                                                0)
        self._cpu_allocator = PhysicalAllocator(self._phy_mem, num_cpu_blocks,
                                                self._cpu_mem_offset)

        num_blocks = self._log_mem.num_blocks()
        self._num_blocks = num_blocks
        self._free_blocks = np.arange(num_blocks)
        self._free_count = num_blocks

    def get_phy_allocator(self, device: str):
        """get allocator."""
        if device == 'gpu':
            return self._gpu_allocator
        elif device == 'cpu':
            return self._cpu_allocator
        else:
            raise ValueError(f'Unsupported device: {device}')

    def allocate(self, num_blocks: int, device: str = 'gpu'):
        """allocate logical blocks."""
        if num_blocks == 0:
            return np.empty((0, ), dtype=np.int64)
        phy_allocator = self.get_phy_allocator(device)
        logical_enable = self.get_num_free_blocks() >= num_blocks
        physical_enable = phy_allocator.get_num_free_blocks() >= num_blocks
        if logical_enable and physical_enable:
            num_used = self._num_blocks - self._free_count
            blocks = self._free_blocks[num_used:num_used + num_blocks]
            phy_blocks = phy_allocator.allocate(num_blocks)
            self._log_mem.phy_map.put(blocks, phy_blocks)
            self._free_count -= num_blocks
            return blocks.copy()
        else:
            raise MemoryError('No enough free memory blocks.')

    def free(self, blocks: np.ndarray):
        """Free logical block."""
        phy_blocks = self.get_physical_blocks(blocks)

        cpu_blocks = phy_blocks[phy_blocks >= self._cpu_mem_offset]
        gpu_blocks = phy_blocks[phy_blocks < self._cpu_mem_offset]
        if len(cpu_blocks) > 0:
            self._cpu_allocator.free(cpu_blocks)
        if len(gpu_blocks) > 0:
            self._gpu_allocator.free(gpu_blocks)

        ref_count = self._phy_mem.ref_count[phy_blocks]
        freed_blocks = blocks[ref_count == 0]
        num_freed_blocks = len(freed_blocks)
        if num_freed_blocks > 0:
            num_used = self._num_blocks - self._free_count
            self._free_blocks[num_used -
                              num_freed_blocks:num_used] = freed_blocks
            self._free_count += num_freed_blocks

    def get_num_free_blocks(self):
        """Get numbers of free blocks."""
        return self._free_count

    def get_physical_blocks(self, blocks: np.ndarray):
        """get physical address."""
        return self._log_mem.get_physical_blocks(blocks)

    def get_ref_count(self, blocks: np.ndarray):
        """get ref count."""
        phy_blocks = self.get_physical_blocks(blocks)
        return self._phy_mem.ref_count[phy_blocks]

    def add_ref_count(self, blocks: np.ndarray, value: np.ndarray):
        """update ref count."""
        phy_blocks = self.get_physical_blocks(blocks)
        np.add.at(self._phy_mem.ref_count, phy_blocks, value)

    def cpu_mem_offset(self):
        """get cpu mem offset in unified physical memory."""
        return self._cpu_mem_offset

    def count_cpu_blocks(self, blocks: np.ndarray):
        """count cpu blocks."""
        phy_blocks = self.get_physical_blocks(blocks)
        return np.count_nonzero(phy_blocks >= self.cpu_mem_offset())

    def count_gpu_blocks(self, blocks: np.ndarray):
        """count gpu blocks."""
        phy_blocks = self.get_physical_blocks(blocks)
        return np.count_nonzero(phy_blocks < self.cpu_mem_offset())

    def update_phy_map(self, log_blocks: np.ndarray, phy_blocks: np.ndarray):
        """update physical map."""
        assert len(phy_blocks) == len(log_blocks)
        self._log_mem.phy_map.put(log_blocks, phy_blocks)

    def on_device(self, blocks: np.ndarray, device: str):
        """blocks on given device."""
        if len(blocks) == 0:
            return False

        # TODO: check all blocks
        cpu_mem_offset = self.cpu_mem_offset()

        phy_blocks = self.get_physical_blocks(blocks[:1])
        if phy_blocks[0] < cpu_mem_offset:
            phy_device = 'gpu'
        else:
            phy_device = 'cpu'
        return device == phy_device


BlockTable = np.ndarray


class BlockManager:
    """Manage the usage of blocks, generate block tables.

    Args:
        num_gpu_blocks (int): number of gpu blocks.
        num_cpu_blocks (int): number of cpu blocks.
    """

    def __init__(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        self.num_gpu_blocks = num_gpu_blocks
        self.num_cpu_blocks = num_cpu_blocks

        self.allocator = LogicalAllocator(num_cpu_blocks, num_gpu_blocks)

        self.block_tables: Dict[int, BlockTable] = {}

    def get_block_table(self, msg: SchedulerSequence):
        """Get the block table of given msg.

        Args:
            msg (SchedulerSequence): The msg to get block table.
        """
        logical_blocks = msg.logical_blocks
        return self.allocator.get_physical_blocks(
            logical_blocks.get_real_blocks())

    def can_allocate(self, msg: SchedulerSequence):
        """Return if physical block can be allocated for given message."""
        num_required_blocks = msg.num_required_blocks()
        num_free_phy = self.get_num_free_gpu_blocks()
        return num_required_blocks <= num_free_phy

    def allocate(self, msg: SchedulerSequence):
        """Allocate physical blocks for given message according to logical
        blocks."""
        logical_blocks = msg.logical_blocks
        num_required_tokens = msg.num_required_tokens()
        num_required_blocks = logical_blocks.num_required_blocks(
            num_required_tokens)
        if num_required_blocks > 0:
            blocks = self.allocator.allocate(num_required_blocks, 'gpu')
            logical_blocks.append(blocks)
            logical_blocks.add_tokens(num_required_tokens)

    def free(self, msg: SchedulerSequence):
        """Free all physical blocks allocated for the session."""
        self.allocator.free(msg.logical_blocks.get_real_blocks())
        msg.logical_blocks.reset()

    def can_append_slot(self, msg: SchedulerSequence):
        """Return true if the message can append new slot."""
        return self.can_allocate(msg)

    def append_slot(self, msg: SchedulerSequence):
        """Append new slot to message."""
        return self.allocate(msg)

    def can_fork(self, from_msg: SchedulerSequence):
        """Return true if blocks can be folked."""
        logical_blocks = from_msg.logical_blocks
        if logical_blocks.last_block_size() == logical_blocks.get_block_size():
            return True

        cpu_mem_offset = self.allocator.cpu_mem_offset()
        phy_block = self.allocator.get_physical_blocks(logical_blocks[-1])
        if phy_block < cpu_mem_offset:
            device = 'gpu'
        else:
            device = 'cpu'
        phy_allocator = self.allocator.get_phy_allocator(device)
        return phy_allocator.get_num_free_blocks() >= 1

    def fork(self, from_msg: SchedulerSequence, to_msg: SchedulerSequence):
        """Fork new message."""

        def _copy_lask_block(logical_blocks, copy_map):
            cpu_mem_offset = self.allocator.cpu_mem_offset()
            phy_block = self.allocator.get_physical_blocks(logical_blocks[-1])
            if phy_block < cpu_mem_offset:
                device = 'gpu'
            else:
                device = 'cpu'
            block = self.allocator.allocate(1, device)
            new_phy_block = self.allocator.get_physical_blocks(block[0])
            copy_map[phy_block] = new_phy_block
            return block[0]

        logical_blocks = from_msg.logical_blocks
        copy_map: Dict[int, int] = dict()
        if logical_blocks.last_block_size() == logical_blocks.get_block_size():
            self.allocator.add_ref_count(logical_blocks, 1)
        else:
            new_logical_blocks = logical_blocks.clone()
            self.allocator.add_ref_count(new_logical_blocks[:-1], 1)
            block = _copy_lask_block(logical_blocks, copy_map)
            new_logical_blocks[-1] = block
            to_msg.logical_blocks = new_logical_blocks

        return copy_map

    def try_swap_out(self, msg: SchedulerSequence):
        """Try swap msg out."""
        swap_map = dict()
        logical_blocks = msg.logical_blocks
        cpu_mem_offset = self.allocator.cpu_mem_offset()
        phy_blocks = self.allocator.get_physical_blocks(logical_blocks)
        cpu_allocator = self.allocator.get_phy_allocator('cpu')
        gpu_allocator = self.allocator.get_phy_allocator('gpu')

        def _can_swap():
            """check swap."""
            if len(logical_blocks) == 0:
                return False

            # we only support all blocks of a sequence on same device
            if phy_blocks[0] >= cpu_mem_offset:
                return False

            # no free blocks
            num_free = self.get_num_free_cpu_blocks()
            if num_free < len(phy_blocks):
                return False

            # don't swap sequence with multiple reference
            ref_count = gpu_allocator.get_ref_count(phy_blocks)
            if np.count_nonzero(ref_count != 1) > 0:
                return False

            return True

        def _do_swap():
            """perform swap."""
            new_blocks = cpu_allocator.allocate(len(logical_blocks))

            old_blocks = phy_blocks
            swap_map = dict(zip(old_blocks, new_blocks))

            gpu_allocator.free(old_blocks)
            self.allocator.update_phy_map(logical_blocks.get_real_blocks(),
                                          new_blocks)
            return True, swap_map

        if not _can_swap():
            return False, swap_map
        else:
            return _do_swap()

    def try_swap_in(self, msg: SchedulerSequence):
        """Try swap msg in."""
        swap_map = dict()
        logical_blocks = msg.logical_blocks
        cpu_mem_offset = self.allocator.cpu_mem_offset()
        phy_blocks = self.allocator.get_physical_blocks(logical_blocks)
        cpu_allocator = self.allocator.get_phy_allocator('cpu')
        gpu_allocator = self.allocator.get_phy_allocator('gpu')

        def _can_swap():
            """check swap."""
            if len(logical_blocks) == 0:
                return False

            # we only support all blocks of a sequence on same device
            if phy_blocks[0] < cpu_mem_offset:
                return False

            # no free blocks
            num_free = self.get_num_free_gpu_blocks()
            if num_free < len(phy_blocks):
                return False

            # don't swap sequence with multiple reference
            ref_count = cpu_allocator.get_ref_count(phy_blocks)
            if np.count_nonzero(ref_count != 1) > 0:
                return False

            return True

        def _do_swap():
            """perform swap."""
            new_blocks = gpu_allocator.allocate(len(logical_blocks))

            old_blocks = phy_blocks
            swap_map = dict(zip(old_blocks, new_blocks))

            cpu_allocator.free(old_blocks)
            self.allocator.update_phy_map(logical_blocks.get_real_blocks(),
                                          new_blocks)
            return True, swap_map

        if not _can_swap():
            return False, swap_map
        else:
            return _do_swap()

    def get_num_free_gpu_blocks(self) -> int:
        """Get number of free gpu blocks."""
        return self.allocator.get_phy_allocator('gpu').get_num_free_blocks()

    def get_num_free_cpu_blocks(self) -> int:
        """Get number of free cpu blocks."""
        return self.allocator.get_phy_allocator('cpu').get_num_free_blocks()

    def on_device(self, msg: SchedulerSequence, device: str):
        allocator = self.allocator
        logical_blocks = msg.logical_blocks
        return allocator.on_device(logical_blocks.get_real_blocks(), device)
