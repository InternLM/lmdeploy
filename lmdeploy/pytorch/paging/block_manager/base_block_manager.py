# Copyright (c) OpenMMLab. All rights reserved.
import time
from typing import Dict

import numpy as np

from ...messages import SchedulerSequence


class LogicalMemory:
    """Logical memory blocks."""

    def __init__(self, num_blocks: int) -> None:
        self._num_blocks = num_blocks

        self.phy_map: np.ndarray = np.zeros(self._num_blocks, dtype=np.int64)
        self.ref_count: np.ndarray = np.zeros((self._num_blocks, ),
                                              dtype=np.int64)
        self.access_time: np.ndarray = np.zeros((self._num_blocks, ),
                                                dtype=np.int64)

    def get_physical_blocks(self, logical_address: np.ndarray):
        """get physical address."""
        if isinstance(logical_address,
                      np.ndarray) and len(logical_address) == 0:
            return np.empty((0, ), dtype=np.int64)
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
            self._free_count -= num_blocks
            return blocks
        else:
            raise MemoryError('No enough free memory blocks.')

    def free(self, blocks: np.ndarray):
        """Free block to block pool."""
        freed_blocks = blocks
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
            self._log_mem.ref_count.put(blocks, 1)
            self.update_access_time(blocks)
            self._free_count -= num_blocks
            return blocks.copy()
        else:
            raise MemoryError('No enough free memory blocks.')

    def free(self, blocks: np.ndarray):
        """Free logical block."""

        self.add_ref_count(blocks, -1)
        self.update_access_time(blocks)
        ref_count = self.get_ref_count(blocks)
        freed_blocks = blocks[ref_count == 0]
        num_freed_blocks = len(freed_blocks)
        if num_freed_blocks <= 0:
            return

        # free logical
        num_used = self._num_blocks - self._free_count
        self._free_blocks[num_used - num_freed_blocks:num_used] = freed_blocks
        self._free_count += num_freed_blocks

        # free physical
        phy_blocks = self.get_physical_blocks(freed_blocks)

        cpu_blocks = phy_blocks[phy_blocks >= self._cpu_mem_offset]
        gpu_blocks = phy_blocks[phy_blocks < self._cpu_mem_offset]
        if len(cpu_blocks) > 0:
            self._cpu_allocator.free(cpu_blocks)
        if len(gpu_blocks) > 0:
            self._gpu_allocator.free(gpu_blocks)

    def get_num_free_blocks(self):
        """Get numbers of free blocks."""
        return self._free_count

    def get_physical_blocks(self, blocks: np.ndarray):
        """get physical address."""
        return self._log_mem.get_physical_blocks(blocks)

    def get_ref_count(self, blocks: np.ndarray):
        """get ref count."""
        return self._log_mem.ref_count[blocks]

    def add_ref_count(self, blocks: np.ndarray, value: np.ndarray):
        """update ref count."""
        np.add.at(self._log_mem.ref_count, blocks, value)

    def get_access_time(self, blocks: np.ndarray):
        """get access time."""
        return self._log_mem.access_time[blocks]

    def update_access_time(self, blocks: np.ndarray):
        """update access time."""
        now = time.perf_counter()
        self._log_mem.access_time[blocks] = now

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


class BaseBlockManager:
    """ABC of block manager.

    Args:
        num_gpu_blocks (int): number of gpu blocks.
        num_cpu_blocks (int): number of cpu blocks.
    """

    def __init__(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        self.num_gpu_blocks = num_gpu_blocks
        self.num_cpu_blocks = num_cpu_blocks

        self.allocator = LogicalAllocator(num_cpu_blocks, num_gpu_blocks)

        self.block_tables: Dict[int, BlockTable] = {}

    @classmethod
    def num_required_blocks(cls,
                            obj: SchedulerSequence,
                            prealloc_size: int = 0):
        """get num required blocks."""
        raise NotImplementedError('Not implemented.')

    @classmethod
    def last_block_size(cls, seq: SchedulerSequence) -> int:
        """get last block size."""
        raise NotImplementedError('Not implemented.')

    def can_allocate(self, msg: SchedulerSequence, prealloc_size: int = 0):
        """Return if physical block can be allocated for given message."""
        raise NotImplementedError('Not implemented.')

    def allocate_msg(self, msg: SchedulerSequence, prealloc_size: int = 0):
        """Allocate physical blocks for given message according to logical
        blocks."""
        raise NotImplementedError('Not implemented.')

    def free(self, msg: SchedulerSequence):
        """Free all physical blocks allocated for the session."""
        raise NotImplementedError('Not implemented.')

    def try_swap_out(self, msg: SchedulerSequence):
        """Try swap msg out."""
        raise NotImplementedError('Not implemented.')

    def try_swap_in(self, msg: SchedulerSequence):
        """Try swap msg in."""
        raise NotImplementedError('Not implemented.')

    def get_block_table(self, msg: SchedulerSequence):
        """Get the block table of given msg.

        Args:
            msg (SchedulerSequence): The msg to get block table.
        """
        logical_blocks = msg.logical_blocks
        return self.allocator.get_physical_blocks(
            logical_blocks.get_real_blocks())

    def allocate(self, data: SchedulerSequence, prealloc_size: int = 0):
        """allocate stuff."""
        if isinstance(data, SchedulerSequence):
            return self.allocate_msg(data, prealloc_size)
        else:
            raise TypeError(f'Unsupported allocate type: {type(data)}')

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
