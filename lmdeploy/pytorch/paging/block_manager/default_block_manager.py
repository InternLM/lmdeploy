# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm
from typing import Union

import numpy as np

from ...adapter.adapter import SchedulerAdapter
from ...messages import SchedulerSequence
from .base_block_manager import BaseBlockManager


def _div_up(x, n):
    """perform div up."""
    return (x + n - 1) // n


BlockTable = np.ndarray


class DefaultBlockManager(BaseBlockManager):
    """Manage the usage of blocks, generate block tables.

    Args:
        num_gpu_blocks (int): number of gpu blocks.
        num_cpu_blocks (int): number of cpu blocks.
    """

    @classmethod
    def num_required_blocks(cls,
                            obj: Union[SchedulerSequence, SchedulerAdapter],
                            prealloc_size: int = 0):
        """get num required blocks."""
        if isinstance(obj, SchedulerSequence):
            num_tokens = obj.num_all_tokens() + prealloc_size
            num_all_blocks = _div_up(num_tokens, obj.block_size)
            return max(0, num_all_blocks - len(obj.logical_blocks))
        else:
            if obj.is_actived():
                return 0
            else:
                return obj.num_required_blocks

    @classmethod
    def last_block_size(cls, seq: SchedulerSequence) -> int:
        """get last block size."""
        num_blocks = len(seq.logical_blocks)
        if num_blocks == 0:
            return 0
        elif num_blocks * seq.block_size < seq.history_len:
            return seq.block_size
        return seq.history_len % seq.block_size

    def can_allocate(self, msg: SchedulerSequence, prealloc_size: int = 0):
        """Return if physical block can be allocated for given message."""
        num_required_blocks = self.num_required_blocks(msg, prealloc_size)
        num_free_phy = self.get_num_free_gpu_blocks()
        if msg.adapter_name is not None:
            adapter = self.adapter_manager.get_adapter(msg.adapter_name)
            num_required_blocks += self.num_required_blocks(adapter)
        return num_required_blocks <= num_free_phy

    def allocate_msg(self, msg: SchedulerSequence, prealloc_size: int = 0):
        """Allocate physical blocks for given message according to logical
        blocks."""
        logical_blocks = msg.logical_blocks
        num_required_blocks = self.num_required_blocks(msg, prealloc_size)
        if num_required_blocks > 0:
            blocks = self.allocator.allocate(num_required_blocks, 'gpu')
            logical_blocks.append(blocks)

    def allocate_adapter(self, adapter: SchedulerAdapter):
        """Allocate cpu blocks for given adapter."""
        num_required_blocks = self.num_required_blocks(adapter)
        if num_required_blocks > 0:
            blocks = self.allocator.allocate(num_required_blocks, 'cpu')
            adapter.logical_blocks.append(blocks)

    def free(self, msg: SchedulerSequence):
        """Free all physical blocks allocated for the session."""
        self.allocator.free(msg.logical_blocks.get_real_blocks())
        msg.logical_blocks.reset()

    def try_swap_out(self, msg: Union[SchedulerSequence, SchedulerAdapter]):
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
            ref_count = self.allocator.get_ref_count(logical_blocks)
            if np.count_nonzero(ref_count != 1) > 0:
                return False

            return True

        def _do_swap():
            """perform swap."""
            new_blocks = cpu_allocator.allocate(len(logical_blocks))

            old_blocks = phy_blocks
            swap_map = dict(zip(old_blocks, new_blocks - self.num_gpu_blocks))

            gpu_allocator.free(old_blocks)
            self.allocator.update_phy_map(logical_blocks.get_real_blocks(),
                                          new_blocks)
            if isinstance(msg, SchedulerAdapter):
                msg.active(False)
            return True, swap_map

        if not _can_swap():
            return False, swap_map
        else:
            return _do_swap()

    def try_swap_in(self, msg: Union[SchedulerSequence, SchedulerAdapter]):
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
            ref_count = self.allocator.get_ref_count(logical_blocks)
            if np.count_nonzero(ref_count != 1) > 0:
                return False

            return True

        def _do_swap():
            """perform swap."""
            new_blocks = gpu_allocator.allocate(len(logical_blocks))

            old_blocks = phy_blocks
            swap_map = dict(zip(old_blocks - self.num_gpu_blocks, new_blocks))

            cpu_allocator.free(old_blocks)
            self.allocator.update_phy_map(logical_blocks.get_real_blocks(),
                                          new_blocks)
            if isinstance(msg, SchedulerAdapter):
                msg.active(True)
            return True, swap_map

        if not _can_swap():
            return False, swap_map
        else:
            return _do_swap()
