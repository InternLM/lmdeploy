# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from ...block import LogicalTokenBlocks
from ...messages import SchedulerSequence
from .default_block_manager import DefaultBlockManager

BlockTable = np.ndarray


def _num_blocks_to_drop(seq: SchedulerSequence, window_size: int):
    """Num blocks to free."""
    history_len = seq.num_history_ids
    if seq.num_history_ids <= window_size:
        return 0
    block_size = seq.block_size
    num_blocks = len(seq.logical_blocks)
    win_start_block_id = (history_len - window_size) // block_size
    win_end_block_id = (history_len - 1) // block_size
    num_win_blocks = win_end_block_id - win_start_block_id + 1
    return max(0, num_blocks - num_win_blocks)


class WindowBlockManager(DefaultBlockManager):
    """Manage the usage of blocks, generate block tables.

    Args:
        num_gpu_blocks (int): number of gpu blocks.
        num_cpu_blocks (int): number of cpu blocks.
    """

    def __init__(self, num_gpu_blocks: int, num_cpu_blocks: int, window_size: int, num_gpu_reserved: int = 0):
        super().__init__(num_gpu_blocks, num_cpu_blocks, num_gpu_reserved)
        assert window_size > 0, ('expect window size > 0, '
                                 f'but get window_size = {window_size}')
        self.window_size = window_size

    def num_required_blocks(self, obj: SchedulerSequence, prealloc_size: int = 0):
        """Get num required blocks."""

        # blocks is not enough
        if obj.num_history_ids <= self.window_size:
            return super().num_required_blocks(obj, prealloc_size)

        return super().num_required_blocks(obj, prealloc_size) - obj.num_ignored_history // obj.block_size

    def can_allocate(self, msg: SchedulerSequence, prealloc_size: int = 0):
        """Return if physical block can be allocated for given message."""
        num_drop_blocks = _num_blocks_to_drop(msg, self.window_size)
        num_required_blocks = self.num_required_blocks(msg, prealloc_size)
        num_free_phy = self.get_num_free_gpu_blocks()
        return num_required_blocks <= num_free_phy + num_drop_blocks

    def allocate_msg(self, msg: SchedulerSequence, prealloc_size: int = 0):
        """Allocate physical blocks for given message according to logical
        blocks."""
        logical_blocks = msg.logical_blocks

        def __get_droped_blocks(num_drop_blocks):
            """Get dropped blocks."""
            nonlocal logical_blocks
            droped_blocks = None
            if num_drop_blocks > 0:
                remain_blocks = logical_blocks[num_drop_blocks:]
                droped_blocks = logical_blocks[:num_drop_blocks]
                logical_blocks = LogicalTokenBlocks(remain_blocks)
                msg.logical_blocks = logical_blocks
            return droped_blocks

        def __reuse_droped_blocks(num_required_blocks, num_drop_blocks, droped_blocks):
            """Reuse dropped blocks."""
            num_used_blocks = min(num_drop_blocks - num_required_blocks, num_required_blocks)
            if num_used_blocks > 0:
                reused_blocks = droped_blocks[:num_used_blocks]
            else:
                reused_blocks = droped_blocks
            logical_blocks.append(reused_blocks)

            if num_used_blocks > 0:
                droped_blocks = droped_blocks[num_used_blocks:]
            else:
                num_used_blocks = num_drop_blocks
                droped_blocks = None
            num_required_blocks = num_required_blocks - num_used_blocks
            return num_required_blocks, droped_blocks

        num_drop_blocks = _num_blocks_to_drop(msg, self.window_size)
        num_required_blocks = self.num_required_blocks(msg, prealloc_size)
        msg.num_ignored_history += num_drop_blocks * msg.block_size

        droped_blocks = __get_droped_blocks(num_drop_blocks)

        if num_required_blocks > 0:
            if num_drop_blocks > 0:
                num_required_blocks, droped_blocks = __reuse_droped_blocks(num_required_blocks, num_drop_blocks,
                                                                           droped_blocks)
            if num_required_blocks > 0:
                blocks = self.allocator.allocate(num_required_blocks, 'gpu')
                logical_blocks.append(blocks)

        # drop unused blocks
        if droped_blocks is not None:
            self.allocator.free(droped_blocks)
