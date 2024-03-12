# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import numpy as np

from ...adapter.adapter import ADAPTER_MANAGER, SchedulerAdapter
from ...block import LogicalTokenBlocks
from ...messages import SchedulerSequence
from .default_block_manager import DefaultBlockManager

BlockTable = np.ndarray


def _div_up(x, n):
    """perform div up."""
    return (x + n - 1) // n


def _last_block_size(history_len: int, block_size: int):
    """last block size."""
    last = history_len % block_size
    last = last if last != 0 else block_size
    return last


def _num_blocks_to_drop(seq: SchedulerSequence, window_size: int):
    """num blocks to free."""
    if seq.history_len <= window_size:
        return 0
    block_size = seq.block_size
    history_len = seq.history_len
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

    def __init__(self, num_gpu_blocks: int, num_cpu_blocks: int,
                 window_size: int):
        super().__init__(num_gpu_blocks, num_cpu_blocks)
        assert window_size > 0, ('expect window size > 0, '
                                 f'but get window_size = {window_size}')
        self.window_size = window_size

    @classmethod
    def num_required_blocks(cls, obj: Union[SchedulerSequence,
                                            SchedulerAdapter]):
        """get num required blocks."""

        def __num_req_seq(seq: SchedulerSequence):
            """get num required seq blocks."""
            block_size = seq.block_size
            lb_tokens = cls.last_block_size(seq)
            lb_remain_tokens = 0
            if len(seq.logical_blocks) > 0:
                lb_remain_tokens = block_size - lb_tokens
            num_input_tokens = len(seq.token_ids)
            num_req_tokens = max(0, num_input_tokens - lb_remain_tokens)
            return _div_up(num_req_tokens, block_size)

        def __num_req_adapter(adapter: SchedulerAdapter):
            """get num required adapter blocks."""
            if adapter.is_actived():
                return 0
            else:
                return adapter.rank * len(adapter.target_modules)

        if isinstance(obj, SchedulerSequence):
            return __num_req_seq(obj)
        else:
            return __num_req_adapter(obj)

    @classmethod
    def last_block_size(cls, seq: SchedulerSequence) -> int:
        """get last block size."""
        num_blocks = len(seq.logical_blocks)
        if num_blocks == 0:
            return 0
        return _last_block_size(seq.history_len, seq.block_size)

    def can_allocate(self, msg: SchedulerSequence):
        """Return if physical block can be allocated for given message."""
        num_drop_blocks = _num_blocks_to_drop(msg, self.window_size)
        num_required_blocks = self.num_required_blocks(msg)
        num_free_phy = self.get_num_free_gpu_blocks()
        if msg.adapter_name is not None:
            adapter = ADAPTER_MANAGER.get_adapter(msg.adapter_name)
            num_required_blocks += self.num_required_blocks(adapter)
        return num_required_blocks <= num_free_phy + num_drop_blocks

    def allocate_msg(self, msg: SchedulerSequence):
        """Allocate physical blocks for given message according to logical
        blocks."""
        logical_blocks = msg.logical_blocks

        def __get_droped_blocks(num_drop_blocks):
            """get dropped blocks."""
            nonlocal logical_blocks
            droped_blocks = None
            if num_drop_blocks > 0:
                remain_blocks = logical_blocks[num_drop_blocks:]
                droped_blocks = logical_blocks[:num_drop_blocks]
                logical_blocks = LogicalTokenBlocks(remain_blocks)
                msg.logical_blocks = logical_blocks
            return droped_blocks

        def __reuse_droped_blocks(num_required_blocks, num_drop_blocks,
                                  droped_blocks):
            """reuse dropped blocks."""
            num_used_blocks = min(num_drop_blocks - num_required_blocks,
                                  num_required_blocks)
            if num_used_blocks > 0:
                reused_blocks = droped_blocks[:num_used_blocks]
            else:
                # when num_required_blocks > num_drop_blocks, all blocks in num_drop_blocks is used
                reused_blocks = droped_blocks
            logical_blocks.append(reused_blocks)

            num_drop_blocks = num_drop_blocks - num_used_blocks
            if num_used_blocks > 0:
                droped_blocks = droped_blocks[num_used_blocks:]
            else:
                num_used_blocks = num_drop_blocks
                droped_blocks = None
            num_required_blocks = num_required_blocks - num_used_blocks
            return num_required_blocks, droped_blocks

        num_drop_blocks = _num_blocks_to_drop(msg, self.window_size)
        num_required_blocks = self.num_required_blocks(msg)

        droped_blocks = __get_droped_blocks(num_drop_blocks)

        if num_required_blocks > 0:
            if num_drop_blocks > 0:
                num_required_blocks, droped_blocks = __reuse_droped_blocks(
                    num_required_blocks, num_drop_blocks, droped_blocks)
            if num_required_blocks > 0:
                blocks = self.allocator.allocate(num_required_blocks, 'gpu')
                logical_blocks.append(blocks)

        # drop unused blocks
        if droped_blocks is not None:
            self.allocator.free(droped_blocks)

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

    def can_append_slot(self, msg: SchedulerSequence):
        """Return true if the message can append new slot."""
        return self.can_allocate(msg)

    def append_slot(self, msg: SchedulerSequence):
        """Append new slot to message."""
        return self.allocate(msg)
