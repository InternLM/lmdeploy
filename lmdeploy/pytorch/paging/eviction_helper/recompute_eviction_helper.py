# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

from lmdeploy.pytorch.paging.block_manager import BlockManager

from ...messages import SchedulerSequence
from .base_eviction_helper import BaseEvictionHelper


def _div_up(x, n):
    """perform div up."""
    return (x + n - 1) // n


class RecomputeEvictionHelper(BaseEvictionHelper):
    """recompute eviction."""

    def __init__(self, block_manager: BlockManager):
        super().__init__(block_manager)

    def need_swap_in(self, seq: SchedulerSequence):
        """sequence need swap in."""
        return False

    def swap_in(self, seq: SchedulerSequence, swap_in_map: Dict[int, int]):
        """sequence swap in."""
        self.block_manager.allocate(seq)

    def swap_out(self, seq: SchedulerSequence, swap_out_map: Dict[int, int]):
        """sequence swap out."""
        self.block_manager.free(seq)
        seq.set_step(0)
        # num_blocks = _div_up(seq, seq.block_size)
        # seq.logical_blocks.resize(num_blocks)
        seq.logical_blocks.reset()

    def try_swap_out(self, seq: SchedulerSequence, swap_out_map: Dict[int,
                                                                      int]):
        """try swap out."""
        if seq.history_len > 0:
            self.swap_out(seq, swap_out_map)
            return True
        else:
            return False

    def try_swap_in(self, seq: SchedulerSequence, swap_in_map: Dict[int, int]):
        """try swap in."""
        if self.block_manager.can_allocate(seq):
            self.swap_in(seq, swap_in_map)
            return True
        else:
            return False
