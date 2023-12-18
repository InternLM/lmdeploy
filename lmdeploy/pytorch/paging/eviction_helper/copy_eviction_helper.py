# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

from ...messages import SchedulerSequence
from .base_eviction_helper import BaseEvictionHelper


class CopyEvictionHelper(BaseEvictionHelper):
    """Copy to host memory eviction."""

    def __init__(self, block_manager):
        super().__init__(block_manager)

    def need_swap_in(self, seq: SchedulerSequence):
        """sequence need swap in."""
        return self.block_manager.on_device(seq, 'cpu')

    def try_swap_out(self, seq: SchedulerSequence, swap_out_map: Dict[int,
                                                                      int]):
        """try swap out."""
        success, swap_map = self.block_manager.try_swap_out(seq)
        if success:
            swap_out_map.update(swap_map)
        return success

    def try_swap_in(self, seq: SchedulerSequence, swap_in_map: Dict[int, int]):
        """try swap in."""
        success, swap_map = self.block_manager.try_swap_in(seq)
        if success:
            swap_in_map.update(swap_map)
        return success
