# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

from ...messages import SchedulerSequence
from .base_eviction_helper import BaseEvictionHelper, SeqList


class CopyEvictionHelper(BaseEvictionHelper):

    def __init__(self, block_manager):
        super().__init__(block_manager)

    def can_swap_out(self, seq: SchedulerSequence):
        block_table = self.block_manager.get_block_table(seq)
        if block_table is None or len(block_table) == 0:
            return False
        first_block = block_table[0]
        device = first_block.device
        return device == 'gpu'

    def can_swap_in(self, seq: SchedulerSequence):
        return self.block_manager.can_swap_in(seq)

    def need_swap_in(self, seq: SchedulerSequence):
        block_table = self.block_manager.get_block_table(seq)
        if block_table is None or len(block_table) == 0:
            return False
        first_block = block_table[0]
        device = first_block.device
        return device == 'cpu'

    def swap_in(self, seq: SchedulerSequence, swap_in_map: Dict[int, int]):
        swap_in_map.update(self.block_manager.swap_in(seq))

    def swap_out(self, seq: SchedulerSequence, swap_out_map: Dict[int, int]):
        swap_out_map.update(self.block_manager.swap_out(seq))

    def try_swap_out_seqs(self, seqs: SeqList, swap_out_map: Dict[int, int]):
        for seq in seqs:
            if not self.can_swap_out(seq):
                continue
            if not self.block_manager.can_swap_out(seq):
                continue
            self.swap_out(seq, swap_out_map)
            return True

        return False
