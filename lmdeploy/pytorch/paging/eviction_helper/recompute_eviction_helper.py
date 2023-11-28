# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

from lmdeploy.pytorch.paging.block_manager import BlockManager

from ...messages import SchedulerSequence
from .base_eviction_helper import BaseEvictionHelper, SeqList


class RecomputeEvictionHelper(BaseEvictionHelper):

    def __init__(self, block_manager: BlockManager):
        super().__init__(block_manager)

    def can_swap_out(self, seq: SchedulerSequence):
        return True

    def can_swap_in(self, seq: SchedulerSequence):
        return self.block_manager.can_allocate(seq)

    def need_swap_in(self, seq: SchedulerSequence):
        return False

    def swap_in(self, seq: SchedulerSequence, swap_in_map: Dict[int, int]):
        self.block_manager.allocate(seq)

    def swap_out(self, seq: SchedulerSequence, swap_out_map: Dict[int, int]):
        self.block_manager.free(seq)
        seq.set_step(0)

    def try_swap_out_seqs(self, seqs: SeqList, swap_out_map: Dict[int, int]):
        for seq in seqs:
            if not self.can_swap_out(seq):
                continue
            self.swap_out(seq, )
            swap_out_map.update(self.block_manager.swap_out(seq))
            return True

        return False
