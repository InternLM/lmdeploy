# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

from ...messages import SchedulerSequence
from ..block_manager import BlockManager

SeqList = List[SchedulerSequence]


class BaseEvictionHelper:

    def __init__(self, block_manager: BlockManager):
        self.block_manager: BlockManager = block_manager

    def can_swap_out(self, seq: SchedulerSequence):
        raise NotImplementedError('Not implemented.')

    def can_swap_in(self, seq: SchedulerSequence):
        raise NotImplementedError('Not implemented.')

    def need_swap_in(self, seq: SchedulerSequence):
        raise NotImplementedError('Not implemented.')

    def try_swap_out_seqs(self, seqs: SeqList, swap_out_map: Dict[int, int]):
        raise NotImplementedError('Not implemented.')

    def swap_in(self, seq: SchedulerSequence, swap_in_map: Dict[int, int]):
        raise NotImplementedError('Not implemented.')

    def swap_out(self, seq: SchedulerSequence, swap_out_map: Dict[int, int]):
        raise NotImplementedError('Not implemented.')

    def try_swap_out(self, hanging: SeqList, waiting: SeqList,
                     swap_out_map: Dict[int, int]):
        if self.try_swap_out_seqs(hanging, swap_out_map):
            return True
        else:
            return self.try_swap_out_seqs(waiting, swap_out_map)
