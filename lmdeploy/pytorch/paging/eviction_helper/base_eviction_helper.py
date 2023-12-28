# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

from ...messages import SchedulerSequence
from ..block_manager import BlockManager

SeqList = List[SchedulerSequence]


class BaseEvictionHelper:
    """Base eviction helper."""

    def __init__(self, block_manager: BlockManager):
        self.block_manager: BlockManager = block_manager

    def need_swap_in(self, seq: SchedulerSequence):
        """sequence need swap in."""
        raise NotImplementedError('Not implemented.')

    def try_swap_out(self, seq: SchedulerSequence, swap_out_map: Dict[int,
                                                                      int]):
        """try swap out."""
        raise NotImplementedError('Not implemented.')

    def try_swap_in(self, seq: SchedulerSequence, swap_in_map: Dict[int, int]):
        """try swap in."""
        raise NotImplementedError('Not implemented.')

    def try_swap_out_seqs(self, seqs: SeqList, swap_out_map: Dict[int, int]):
        """try swap sequence out."""
        for seq in reversed(seqs):
            if self.try_swap_out(seq, swap_out_map):
                return True
        return False

    def try_swap_out_unused(self, hanging: SeqList, waiting: SeqList,
                            swap_out_map: Dict[int, int]):
        """try swap out hanging and waiting sequence."""
        if self.try_swap_out_seqs(hanging, swap_out_map):
            return True
        else:
            return self.try_swap_out_seqs(waiting, swap_out_map)
