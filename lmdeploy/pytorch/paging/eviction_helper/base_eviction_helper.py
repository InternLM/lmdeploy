# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from ...messages import SchedulerSequence
from ..scheduler import Scheduler

SeqList = List[SchedulerSequence]


class BaseEvictionHelper:
    """Base eviction helper."""

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler
        self.block_manager = scheduler.block_manager
        self.block_trie = scheduler.block_trie
        self.state_manager = scheduler.state_manager
        self.cache_config = scheduler.cache_config

    def need_swap_in(self, seq: SchedulerSequence):
        """Sequence need swap in."""
        raise NotImplementedError('Not implemented.')

    def evict_for_seq(self, seq: SchedulerSequence, evictable_seqs: List[SchedulerSequence], prealloc_size: int):
        """Evict seqs."""
        raise NotImplementedError('Not implemented.')
