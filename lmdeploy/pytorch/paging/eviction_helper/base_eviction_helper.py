# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from ...messages import SchedulerSequence
from ..radix_tree_manager import TreeNode
from ..scheduler import Scheduler

SeqList = List[SchedulerSequence]


class BaseEvictionHelper:
    """Base eviction helper."""

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler
        self.rtree_manager = scheduler.rtree_manager
        self.block_manager = scheduler.block_manager

    def evict_for_seq(self, seq: SchedulerSequence,
                      sorted_nodes: List[TreeNode]):
        """evict until can alloc."""
        raise NotImplementedError('Not implemented.')
