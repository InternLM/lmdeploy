# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from ...adapter.adapter import ADAPTER_MANAGER
from ...messages import SchedulerSequence
from ..radix_tree_manager import TreeNode
from .base_eviction_helper import BaseEvictionHelper


class RecomputeEvictionHelper(BaseEvictionHelper):
    """recompute eviction."""

    def evict_for_seq(self, seq: SchedulerSequence,
                      sorted_nodes: List[TreeNode]):
        """evict until can alloc."""
        num_required_blocks = self.block_manager.num_required_blocks(seq)
        if seq.adapter_name is not None:
            adapter = ADAPTER_MANAGER.get_adapter(seq.adapter_name)
            num_required_blocks += self.block_manager.num_required_blocks(
                adapter)

        ignore_nodes = self.rtree_manager.get_all_nodes(seq)
        removed_nodes = []

        success = False
        for node in sorted_nodes:
            num_blocks = node.num_blocks
            self.block_manager.free(node.sequence, num_blocks)
            self.rtree_manager.remove_node(node)
            removed_nodes.append(node)
            num_required_blocks -= num_blocks
            if num_required_blocks <= 0:
                success = True
                break

        if success:
            removed_nodes += ignore_nodes
            for node in removed_nodes:
                try:
                    sorted_nodes.remove(node)
                except Exception:
                    continue

        return success
