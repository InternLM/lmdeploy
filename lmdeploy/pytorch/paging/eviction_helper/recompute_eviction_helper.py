# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from ...messages import SchedulerSequence
from .base_eviction_helper import BaseEvictionHelper


class RecomputeEvictionHelper(BaseEvictionHelper):
    """recompute eviction."""

    def evict_for_seq(self, seq: SchedulerSequence,
                      evictable_seqs: List[SchedulerSequence],
                      prealloc_size: int):
        """evict seqs."""
        block_manager = self.block_manager
        block_trie = self.block_trie
        num_required_blocks = block_manager.num_required_blocks(
            seq, prealloc_size)

        if block_manager.get_num_free_gpu_blocks() >= num_required_blocks:
            return True

        success = False
        while len(evictable_seqs) > 0:
            evict_seq = evictable_seqs.pop(0)

            block_trie.free(evict_seq)
            num_req = (num_required_blocks -
                       block_manager.get_num_free_gpu_blocks())
            if num_req <= 0:
                success = True
                break

            block_trie.evict(num_req)
            num_req = (num_required_blocks -
                       block_manager.get_num_free_gpu_blocks())
            if num_req <= 0:
                success = True
                break

        # for empty evictable_seqs case
        num_req = num_required_blocks - block_manager.get_num_free_gpu_blocks()
        if num_req > 0:
            block_trie.evict(num_req)
            if num_required_blocks <= block_manager.get_num_free_gpu_blocks():
                success = True

        return success
