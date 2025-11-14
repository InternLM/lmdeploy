# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from ...messages import SchedulerSequence
from ..scheduler import Scheduler
from .base_eviction_helper import BaseEvictionHelper


class RecomputeEvictionHelper(BaseEvictionHelper):
    """Recompute eviction."""

    def __init__(self, scheduler: Scheduler):
        super().__init__(scheduler)

        if len(self.cache_config.states_shapes) == 0:
            self.evict_for_seq = self._evict_for_seq_default
        else:
            self.evict_for_seq = self._evict_for_ssm

    def _evict_for_seq_default(self, seq: SchedulerSequence, evictable_seqs: List[SchedulerSequence],
                               prealloc_size: int):
        """Evict seqs."""
        block_manager = self.block_manager
        block_trie = self.block_trie
        num_required_blocks = block_manager.num_required_blocks(seq, prealloc_size)

        if block_manager.get_num_free_gpu_blocks() >= num_required_blocks:
            return True

        success = False
        while len(evictable_seqs) > 0:
            evict_seq = evictable_seqs.pop(0)

            # skip sequence with no blocks
            if evict_seq.num_blocks == 0:
                continue

            block_manager.free(evict_seq)
            evict_seq.set_step(0)
            num_req = (num_required_blocks - block_manager.get_num_free_gpu_blocks())
            if num_req <= 0:
                success = True
                break

            block_trie.evict(num_req)
            num_req = (num_required_blocks - block_manager.get_num_free_gpu_blocks())
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

    def _evict_for_ssm(self, seq: SchedulerSequence, evictable_seqs: List[SchedulerSequence], prealloc_size: int):
        """Evict seqs."""
        block_manager = self.block_manager
        state_manager = self.state_manager
        block_trie = self.block_trie
        num_required_blocks = block_manager.num_required_blocks(seq, prealloc_size)
        has_free_state = state_manager.get_num_free() > 0

        if has_free_state and block_manager.get_num_free_gpu_blocks() >= num_required_blocks:
            return True

        success = False
        while len(evictable_seqs) > 0:
            evict_seq = evictable_seqs.pop(0)

            # skip sequence with no blocks
            if evict_seq.num_blocks == 0 and evict_seq.logical_state < 0:
                continue

            # free sequence
            block_manager.free(evict_seq)
            evict_seq.set_step(0)
            state_manager.free(evict_seq)
            has_free_state = True
            num_req = (num_required_blocks - block_manager.get_num_free_gpu_blocks())
            if num_req <= 0:
                success = True
                break

            # clear cached prefix
            block_trie.evict(num_req)
            num_req = (num_required_blocks - block_manager.get_num_free_gpu_blocks())
            if num_req <= 0:
                success = True
                break

        if not has_free_state:
            return False

        # for empty evictable_seqs case
        num_req = num_required_blocks - block_manager.get_num_free_gpu_blocks()
        if num_req > 0:
            block_trie.evict(num_req)
            if num_required_blocks <= block_manager.get_num_free_gpu_blocks():
                success = True

        return success
