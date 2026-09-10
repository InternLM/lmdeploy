# Copyright (c) OpenMMLab. All rights reserved.

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

    def _evict_for_seq_default(self, seq: SchedulerSequence, evictable_seqs: list[SchedulerSequence],
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

            # A completed remote load has published fresh KV into these blocks.
            # Preserve it until prefill consumes the result instead of paying
            # for the transfer again on a later scheduling turn.
            if self.scheduler.kv_load_coordinator.is_remote_ready(evict_seq):
                continue

            # skip sequence with no blocks
            if evict_seq.num_blocks == 0:
                continue

            if block_trie.enabled:
                evict_seq.prefix_cache.suppress_match_stats = True
            # Eviction also ends the tracked prefill; otherwise its soft block
            # reservation would outlive the local KV blocks freed below.
            self.scheduler.kv_load_coordinator.release(evict_seq)
            evict_seq.state.free()
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

    def _evict_for_ssm(self, seq: SchedulerSequence, evictable_seqs: list[SchedulerSequence], prealloc_size: int):
        """Evict blocks and checkpoint states for an SSM sequence.

        SSM scheduling needs both KV blocks and a runtime state slot.  Before evicting live sequences, try dropping old
        unpinned checkpoints because they are cheaper to recompute than an active request.
        """
        block_manager = self.block_manager
        state_manager = self.state_manager
        block_trie = self.block_trie
        num_required_blocks = block_manager.num_required_blocks(seq, prealloc_size)
        # avoid requiring free state when already allocated.
        has_runtime_state = state_manager.is_allocated(seq)
        has_free_state = has_runtime_state or state_manager.get_num_free_runtime() > 0
        if not has_free_state:
            block_trie.state_checkpoints.evict(1)
            has_free_state = state_manager.get_num_free_runtime() > 0

        if has_free_state and block_manager.get_num_free_gpu_blocks() >= num_required_blocks:
            return True

        success = False
        while len(evictable_seqs) > 0:
            evict_seq = evictable_seqs.pop(0)

            # READY remote KV is already transferred and awaiting consumption;
            # do not discard that result merely to admit another prefill.
            if self.scheduler.kv_load_coordinator.is_remote_ready(evict_seq):
                continue

            # skip sequence with no blocks
            if evict_seq.num_blocks == 0 and evict_seq.logical_state < 0:
                continue

            # free sequence
            if block_trie.enabled:
                evict_seq.prefix_cache.suppress_match_stats = True
            # Keep coordinator ownership and its soft admission budget in sync
            # with the KV blocks and SSM runtime state released by free().
            self.scheduler.kv_load_coordinator.release(evict_seq)
            evict_seq.state.free()
            has_free_state = has_runtime_state or state_manager.get_num_free_runtime() > 0
            if not has_free_state:
                block_trie.state_checkpoints.evict(1)
                has_free_state = state_manager.get_num_free_runtime() > 0
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
