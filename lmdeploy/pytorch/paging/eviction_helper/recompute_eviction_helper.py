# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from ...messages import SchedulerSequence

if TYPE_CHECKING:
    from ..block_manager import BaseBlockManager
    from ..block_trie import BlockTrie
    from ..kv_load_coordinator import KVLoadCoordinator
    from ..state_manager import StateManager


class RecomputeEvictionHelper:
    """Reclaim paging resources so a sequence can be recomputed."""

    def __init__(
        self,
        *,
        block_manager: BaseBlockManager,
        block_trie: BlockTrie,
        state_manager: StateManager,
        load_coordinator: KVLoadCoordinator,
        is_ssm: bool,
    ) -> None:
        self.block_manager = block_manager
        self.block_trie = block_trie
        self.state_manager = state_manager
        self.load_coordinator = load_coordinator
        self._is_ssm = is_ssm

    def try_make_capacity_for(
        self,
        seq: SchedulerSequence,
        evictable_seqs: Iterable[SchedulerSequence],
        prealloc_size: int,
    ) -> bool:
        """Try to reclaim enough paging capacity for one sequence."""
        if self._is_ssm:
            return self._try_make_ssm_capacity(
                seq,
                evictable_seqs,
                prealloc_size,
            )

        block_manager = self.block_manager
        num_required_blocks = block_manager.num_required_blocks(
            seq,
            prealloc_size,
        )
        if block_manager.get_num_free_gpu_blocks() >= num_required_blocks:
            return True

        for evict_seq in evictable_seqs:
            if not self._reclaim_candidate(evict_seq):
                continue
            if self._try_make_block_capacity(num_required_blocks):
                return True

        return self._try_make_block_capacity(num_required_blocks)

    def _try_make_ssm_capacity(
        self,
        seq: SchedulerSequence,
        evictable_seqs: Iterable[SchedulerSequence],
        prealloc_size: int,
    ) -> bool:
        """Try to make both KV-block and runtime-state capacity available."""
        block_manager = self.block_manager
        state_manager = self.state_manager
        state_checkpoints = self.block_trie.state_checkpoints
        num_required_blocks = block_manager.num_required_blocks(
            seq,
            prealloc_size,
        )

        # A running long prefill can reuse its already allocated state.
        has_runtime_state = state_manager.is_allocated(seq)
        has_free_state = (
            has_runtime_state
            or state_checkpoints.make_runtime_state_available()
        )
        if (has_free_state
                and block_manager.get_num_free_gpu_blocks()
                >= num_required_blocks):
            return True

        for evict_seq in evictable_seqs:
            if not self._reclaim_candidate(evict_seq):
                continue
            has_free_state = (
                has_runtime_state
                or state_checkpoints.make_runtime_state_available()
            )
            if self._try_make_block_capacity(num_required_blocks):
                return has_free_state

        if not has_free_state:
            return False
        return self._try_make_block_capacity(num_required_blocks)

    def _reclaim_candidate(self, seq: SchedulerSequence) -> bool:
        """Release one eligible candidate's paging ownership."""
        # Completed remote KV is already transferred and awaits consumption.
        if self.load_coordinator.is_remote_ready(seq):
            return False
        if (seq.num_blocks == 0
                and (not self._is_ssm or seq.logical_state < 0)):
            return False

        if self.block_trie.enabled:
            seq.prefix_cache.suppress_match_stats = True
        # Keep soft admission accounting aligned with released resources.
        self.load_coordinator.release_tracking(seq)
        seq.state.release_paging_resources()
        return True

    def _try_make_block_capacity(self, num_required_blocks: int) -> bool:
        """Evict cached trie blocks until the required capacity is free."""
        block_manager = self.block_manager
        num_missing_blocks = (
            num_required_blocks - block_manager.get_num_free_gpu_blocks()
        )
        if num_missing_blocks > 0:
            self.block_trie.evict(num_missing_blocks)
        return (
            num_required_blocks <= block_manager.get_num_free_gpu_blocks()
        )
