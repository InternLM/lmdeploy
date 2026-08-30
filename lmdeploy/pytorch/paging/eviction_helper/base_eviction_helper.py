# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING

from ...messages import SchedulerSequence

if TYPE_CHECKING:
    from ..block_manager import BaseBlockManager
    from ..block_trie import BlockTrie
    from ..kv_load_coordinator import KVLoadCoordinator
    from ..state_manager import StateManager

SeqList = list[SchedulerSequence]


class BaseEvictionHelper:
    """Base eviction helper."""

    def __init__(
        self,
        *,
        block_manager: BaseBlockManager,
        block_trie: BlockTrie,
        state_manager: StateManager,
        load_coordinator: KVLoadCoordinator,
    ) -> None:
        self.block_manager = block_manager
        self.block_trie = block_trie
        self.state_manager = state_manager
        self.load_coordinator = load_coordinator

    def need_swap_in(self, seq: SchedulerSequence):
        """Sequence need swap in."""
        raise NotImplementedError('Not implemented.')

    def evict_for_seq(self, seq: SchedulerSequence, evictable_seqs: list[SchedulerSequence], prealloc_size: int):
        """Evict seqs."""
        raise NotImplementedError('Not implemented.')
