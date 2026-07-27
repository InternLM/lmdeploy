# Copyright (c) OpenMMLab. All rights reserved.
"""Per-sequence state shared by PyTorch prefix-cache participants.

The records in this module describe protocol state exchanged by the scheduler, BlockTrie, input maker, engine loop, and
decoding strategies. They deliberately do not depend on any of those owners, keeping the sequence data model below the
prefix-cache implementation modules.
"""

from dataclasses import dataclass, field
from typing import Any, NamedTuple, TypeAlias

import numpy as np


class PrefixCacheMeta(NamedTuple):
    """Multimodal span identity used by prefix-cache block keys.

    Placeholder token ids alone are not enough for VLM prefix caching: two
    requests can contain the same image placeholder tokens backed by different
    image/video content. The trie key therefore includes every overlapping
    span's modality and stable content hash.
    """

    start: int
    end: int
    modality: str
    content_hash: str


# The block index stores the same immutable span objects as ``metas``. Keep the
# established aliases for callers that describe their role in trie keys.
PrefixCacheExtraHash: TypeAlias = PrefixCacheMeta
PrefixCacheExtraHashes: TypeAlias = tuple[PrefixCacheMeta, ...]
PrefixCacheBlockExtraHashes: TypeAlias = dict[int, PrefixCacheExtraHashes]


@dataclass
class StateCheckpointRestore:
    """A checkpoint selected to restore into this sequence's runtime state."""

    slot: int = -1
    node: Any = field(default=None, repr=False)
    pinned: bool = False

    @property
    def is_selected(self) -> bool:
        """Whether prefix matching selected a checkpoint."""
        return self.slot >= 0

    def select(self, slot: int, node: Any) -> None:
        """Remember the checkpoint selected by prefix matching."""
        self.slot = slot
        self.node = node
        self.pinned = False

    def clear(self) -> None:
        """Forget the selected checkpoint after release or rollback."""
        self.slot = -1
        self.node = None
        self.pinned = False


@dataclass
class StateCheckpointSaveReservation:
    """An unpublished checkpoint destination reserved before model forward."""

    slot: int = -1
    step: int = 0
    node: Any = field(default=None, repr=False)
    is_decode: bool = False

    @property
    def is_pending(self) -> bool:
        """Whether a save destination is currently reserved."""
        return self.slot >= 0

    def reserve(self, slot: int, step: int, node: Any, is_decode: bool) -> None:
        """Record the slot and trie identity that a forward will save."""
        self.slot = slot
        self.step = step
        self.node = node
        self.is_decode = is_decode

    def clear(self) -> None:
        """Forget a published or discarded save reservation."""
        self.slot = -1
        self.step = 0
        self.node = None
        self.is_decode = False


@dataclass
class StateCheckpointProducerPin:
    """A published checkpoint pinned until its producer forward completes."""

    slot: int = -1
    node: Any = field(default=None, repr=False)

    @property
    def is_acquired(self) -> bool:
        """Whether the producer currently holds a checkpoint pin."""
        return self.slot >= 0

    def acquire(self, slot: int, node: Any) -> None:
        """Remember a pin acquired after checkpoint publication."""
        self.slot = slot
        self.node = node

    def clear(self) -> None:
        """Forget a producer pin after releasing its node reference."""
        self.slot = -1
        self.node = None


@dataclass(slots=True)
class PrefixRecomputeOverlap:
    """Per-sequence state for deliberately recomputing a cached KV suffix.

    Some strategies need target hidden states from the end of an otherwise
    reusable prefix. ``required_blocks`` is that persistent strategy policy.
    A match records the cached-but-dropped suffix as a one-shot fresh block
    range; allocation must keep sequence-owned KV for these blocks instead of
    deduplicating it back to shared trie blocks.

    ``canonical_trie_blocks`` preserves the shared trie identity corresponding
    to those fresh blocks. It outlives the fresh range so a later SSM state
    checkpoint can snapshot the canonical KV path.
    """

    required_blocks: int = 0
    fresh_block_range: range | None = None
    canonical_trie_blocks: dict[int, int] = field(default_factory=dict, repr=False)

    def set_fresh_block_range(self, start_block: int, end_block: int) -> None:
        """Set the half-open block range that needs fresh KV allocation."""
        if end_block <= start_block:
            self.clear_fresh_block_range()
            return
        self.fresh_block_range = range(start_block, end_block)

    def clear_fresh_block_range(self) -> None:
        """Finish the one-shot match-to-allocation window."""
        self.fresh_block_range = None

    def requires_fresh_block(self, block_id: int) -> bool:
        """Whether this block must keep its sequence-owned writable KV."""
        return self.fresh_block_range is not None and block_id in self.fresh_block_range

    def remember_canonical_block(self, block_id: int, trie_block: int) -> None:
        """Associate fresh sequence KV with its canonical shared trie block."""
        self.canonical_trie_blocks[block_id] = trie_block

    def forget_canonical_block(self, block_id: int) -> None:
        """Forget a substitution once the sequence uses shared/canonical KV."""
        self.canonical_trie_blocks.pop(block_id, None)

    def rewrite_to_canonical_path(self, blocks: np.ndarray) -> None:
        """Rewrite sequence block ids to their canonical trie identities."""
        for block_id, trie_block in self.canonical_trie_blocks.items():
            if block_id < len(blocks):
                blocks[block_id] = trie_block

    def reset_runtime_state(self) -> None:
        """Clear transient overlap state while preserving strategy policy."""
        self.clear_fresh_block_range()
        self.canonical_trie_blocks.clear()


@dataclass
class PrefixCacheState:
    """Per-sequence prefix-cache bookkeeping.

    ``metas`` and ``block_extra_hashes`` are persistent request metadata used
    when constructing multimodal-aware trie keys. ``restore``,
    ``pending_save``, and ``producer_save_pin`` expose the three transient SSM
    checkpoint phases explicitly: a matched frozen state is pinned before
    forward, a save reservation is published after the model copies runtime
    state into it, and that published destination remains pinned until its
    producer forward completes. ``last_shared_node`` is the deepest trie node
    already shared by this sequence; ``BlockTrie.match()`` writes it and
    ``BlockTrie.allocate()`` continues inserting new full blocks from it.
    ``match_start_step`` remembers the sequence step before a tentative
    prefix-cache match so long-context chunking can distinguish current-turn
    cached multimodal spans from older session history. ``recompute_overlap``
    groups the strategy requirement, one-shot allocation window, and canonical
    trie identities used when a cached suffix must be recomputed into fresh
    sequence-owned KV. ``suppress_match_stats`` is set while replaying work
    after recompute eviction; cache reuse may still happen, but it should not
    affect the public prefix-cache hit-rate metric.
    """

    # Persistent request metadata used to build multimodal-aware trie keys.
    metas: list[PrefixCacheMeta] = field(default_factory=list)
    block_extra_hashes: PrefixCacheBlockExtraHashes = field(default_factory=dict, repr=False)
    num_indexed_metas: int = 0

    # Trie cursor for the deepest prefix block already shared by this sequence.
    last_shared_node: Any = field(default=None, repr=False)

    # SSM checkpoint state grouped by its distinct lifecycle phase.
    restore: StateCheckpointRestore = field(default_factory=StateCheckpointRestore)
    pending_save: StateCheckpointSaveReservation = field(default_factory=StateCheckpointSaveReservation)
    producer_save_pin: StateCheckpointProducerPin = field(default_factory=StateCheckpointProducerPin)

    # Latest decode checkpoint node owned by this sequence.
    decode_checkpoint_node: Any = field(default=None, repr=False)

    # Tentative match state used for chunking, recompute overlap, and metrics.
    match_start_step: int = -1
    recompute_overlap: PrefixRecomputeOverlap = field(default_factory=PrefixRecomputeOverlap)
    suppress_match_stats: bool = False
