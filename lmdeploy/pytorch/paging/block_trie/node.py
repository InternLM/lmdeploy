# Copyright (c) OpenMMLab. All rights reserved.
"""Trie node topology and checkpoint-path invalidation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lmdeploy.pytorch.messages import PrefixCacheExtraHashes

if TYPE_CHECKING:
    from .checkpoint import StateCheckpointMatchData


class Node:
    """One full-token-block edge in the prefix-cache trie.

    A non-root node owns one trie KV-block reference. ``hash_key``, ``tokens``,
    and ``extra_hashes`` define the block identity; ``extra_hashes`` carries
    VLM content identity for blocks that overlap multimodal spans.

    The same node may also own an optional SSM checkpoint. ``state_idx`` is the
    checkpoint slot, and ``state_ref_count`` pins the slot while an async
    restore may still read it or a producer save may still write it.
    ``state_ready`` together with non-``None`` ``state_match_data`` means the
    slot has been published and is matchable. A topology change invalidates
    ``state_match_data`` immediately; lookup later removes that temporarily
    stale ready/index entry.
    ``state_match_data`` caches the immutable host identity and logical KV path
    used to prove and apply an exact checkpoint hit without repeated Python
    block scans.

    ``parent`` is intentionally stateful: assigning it updates the old and new
    parent ``children`` maps and invalidates cached checkpoint paths in the
    moved subtree. Detached nodes can therefore still exist as stale
    auxiliary-index entries, but they are no longer trie truth.
    ``_topology_epoch`` changes during that invalidation;
    ``state_topology_epoch`` records the version captured by a checkpoint
    reservation so a path change before publication is also detectable.
    """

    def __init__(self,
                 hash_key: int,
                 block: int,
                 tokens: np.ndarray,
                 num_matched: int = 0,
                 extra_hashes: PrefixCacheExtraHashes = (),
                 state_idx: int = -1,
                 state_ready: bool = False,
                 state_ref_count: int = 0,
                 state_access_time: float = 0.0,
                 routed_experts: np.ndarray = None,
                 adapter_name: str = None):
        self.hash_key = hash_key
        self.block = block
        self.tokens = tokens
        self.num_matched = num_matched
        self.extra_hashes = extra_hashes
        self.state_idx = state_idx
        self.state_ready = state_ready
        self.state_ref_count = state_ref_count
        self.state_access_time = state_access_time
        self.state_match_data: StateCheckpointMatchData | None = None
        self._topology_epoch = 0
        self.state_topology_epoch = -1
        self.routed_experts = routed_experts
        self.adapter_name = adapter_name
        self.children: dict[int, Node] = {}
        self._parent: Node | None = None

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, val: Node | None):
        old_parent = self._parent
        if old_parent is val:
            return
        if old_parent is not None and old_parent.children.get(self.hash_key) is self:
            old_parent.children.pop(self.hash_key)
        if val is not None:
            displaced = val.children.get(self.hash_key)
            if displaced is not None and displaced is not self:
                displaced._parent = None
                displaced._invalidate_state_match_data()
            val.children[self.hash_key] = self
        self._parent = val
        self._invalidate_state_match_data()

    def _invalidate_state_match_data(self):
        """Invalidate checkpoint paths affected by moving this subtree."""
        pending = [self]
        while pending:
            node = pending.pop()
            node._topology_epoch += 1
            node.state_match_data = None
            pending.extend(node.children.values())

    def __lt__(self, other):
        return True

    def __le__(self, other):
        return True
