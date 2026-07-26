# Copyright (c) OpenMMLab. All rights reserved.
"""Trie node topology and checkpoint-path invalidation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from lmdeploy.pytorch.messages import PrefixCacheExtraHashes

if TYPE_CHECKING:
    from .checkpoint import StateCheckpointMatchData


@dataclass(slots=True)
class NodeStateCheckpoint:
    """State-checkpoint metadata allocated only for checkpoint owners.

    ``slot`` and ``reserved_topology_epoch`` describe the reservation.
    ``published`` exposes it to exact matching, while ``exact_match_data``
    caches the immutable prefix identity and logical KV path used after sparse
    lookup. ``pin_count`` protects async save/restore copies, and
    ``last_access_time`` drives state-only LRU eviction.
    """

    slot: int
    reserved_topology_epoch: int = -1
    published: bool = False
    exact_match_data: StateCheckpointMatchData | None = None
    pin_count: int = 0
    last_access_time: float = 0.0


class Node:
    """One full-token-block edge in the prefix-cache trie.

    A non-root node owns one trie KV-block reference. ``hash_key``, ``tokens``,
    and ``extra_hashes`` define the block identity; ``extra_hashes`` carries
    VLM content identity for blocks that overlap multimodal spans.

    ``state_checkpoint`` is allocated lazily when this node reserves an SSM
    state slot. It groups state ownership, publication, pinning, and cached
    exact-match data so ordinary KV-only nodes do not carry those lifecycle
    fields individually.

    ``parent`` is intentionally stateful: assigning it updates the old and new
    parent ``children`` maps and invalidates cached checkpoint paths in the
    moved subtree. Detached nodes can therefore still exist as stale
    auxiliary-index entries, but they are no longer trie truth.
    ``_topology_epoch`` changes during that invalidation. A checkpoint records
    the version captured by its reservation so a path change before
    publication is also detectable.
    """

    def __init__(self,
                 hash_key: int,
                 block: int,
                 tokens: np.ndarray,
                 num_matched: int = 0,
                 extra_hashes: PrefixCacheExtraHashes = (),
                 routed_experts: np.ndarray = None,
                 adapter_name: str = None):
        self.hash_key = hash_key
        self.block = block
        self.tokens = tokens
        self.num_matched = num_matched
        self.extra_hashes = extra_hashes
        self.state_checkpoint: NodeStateCheckpoint | None = None
        self._topology_epoch = 0
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
                displaced._invalidate_checkpoint_paths()
            val.children[self.hash_key] = self
        self._parent = val
        self._invalidate_checkpoint_paths()

    def is_attached(self):
        """Check whether this node is still linked from its parent."""
        parent = self.parent
        return parent is not None and parent.children.get(self.hash_key) is self

    def path_from_root(self):
        """Return non-root nodes from the adapter root to this node."""
        nodes = []
        node = self
        while node is not None and node.parent is not None:
            nodes.append(node)
            node = node.parent
        nodes.reverse()
        return nodes

    def _invalidate_checkpoint_paths(self):
        """Invalidate checkpoint paths affected by moving this subtree."""
        pending = [self]
        while pending:
            node = pending.pop()
            node._topology_epoch += 1
            checkpoint = node.state_checkpoint
            if checkpoint is not None:
                checkpoint.exact_match_data = None
            pending.extend(node.children.values())
