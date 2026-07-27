# Copyright (c) OpenMMLab. All rights reserved.
"""Monotonic prefix-cache trie node topology."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from lmdeploy.pytorch.prefix_cache_state import PrefixCacheExtraHashes

if TYPE_CHECKING:
    from .checkpoint import StateCheckpointMatchData


@dataclass(slots=True)
class NodeStateCheckpoint:
    """State-checkpoint metadata allocated only for checkpoint owners.

    ``slot`` describes the reservation. ``published`` exposes it to exact
    matching, while ``exact_match_data`` caches the immutable prefix identity
    and logical KV path used after sparse lookup. ``pin_count`` protects async
    save/restore copies, and ``last_access_time`` drives state-only LRU
    eviction.
    """

    slot: int
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

    Trie topology is monotonic: a fresh node attaches once, and eviction may
    detach only a leaf. Disallowing subtree moves keeps every attached node's
    ancestor path stable and removes the need for topology-version bookkeeping.
    Detached nodes may still remain temporarily in auxiliary indexes, but they
    are no longer trie truth.
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
        self.routed_experts = routed_experts
        self.adapter_name = adapter_name
        self.children: dict[int, Node] = {}
        self._parent: Node | None = None

    @property
    def parent(self):
        return self._parent

    def attach_to(self, parent: Node):
        """Attach a fresh node without replacing an existing trie edge."""
        if self._parent is not None:
            raise RuntimeError('Cannot reattach a prefix-cache trie node.')
        if self.hash_key in parent.children:
            raise RuntimeError('Cannot replace an existing prefix-cache trie child.')
        parent.children[self.hash_key] = self
        self._parent = parent

    def detach_leaf(self):
        """Detach an evicted leaf while preserving monotonic ancestry."""
        if len(self.children) > 0:
            raise RuntimeError('Cannot detach a non-leaf prefix-cache trie node.')
        parent = self._parent
        if parent is None:
            return False
        if parent.children.get(self.hash_key) is not self:
            raise RuntimeError('Cannot detach an inconsistent prefix-cache trie edge.')
        parent.children.pop(self.hash_key)
        self._parent = None
        return True

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
