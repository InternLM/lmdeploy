# Copyright (c) OpenMMLab. All rights reserved.
"""Monotonic prefix-cache trie node topology."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from lmdeploy.pytorch.prefix_cache_state import PrefixCacheExtraIdentity

if TYPE_CHECKING:
    from .checkpoint import StateCheckpointMatchData


@dataclass(slots=True)
class NodeStateCheckpoint:
    """State and optional partial-KV checkpoint owned by one trie anchor.

    ``step`` is the exact model-forward boundary. ``frozen_block_id`` owns a
    copy of the partial logical block when ``step`` is not block-aligned.
    ``published`` exposes the checkpoint to exact matching, while
    ``exact_match_data`` caches the immutable prefix identity and logical KV
    path used after sparse lookup. ``pin_count`` protects async save/restore
    copies, and ``last_access_time`` drives checkpoint LRU eviction.
    """

    slot: int
    step: int
    frozen_block_id: int = -1
    published: bool = False
    exact_match_data: StateCheckpointMatchData | None = None
    pin_count: int = 0
    last_access_time: float = 0.0


class Node:
    """One full-token-block edge in the prefix-cache trie.

    A non-root node owns one trie KV-block reference. ``block_hash``,
    ``token_ids``, and ``extra_identity`` define the block identity;
    ``extra_identity`` carries VLM content identity for blocks that overlap
    multimodal spans.

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
                 block_hash: int,
                 block_id: int,
                 token_ids: np.ndarray,
                 prefix_len: int = 0,
                 extra_identity: PrefixCacheExtraIdentity = (),
                 routed_experts: np.ndarray = None,
                 adapter_name: str = None):
        self.block_hash = block_hash
        self.block_id = block_id
        self.token_ids = token_ids
        self.prefix_len = prefix_len
        self.extra_identity = extra_identity
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
        if self.block_hash in parent.children:
            raise RuntimeError('Cannot replace an existing prefix-cache trie child.')
        parent.children[self.block_hash] = self
        self._parent = parent

    def detach_leaf(self):
        """Detach an evicted leaf while preserving monotonic ancestry."""
        if len(self.children) > 0:
            raise RuntimeError('Cannot detach a non-leaf prefix-cache trie node.')
        parent = self._parent
        if parent is None:
            return False
        if parent.children.get(self.block_hash) is not self:
            raise RuntimeError('Cannot detach an inconsistent prefix-cache trie edge.')
        parent.children.pop(self.block_hash)
        self._parent = None
        return True

    def is_attached(self):
        """Check whether this node is still linked from its parent."""
        parent = self.parent
        return parent is not None and parent.children.get(self.block_hash) is self

    def is_attached_or_root(self):
        """Check whether this node is attached or is an adapter root."""
        if self.parent is None:
            return self.block_id < 0
        return self.is_attached()

    def path_from_root(self):
        """Return non-root nodes from the adapter root to this node."""
        nodes = []
        node = self
        while node is not None and node.parent is not None:
            nodes.append(node)
            node = node.parent
        nodes.reverse()
        return nodes
