# Copyright (c) OpenMMLab. All rights reserved.
"""KV-block references and leaf eviction for the prefix-cache trie."""

from __future__ import annotations

import heapq
import logging
from typing import TYPE_CHECKING

import numpy as np

from lmdeploy.utils import get_logger

from .node import Node

if TYPE_CHECKING:
    from ..block_manager.base_block_manager import LogicalAllocator
    from .checkpoint_lifecycle import StateCheckpointLifecycle

logger = get_logger('lmdeploy')


class KVBlockLifecycle:
    """Own trie KV references and the auxiliary leaf-candidate index.

    ``BlockTrie`` decides token identity, collision deduplication, and private
    recompute policy. This component applies the resulting reference-count
    transaction and owns eviction. KV eviction depends on the checkpoint
    lifecycle because a pinned state checkpoint prevents its node from being
    detached, while an unpinned checkpoint must be released first.
    """

    def __init__(self, allocator: LogicalAllocator, state_checkpoints: StateCheckpointLifecycle):
        self.allocator = allocator
        self.state_checkpoints = state_checkpoints
        self.leaves: set[Node] = set()

    def begin_path_extension(self, node: Node):
        """Remove a leaf that is about to gain a child."""
        if node.parent is not None and len(node.children) == 0:
            self.leaves.discard(node)

    def commit_path_extension(self,
                              node: Node,
                              additional_ref_blocks: list[int],
                              duplicate_blocks: list[int]):
        """Commit leaf bookkeeping and batched allocator ref changes.

        Each block in ``additional_ref_blocks`` needs one new owner: either the
        trie ref for a fresh node or the sequence ref after collision
        deduplication selected an existing trie block. ``duplicate_blocks`` are
        the fresh sequence blocks replaced by that deduplication.
        """
        if node.parent is not None and len(node.children) == 0:
            self.leaves.add(node)
        if len(additional_ref_blocks) > 0:
            self.allocator.add_ref_count(np.array(additional_ref_blocks), 1)
        if len(duplicate_blocks) > 0:
            self.allocator.free(np.array(duplicate_blocks))

    @classmethod
    def _is_attached_leaf(cls, node: Node):
        return node.is_attached() and len(node.children) == 0

    @classmethod
    def _is_evict_candidate_leaf(cls, node: Node):
        """Allow stale detached leaves to be pruned from the candidate set."""
        return (node.block >= 0 and len(node.children) == 0
                and (node.parent is None or node.is_attached()))

    def _remove_leaf(self,
                     leaves: list[tuple[float, int, Node]],
                     evicted_blocks: list[int]) -> tuple[bool, Node | None]:
        while len(leaves) > 0:
            _, _, leaf = heapq.heappop(leaves)
            if leaf not in self.leaves:
                continue
            if not self._is_evict_candidate_leaf(leaf):
                self.leaves.discard(leaf)
                continue
            if self.state_checkpoints.is_pinned(leaf):
                continue
            if int(self.allocator.get_ref_count(leaf.block)) != 1:
                continue
            break
        else:
            return False, None

        evicted_blocks.append(leaf.block)
        self.state_checkpoints.release(leaf)
        parent = leaf.parent
        if parent is not None:
            leaf.parent = None
        self.leaves.discard(leaf)
        return True, parent

    def _add_leaf(self, leaves: list[tuple[float, int, Node]], parent: Node):
        if not self._is_attached_leaf(parent) or parent in self.leaves:
            return
        self.leaves.add(parent)
        if self.allocator.get_ref_count(parent.block) == 1:
            access_time = self.allocator.get_access_time(parent.block)
            heapq.heappush(leaves, (access_time, id(parent), parent))

    def evict(self, max_num_blocks: int):
        """Evict least-recently-used trie-owned KV leaf blocks."""
        if len(self.leaves) == 0:
            return 0

        old_leaf_count = len(self.leaves)
        candidates = [leaf for leaf in self.leaves if self._is_evict_candidate_leaf(leaf)]
        if len(candidates) != old_leaf_count:
            self.leaves.intersection_update(candidates)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'Dropped stale prefix-cache leaf candidates before eviction: '
                             f'old_count={old_leaf_count} new_count={len(candidates)}')
        if len(candidates) == 0:
            return 0

        # A ref count of one means only the trie owns the block.
        candidate_blocks = np.array([leaf.block for leaf in candidates])
        ref_counts = self.allocator.get_ref_count(candidate_blocks)
        evictable_indices = (ref_counts == 1).nonzero()[0]
        if len(evictable_indices) == 0:
            return 0

        access_times = self.allocator.get_access_time(candidate_blocks)
        leaves = [(access_times[index], id(candidates[index]), candidates[index]) for index in evictable_indices]
        heapq.heapify(leaves)

        evicted_blocks: list[int] = []
        while len(leaves) > 0 and len(evicted_blocks) < max_num_blocks:
            removed, parent = self._remove_leaf(leaves, evicted_blocks)
            if not removed:
                break
            if parent is None or parent.parent is None:
                # Ignore the adapter root.
                continue
            if len(parent.children) == 0:
                self._add_leaf(leaves, parent)

        if len(evicted_blocks) == 0:
            return 0
        self.allocator.free(np.array(evicted_blocks))
        return len(evicted_blocks)
