# Copyright (c) OpenMMLab. All rights reserved.
"""KV-block references and leaf eviction for the prefix-cache trie."""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING

import numpy as np

from lmdeploy.utils import get_logger

from .node import Node

if TYPE_CHECKING:
    from ..block_manager.base_block_manager import LogicalAllocator
    from ..block_manager.group_allocator import GroupAllocator
    from .checkpoint_lifecycle import StateCheckpointLifecycle

logger = get_logger('lmdeploy')


class KVBlockLifecycle:
    """Own trie KV references and the auxiliary leaf-candidate index.

    ``BlockTrie`` decides token identity, collision deduplication, and recompute
    overlap policy. This component applies the resulting reference-count
    transaction and owns eviction. KV eviction depends on the checkpoint
    lifecycle because a pinned state checkpoint prevents its node from being
    detached, while an unpinned checkpoint must be released first.
    """

    def __init__(self,
                 allocator: LogicalAllocator,
                 state_checkpoints: StateCheckpointLifecycle,
                 group_allocator: GroupAllocator | None = None):
        self.allocator = allocator
        self.group_allocator = group_allocator or getattr(allocator, 'group_allocator', None)
        self.state_checkpoints = state_checkpoints
        self.leaves: set[Node] = set()

    def begin_path_extension(self, node: Node):
        """Remove a leaf that is about to gain a child."""
        if node.parent is not None and len(node.children) == 0:
            self.leaves.discard(node)

    def commit_path_extension(self,
                              node: Node,
                              ref_blocks: list[int],
                              free_blocks: list[int]):
        """Commit leaf bookkeeping and batched allocator ref changes.

        Each block in ``ref_blocks`` needs one new owner: either the
        trie ref for a fresh node or the sequence ref after collision
        deduplication selected an existing trie block. ``free_blocks`` are
        the fresh sequence blocks replaced by that deduplication.
        """
        if node.parent is not None and len(node.children) == 0:
            self.leaves.add(node)
        if len(ref_blocks) > 0:
            self.allocator.add_ref_count(np.array(ref_blocks), 1)
        if len(free_blocks) > 0:
            self.allocator.free(np.array(free_blocks))

    @classmethod
    def _is_attached_leaf(cls, node: Node):
        return node.is_attached() and len(node.children) == 0

    @classmethod
    def _is_leaf_eviction_candidate(cls, node: Node):
        """Allow stale detached leaves to be pruned from the candidate set."""
        return (node.block_id >= 0 and len(node.children) == 0
                and (node.parent is None or node.is_attached()))

    def _try_evict_leaf(self,
                        candidate_heap: list[tuple[float, int, Node]],
                        evicted_blocks: list[int]) -> tuple[bool, Node | None]:
        while len(candidate_heap) > 0:
            _, _, leaf = heapq.heappop(candidate_heap)
            if leaf not in self.leaves:
                continue
            if not self._is_leaf_eviction_candidate(leaf):
                self.leaves.discard(leaf)
                continue
            if self.state_checkpoints.is_pinned(leaf):
                continue
            if int(self.allocator.get_ref_count(leaf.block_id)) != 1:
                continue
            break
        else:
            return False, None

        evicted_blocks.append(leaf.block_id)
        self.state_checkpoints.release_checkpoint(leaf)
        parent = leaf.parent
        if parent is not None:
            leaf.detach_leaf()
        self.leaves.discard(leaf)
        return True, parent

    def _add_parent_leaf_candidate(self, candidate_heap: list[tuple[float, int, Node]], parent: Node):
        if not self._is_attached_leaf(parent) or parent in self.leaves:
            return
        self.leaves.add(parent)
        if self.allocator.get_ref_count(parent.block_id) == 1:
            access_time = self.allocator.get_access_time(parent.block_id)
            heapq.heappush(candidate_heap, (access_time, id(parent), parent))

    def evict(self, max_num_blocks: int):
        """Evict least-recently-used trie-owned KV leaf blocks."""
        if len(self.leaves) == 0:
            return 0

        old_leaf_count = len(self.leaves)
        candidates = [leaf for leaf in self.leaves if self._is_leaf_eviction_candidate(leaf)]
        if len(candidates) != old_leaf_count:
            self.leaves.intersection_update(candidates)
            logger.debug('Dropped stale prefix-cache leaf candidates before eviction: old_count=%s new_count=%s',
                         old_leaf_count, len(candidates))
        if len(candidates) == 0:
            return 0

        # A ref count of one means only the trie owns the block.
        candidate_blocks = np.array([leaf.block_id for leaf in candidates])
        ref_counts = self.allocator.get_ref_count(candidate_blocks)
        evictable_indices = (ref_counts == 1).nonzero()[0]
        if len(evictable_indices) == 0:
            return 0

        access_times = self.allocator.get_access_time(candidate_blocks)
        candidate_heap = [(access_times[index], id(candidates[index]), candidates[index])
                          for index in evictable_indices]
        heapq.heapify(candidate_heap)

        evicted_blocks: list[int] = []
        while len(candidate_heap) > 0 and len(evicted_blocks) < max_num_blocks:
            removed, parent = self._try_evict_leaf(candidate_heap, evicted_blocks)
            if not removed:
                break
            if parent is None or parent.parent is None:
                # Ignore the adapter root.
                continue
            if len(parent.children) == 0:
                self._add_parent_leaf_candidate(candidate_heap, parent)

        if len(evicted_blocks) == 0:
            return 0
        self.allocator.free(np.array(evicted_blocks))
        return len(evicted_blocks)

    def evict_one_kv_group(self) -> int:
        """Evict one complete shared KV group when it is independently safe.

        The proof is intentionally conservative: every live physical slot in
        the candidate group must be represented by an attached trie node with
        trie-only ownership, and all descendants of those nodes must stay in
        the same group. This keeps group release atomic without adding an
        allocator-side reverse index.
        """
        if self.group_allocator is None:
            return 0

        nodes_by_block: dict[int, Node] = {}
        visited_nodes: set[Node] = set()

        def visit(node: Node):
            if node in visited_nodes:
                return
            visited_nodes.add(node)
            for child in node.children.values():
                nodes_by_block[child.block_id] = child
                visit(child)

        # Discover roots from the leaf paths because this lifecycle does not
        # own BlockTrie's adapter-root table.
        roots: set[Node] = set()
        for leaf in tuple(self.leaves):
            node = leaf
            while node.parent is not None:
                node = node.parent
            roots.add(node)
        for root in roots:
            visit(root)

        best_candidate = None
        blocks_by_group = self.allocator.live_logical_blocks_by_group()
        for group_id, block_ids in blocks_by_group.items():
            if self.group_allocator.group_role(group_id) != 'kv':
                continue
            if np.any(self.allocator.get_ref_count(block_ids) != 1):
                continue
            group_nodes = [nodes_by_block.get(int(block_id)) for block_id in block_ids]
            if any(node is None or not node.is_attached() for node in group_nodes):
                continue
            if any(self.state_checkpoints.is_pinned(node) for node in group_nodes):
                continue
            group_nodes_set = set(group_nodes)
            if any(child not in group_nodes_set for node in group_nodes for child in node.children.values()):
                continue
            access_time = float(self.allocator.get_access_time(block_ids).max())
            candidate = (access_time, group_id, block_ids, group_nodes)
            if best_candidate is None or access_time < best_candidate[0]:
                best_candidate = candidate

        if best_candidate is None:
            return 0
        _, _, block_ids, group_nodes = best_candidate
        for node in sorted(group_nodes, key=lambda item: item.prefix_len, reverse=True):
            self.state_checkpoints.release_checkpoint(node)
            self.leaves.discard(node)
            if node.children:
                return 0
            node.detach_leaf()
        self.allocator.free(block_ids)
        return len(block_ids)
