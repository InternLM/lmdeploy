# Copyright (c) OpenMMLab. All rights reserved.
"""KV-block references and leaf eviction for the prefix-cache trie."""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from lmdeploy.utils import get_logger

from .node import Node

if TYPE_CHECKING:
    from ..block_manager.base_block_manager import LogicalAllocator
    from ..block_manager.group_allocator import GroupAllocator
    from .checkpoint_lifecycle import StateCheckpointLifecycle

logger = get_logger('lmdeploy')


class _GroupEvictionCandidate(NamedTuple):
    """Validated state for evicting one complete shared KV group."""

    access_time: float
    group_id: int
    block_ids: np.ndarray
    group_nodes: list[Node]


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

    def _make_group_eviction_candidate(self, leaf: Node) -> _GroupEvictionCandidate | None:
        """Build an independently removable group ending at ``leaf``."""
        if not self._is_leaf_eviction_candidate(leaf):
            self.leaves.discard(leaf)
            return None

        group_size = self.group_allocator.group_size
        leaf_offset = self.allocator.get_physical_block(leaf.block_id)
        group_id = leaf_offset // group_size
        # KV lifecycle leaves should only reference KV-owned groups. Keep this
        # boundary guard for stale or corrupted auxiliary entries.
        if self.group_allocator.group_role(group_id) != 'kv':
            self.leaves.discard(leaf)
            return None

        block_ids = self.allocator.logical_blocks_for_group(group_id)
        if len(block_ids) == 0 or np.any(self.allocator.get_ref_count(block_ids) != 1):
            return None

        group_nodes = []
        node = leaf
        while node.parent is not None:
            node_offset = self.allocator.get_physical_block(node.block_id)
            if node_offset // group_size != group_id:
                break
            group_nodes.append(node)
            node = node.parent

        if len(group_nodes) != len(block_ids):
            return None
        if set(item.block_id for item in group_nodes) != set(int(block_id) for block_id in block_ids):
            return None
        group_nodes_set = set(group_nodes)
        if any(self.state_checkpoints.is_pinned(item) for item in group_nodes):
            return None
        if any(child not in group_nodes_set for item in group_nodes for child in item.children.values()):
            return None

        access_time = float(self.allocator.get_access_time(block_ids).max())
        return _GroupEvictionCandidate(access_time, group_id, block_ids, group_nodes)

    def evict_kv_groups(self, max_num_groups: int) -> int:
        """Evict up to ``max_num_groups`` complete shared KV groups.

        Returns the number of logical KV blocks released, matching the block-based eviction API used by the scheduler.
        """
        if self.group_allocator is None or max_num_groups <= 0:
            return 0

        leaves = [leaf for leaf in self.leaves if self._is_attached_leaf(leaf)]
        if len(leaves) != len(self.leaves):
            self.leaves.intersection_update(leaves)
        if not leaves:
            return 0

        leaf_blocks = np.fromiter((leaf.block_id for leaf in leaves), dtype=np.int64)
        group_ids = self.allocator.get_physical_blocks(leaf_blocks) // self.group_allocator.group_size
        # Shared groups are sequence-private, so all blocks in one group form
        # one linear trie path and one leaf is sufficient to validate it.
        group_ids, leaf_indices = np.unique(group_ids, return_index=True)
        leaves = [leaves[int(index)] for index in leaf_indices]
        access_times = self.allocator.get_group_access_times(group_ids)
        candidate_heap = [(float(access_time), int(group_id), leaf)
                          for access_time, group_id, leaf in zip(access_times, group_ids, leaves)]
        heapq.heapify(candidate_heap)
        candidate_groups = set(int(group_id) for group_id in group_ids)

        evicted_blocks = 0
        evicted_groups = 0
        while candidate_heap and evicted_groups < max_num_groups:
            _, group_id, leaf = heapq.heappop(candidate_heap)
            candidate_groups.discard(group_id)
            candidate = self._make_group_eviction_candidate(leaf)
            if candidate is None:
                continue
            parent = min(candidate.group_nodes, key=lambda item: item.prefix_len).parent
            for node in sorted(candidate.group_nodes, key=lambda item: item.prefix_len, reverse=True):
                self.state_checkpoints.release_checkpoint(node)
                self.leaves.discard(node)
                if node.children:
                    raise RuntimeError('Shared KV group eviction order is inconsistent.')
                node.detach_leaf()
            self.allocator.free(candidate.block_ids)
            evicted_blocks += len(candidate.block_ids)
            evicted_groups += 1

            if parent is None or parent.parent is None or parent.children:
                continue
            self.leaves.add(parent)
            candidate = self._make_group_eviction_candidate(parent)
            if candidate is None:
                continue
            if candidate.group_id not in candidate_groups:
                heapq.heappush(candidate_heap, (candidate.access_time, candidate.group_id, parent))
                candidate_groups.add(candidate.group_id)

        return evicted_blocks
