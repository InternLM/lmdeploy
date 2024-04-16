# Copyright (c) OpenMMLab. All rights reserved.
import heapq
from typing import Dict, Set

import numpy as np

from lmdeploy.pytorch.messages import SchedulerSequence

from ..config import CacheConfig
from .block_manager import BaseBlockManager


class Node:
    """node of block trie."""

    def __init__(self,
                 hash_key: int,
                 block: int,
                 tokens: np.ndarray,
                 num_matched: int = 0):
        self.hash_key = hash_key
        self.block = block
        self.tokens = tokens
        self.num_matched = num_matched
        self.children: Dict[int, 'Node'] = dict()
        self._parent: 'Node' = None

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, val: 'Node'):
        old_parent = self._parent
        if old_parent is not None:
            old_parent.children.pop(self.hash_key)
        if val is not None:
            val.children[self.hash_key] = self
        self._parent = val

    def __lt__(self, other):
        return True

    def __le__(self, other):
        return True


class BlockTrie:
    """block trie for prefix caching."""

    def __init__(self, cache_config: CacheConfig,
                 block_manager: BaseBlockManager):
        self.block_manager = block_manager
        self.cache_config = cache_config
        self.allocator = self.block_manager.allocator
        self.block_size = cache_config.block_size
        self.enable = self.cache_config.enable_prefix_caching

        # caches with different adapter should not be shared.
        self._roots: Dict[str, Node] = dict()
        self.leaves: Set[Node] = set()

    def get_root(self, adapter_name: str):
        """get root by adapter name."""
        if adapter_name not in self._roots:
            self._roots[adapter_name] = Node(-1, -1, None)
        return self._roots[adapter_name]

    def match(self, seq: SchedulerSequence):
        """match sequence and cache."""
        if not self.enable:
            return

        block_size = self.block_size
        matched_blocks = []

        logical_blocks = seq.logical_blocks
        curr: Node = getattr(logical_blocks, 'last_shared_node', None)
        if curr is None:
            curr = self.get_root(seq.adapter_name)
        num_matched = curr.num_matched

        def __match_success(node: Node):
            nonlocal curr, num_matched
            matched_blocks.append(node.block)
            curr = node
            num_matched += block_size

        while num_matched + block_size < seq.num_all_ids:
            curr_tokens = seq.history_cache[num_matched:num_matched +
                                            block_size]

            key = hash(tuple(curr_tokens))
            if key not in curr.children:
                break

            child = curr.children[key]
            if not np.array_equal(curr_tokens, child.tokens):
                break

            __match_success(child)

        if len(matched_blocks) > 0:
            matched_blocks = np.array(matched_blocks)
            self.allocator.update_access_time(matched_blocks)
            self.allocator.add_ref_count(matched_blocks, 1)
            seq.logical_blocks.append(matched_blocks)
            seq.set_step(num_matched)

        seq.logical_blocks.last_shared_node = curr

    def allocate(self, seq: SchedulerSequence):
        """allocate."""
        if not self.enable:
            return

        block_size = self.block_size
        logical_blocks = seq.logical_blocks
        node: Node = getattr(logical_blocks, 'last_shared_node', None)
        if node is None:
            node = self.get_root(seq.adapter_name)
            logical_blocks.last_shared_node = node

        num_matched = node.num_matched
        num_all_ids = seq.num_all_ids

        if num_matched + block_size > num_all_ids:
            return

        if len(node.children) == 0 and node.parent is not None:
            self.leaves.remove(node)

        block_id = num_matched // block_size
        blocks = []
        free_blocks = []
        while num_matched + block_size <= num_all_ids:
            curr_tokens = seq.history_cache[num_matched:num_matched +
                                            block_size]

            block = logical_blocks[block_id]

            hash_key = hash(tuple(curr_tokens))
            parent = node
            if hash_key in parent.children:
                child = parent.children[hash_key]
                if not np.array_equal(curr_tokens, child.tokens):
                    break
                node = child
                free_blocks.append(block)
                logical_blocks[block_id] = node.block
            else:
                node = Node(hash_key=hash_key,
                            block=block,
                            tokens=curr_tokens,
                            num_matched=num_matched + block_size)
                node.parent = parent
            blocks.append(node.block)
            num_matched += block_size
            block_id += 1

        logical_blocks.last_shared_node = node
        if node.parent is not None and len(node.children) == 0:
            # ignore root
            self.leaves.add(node)
        if len(blocks) > 0:
            self.allocator.add_ref_count(np.array(blocks), 1)
        if len(free_blocks) > 0:
            self.allocator.free(np.array(free_blocks))

    def evict(self, max_num_blocks: int):
        """evict."""
        if not self.enable:
            return 0

        def __remove_leaf(leaves, evicted_blocks):
            _, leaf = heapq.heappop(leaves)
            evicted_blocks.append(leaf.block)
            parent = leaf.parent
            leaf.parent = None
            self.leaves.remove(leaf)
            return parent

        def __add_leaf(leaves, parent):
            self.leaves.add(parent)
            if self.allocator.get_ref_count(parent.block) == 1:
                access_time = self.allocator.get_access_time(parent.block)
                heapq.heappush(leaves, (access_time, parent))

        evicted_blocks = []
        leaves = list(self.leaves)

        # filter ref-cnt == 1 (trie own one block ref)
        leave_blocks = np.array(list(leaf.block for leaf in leaves))
        ref_cnt = self.allocator.get_ref_count(leave_blocks)
        access_times = self.allocator.get_access_time(leave_blocks)
        indices = (ref_cnt == 1).nonzero()[0]
        if len(indices) == 0:
            return 0

        # make heap
        leaves = list(leaves[i] for i in indices)
        access_times = list(access_times[i] for i in indices)
        leaves = list(zip(access_times, leaves))
        heapq.heapify(leaves)

        while len(leaves) > 0 and len(evicted_blocks) < max_num_blocks:
            parent = __remove_leaf(leaves, evicted_blocks)
            if parent.parent is None:
                # ignore root
                break
            if len(parent.children) == 0:
                __add_leaf(leaves, parent)

        self.allocator.free(np.array(evicted_blocks))

        return len(evicted_blocks)
