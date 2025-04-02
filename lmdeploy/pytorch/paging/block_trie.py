# Copyright (c) OpenMMLab. All rights reserved.
import heapq
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

import numpy as np

from lmdeploy.pytorch.messages import SchedulerSequence
from lmdeploy.utils import get_logger, logging_timer

from ..config import CacheConfig
from .block_manager import BaseBlockManager

logger = get_logger('lmdeploy')


def hash_block_tokens(tokens: np.ndarray, mm_hashes=None):
    """hash func."""
    if mm_hashes is None:
        mm_hashes = ('None', )
    hash_data = (mm_hashes, tuple(tokens))
    hash_key = hash(hash_data)
    return hash_key


@dataclass
class Node:
    """node of block trie."""
    hash_key: int
    block: int
    tokens: np.ndarray
    num_matched: int = 0
    mm_hashes: Optional[Tuple[str]] = None

    def __post_init__(self):
        """post init."""
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

    def match_child(self, tokens: np.ndarray, mm_hashes=None):
        """Match child."""
        hash_key = hash_block_tokens(tokens, mm_hashes=mm_hashes)
        hash_collision = None
        matched_child = None
        if hash_key in self.children:
            child = self.children[hash_key]
            if child.mm_hashes == mm_hashes and np.array_equal(tokens, child.tokens):
                matched_child = child
                hash_collision = False
            else:
                hash_collision = True
                logger.error(f'Hash collision found for tokens={tokens}, '
                             f'mm_hashes={mm_hashes} with node={child}')
        return matched_child, hash_collision, hash_key

    def __hash__(self):
        return hash((self.block, self.num_matched, self.hash_key))

    def __lt__(self, other):
        return True

    def __le__(self, other):
        return True

    def __repr__(self):
        return (f'Node(hash_key={self.hash_key}, block={self.block}, '
                f'num_matched={self.num_matched}, mm_hashes={self.mm_hashes}, '
                f'num_children={len(self.children)}, is_root={self.parent is None}, '
                f'tokens={self.tokens})')

    __str__ = __repr__


class BlockTrie:
    """block trie for prefix caching."""

    def __init__(self, cache_config: CacheConfig, block_manager: BaseBlockManager):
        self.block_manager = block_manager
        self.cache_config = cache_config
        self.allocator = self.block_manager.allocator
        self.block_size = cache_config.block_size
        self.enable = self.cache_config.enable_prefix_caching

        # caches with different adapter should not be shared.
        self._roots: Dict[str, Node] = dict()
        self.leaves: Set[Node] = set()
        self.hit_rates = []

    def get_root(self, adapter_name: str):
        """get root by adapter name."""
        if adapter_name not in self._roots:
            self._roots[adapter_name] = Node(-1, -1, None)
        return self._roots[adapter_name]

    @logging_timer('BlockTrie_Match', logger)
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
        init_num_matched = num_matched

        def __match_success(node: Node):
            nonlocal curr, num_matched
            matched_blocks.append(node.block)
            curr = node
            num_matched += block_size

        def __match_pure_text():
            nonlocal curr, num_matched
            while num_matched + block_size <= seq.num_all_ids:
                curr_tokens = seq.history_cache[num_matched:num_matched + block_size]
                child, _, _ = curr.match_child(curr_tokens, mm_hashes=None)
                if child is None:
                    break
                __match_success(child)

        def __match_multimodals():
            nonlocal curr, num_matched, mm_ranges, matched_blocks
            while num_matched + block_size <= seq.num_all_ids:
                if len(mm_ranges) > 0 and num_matched <= mm_ranges[0][0] < (num_matched + block_size):
                    # find last block without img_data intersect
                    last_end_num_matched = -1
                    for mm_idx, (_, end, _) in enumerate(mm_ranges):
                        end_num_matched = (((end - 1) // block_size) + 1) * block_size
                        if end_num_matched > seq.num_all_ids:
                            # last block that include end token is not full, just stop quickly
                            break
                        intersect_ranges = [data for data in mm_ranges[mm_idx + 1:] if data[0] < end_num_matched]
                        if len(intersect_ranges) == 0:
                            last_end_num_matched = end_num_matched
                            break
                    if last_end_num_matched == -1:
                        break

                    mutimodal_matched_blocks = []
                    all_match = True
                    multi_curr = curr
                    for multi_num_matched in range(num_matched, last_end_num_matched, block_size):
                        num_matched_end = multi_num_matched + block_size
                        curr_tokens = seq.history_cache[multi_num_matched:num_matched_end]
                        mm_hashes = tuple([data[2] for data in mm_ranges if data[0] < num_matched_end])
                        child, _, _ = multi_curr.match_child(curr_tokens, mm_hashes=mm_hashes)
                        if child is not None:
                            mutimodal_matched_blocks.append(child.block)
                            mm_ranges = [data for data in mm_ranges if data[1] > num_matched_end]
                            multi_curr = child
                        else:
                            all_match = False
                            break
                    if all_match:
                        matched_blocks += mutimodal_matched_blocks
                        num_matched = last_end_num_matched
                        curr = multi_curr
                    else:
                        break
                else:
                    curr_tokens = seq.history_cache[num_matched:num_matched + block_size]
                    child, _, _ = curr.match_child(curr_tokens, mm_hashes=None)
                    if child is None:
                        break
                    __match_success(child)

        mm_ranges = None

        if seq.history_multimodals is not None and len(seq.history_multimodals.mm_ranges) > 0:
            mm_ranges = list(seq.history_multimodals.mm_ranges)
            mm_ranges = [data for data in mm_ranges if num_matched < data[1]]
            if len(mm_ranges) == 0:
                mm_ranges = None

        if mm_ranges is None:
            __match_pure_text()
        else:
            __match_multimodals()

        if len(matched_blocks) > 0:
            matched_blocks = np.array(matched_blocks)
            self.allocator.update_access_time(matched_blocks)
            self.allocator.add_ref_count(matched_blocks, 1)
            seq.logical_blocks.append(matched_blocks)
            seq.set_step(num_matched)
        hit_rate = 100 * len(matched_blocks) * block_size / float(seq.num_all_ids - init_num_matched)
        self.hit_rates.append(hit_rate)
        seq.logical_blocks.last_shared_node = curr
        logger.info(f'Block Trie current hit rate={hit_rate}%, '
                    f'mean hit rate={np.mean(self.hit_rates)}%, matching seq={seq}')

    @logging_timer('BlockTrie_Allocate', logger)
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
        logger.info(f'Allocate seq={seq}')
        num_matched = node.num_matched
        num_all_ids = seq.num_all_ids

        if num_matched + block_size > num_all_ids:
            return

        if len(node.children) == 0 and node.parent is not None:
            self.leaves.remove(node)

        blocks = []
        free_blocks = []

        def __allocate_text():
            nonlocal node, num_matched, blocks, free_blocks

            block_id = num_matched // block_size
            while num_matched + block_size <= num_all_ids:
                curr_tokens = seq.history_cache[num_matched:num_matched + block_size]

                block = logical_blocks[block_id]
                parent = node

                mm_hashes = None
                matched_child, hash_collision, hash_key = node.match_child(curr_tokens, mm_hashes=mm_hashes)
                if hash_collision:
                    break

                if matched_child is not None:
                    node = matched_child
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

        def __allocate_multimodals():
            nonlocal node, num_matched, blocks, free_blocks, mm_ranges

            block_id = num_matched // block_size

            while num_matched + block_size <= num_all_ids:
                if len(mm_ranges) > 0 and (mm_ranges[0][0] // block_size) == block_id:
                    # find last block without img_data intersect
                    last_end_num_matched = -1
                    for mm_idx, (_, end, _) in enumerate(mm_ranges):
                        end_num_matched = (((end - 1) // block_size) + 1) * block_size
                        if end_num_matched > seq.num_all_ids:
                            # last block that include end token is not full, just stop quickly
                            break
                        intersect_ranges = [data for data in mm_ranges[mm_idx + 1:] if data[0] < end_num_matched]
                        if len(intersect_ranges) == 0:
                            last_end_num_matched = end_num_matched
                            break
                    if last_end_num_matched == -1:
                        break

                    multi_blocks = []
                    multi_free_blocks = []
                    multi_node = node
                    all_allocate = True
                    multi_block_id = block_id
                    for multi_num_matched in range(num_matched, last_end_num_matched, block_size):
                        num_matched_end = multi_num_matched + block_size
                        curr_tokens = seq.history_cache[multi_num_matched:num_matched_end]
                        mm_hashes = tuple([data[2] for data in mm_ranges if data[0] < num_matched_end])
                        matched_child, hash_collision, hash_key = multi_node.match_child(curr_tokens,
                                                                                         mm_hashes=mm_hashes)
                        if hash_collision:
                            all_allocate = False
                            break
                        block = logical_blocks[multi_block_id]
                        parent = multi_node
                        if matched_child is not None:
                            multi_node = matched_child
                            multi_free_blocks.append(block)
                            logical_blocks[multi_block_id] = matched_child.block
                        else:
                            multi_node = Node(hash_key=hash_key,
                                              block=block,
                                              tokens=curr_tokens,
                                              num_matched=num_matched_end,
                                              mm_hashes=mm_hashes)
                            multi_node.parent = parent
                        multi_blocks.append(multi_node.block)
                        multi_block_id += 1
                        mm_ranges = [data for data in mm_ranges if data[1] > num_matched_end]

                    if all_allocate:
                        blocks += multi_blocks
                        free_blocks += multi_free_blocks
                        num_matched = last_end_num_matched
                        node = multi_node
                        block_id = multi_block_id
                    else:
                        break
                else:
                    curr_tokens = seq.history_cache[num_matched:num_matched + block_size]

                    block = logical_blocks[block_id]
                    parent = node

                    mm_hashes = None
                    matched_child, hash_collision, hash_key = node.match_child(curr_tokens, mm_hashes=mm_hashes)
                    if hash_collision:
                        break

                    if matched_child is not None:
                        node = matched_child
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

        mm_ranges = None

        if seq.history_multimodals is not None and len(seq.history_multimodals.mm_ranges) > 0:
            mm_ranges = list(seq.history_multimodals.mm_ranges)
            mm_ranges = [data for data in mm_ranges if num_matched < data[1]]
            if len(mm_ranges) == 0:
                mm_ranges = None

        if mm_ranges is None:
            __allocate_text()
        else:
            __allocate_multimodals()

        logical_blocks.last_shared_node = node
        if node.parent is not None and len(node.children) == 0:
            # ignore root
            self.leaves.add(node)
        if len(blocks) > 0:
            self.allocator.add_ref_count(np.array(blocks), 1)
        if len(free_blocks) > 0:
            self.allocator.free(np.array(free_blocks))

    @logging_timer('BlockTrie_Evict', logger)
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
            return parent, leaf

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
        indices = (ref_cnt == 1).nonzero()[0]
        if len(indices) == 0:
            return 0

        # make heap
        leaves = list(leaves[i] for i in indices)
        access_times = self.allocator.get_access_time(leave_blocks)
        access_times = list(access_times[i] for i in indices)
        leaves = list(zip(access_times, leaves))
        heapq.heapify(leaves)

        while len(leaves) > 0 and len(evicted_blocks) < max_num_blocks:
            parent, removed_leaf = __remove_leaf(leaves, evicted_blocks)
            if parent.parent is None:
                # ignore root
                continue

            # remove nodes of with same mm_hashes
            if removed_leaf.mm_hashes:
                while removed_leaf.mm_hashes == parent.mm_hashes and len(parent.children) == 0:
                    tmp_parent = parent.parent
                    evicted_blocks.append(parent.block)
                    parent.parent = None
                    logger.info(f'Evict multimodal node={parent}')
                    parent = tmp_parent
                    logger.info(f'Next multimodal node={parent}')

                if parent.parent is None:
                    # ignore root
                    continue

            if len(parent.children) == 0:
                __add_leaf(leaves, parent)

        self.allocator.free(np.array(evicted_blocks))

        return len(evicted_blocks)
