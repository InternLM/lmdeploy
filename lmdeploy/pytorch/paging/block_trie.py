# Copyright (c) OpenMMLab. All rights reserved.
import heapq
import time
from dataclasses import dataclass

import numpy as np

from lmdeploy.pytorch.messages import SchedulerSequence

from ..config import CacheConfig
from .block_manager import BaseBlockManager


@dataclass
class PrefixCacheStats:
    """Prefix caching stats."""
    num_query_tokens: int = 0
    num_hit_tokens: int = 0

    def reset(self):
        self.num_query_tokens = 0
        self.num_hit_tokens = 0

    def hit_rate(self):
        return 0.0 if self.num_query_tokens <= 0 else float(self.num_hit_tokens) / self.num_query_tokens


class Node:
    """Node of block trie."""

    def __init__(self,
                 hash_key: int,
                 block: int,
                 tokens: np.ndarray,
                 num_matched: int = 0,
                 extra_hashes: tuple = (),
                 state_idx: int = -1,
                 state_ready: bool = False,
                 state_ref_count: int = 0,
                 state_access_time: float = 0.0,
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
        self.adapter_name = adapter_name
        self.children: dict[int, Node] = dict()
        self._parent: Node = None

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
    """Block trie for prefix caching."""

    def __init__(self, cache_config: CacheConfig, block_manager: BaseBlockManager, state_manager=None):
        self.block_manager = block_manager
        self.cache_config = cache_config
        self.allocator = self.block_manager.allocator
        self.state_manager = state_manager
        self.block_size = cache_config.block_size
        self.enable = self.cache_config.enable_prefix_caching
        self.requires_state_checkpoint = state_manager is not None and len(cache_config.states_shapes) > 0

        # caches with different adapter should not be shared.
        self._roots: dict[str, Node] = dict()
        self.leaves: set[Node] = set()
        self._state_checkpoint_index: dict[tuple, list[Node]] = dict()
        self._state_checkpoint_steps: dict[str, set[int]] = dict()
        self.stats = PrefixCacheStats()

    def hit_rate(self):
        """Get hit rate."""
        return self.stats.hit_rate()

    def get_root(self, adapter_name: str):
        """Get root by adapter name."""
        if adapter_name not in self._roots:
            self._roots[adapter_name] = Node(-1, -1, None, adapter_name=adapter_name)
        return self._roots[adapter_name]

    def _get_block_extra_hashes(self, seq: SchedulerSequence, start: int, end: int):
        """Get extra hashes for a token block."""
        return seq.get_prefix_cache_extra_hashes(start, end)

    def _make_key(self, tokens: np.ndarray, extra_hashes: tuple):
        """Make trie lookup key."""
        return hash(('random', tuple(tokens), extra_hashes))

    def _match_node(self, node: Node, tokens: np.ndarray, extra_hashes: tuple):
        """Check whether node content matches a token block."""
        return np.array_equal(tokens, node.tokens) and extra_hashes == node.extra_hashes

    def _make_state_checkpoint_lookup_key(self, seq: SchedulerSequence, step: int):
        """Make the sparse SSM checkpoint lookup key for a sequence prefix."""
        start = step - self.block_size
        end = step
        tokens = seq.history_cache[start:end]
        extra_hashes = self._get_block_extra_hashes(seq, start, end)
        return (seq.adapter_name, step, self._make_key(tokens, extra_hashes))

    def _make_state_checkpoint_node_key(self, node: Node):
        """Make the sparse SSM checkpoint lookup key for a trie node."""
        return (node.adapter_name, node.num_matched, node.hash_key)

    def _index_state_checkpoint(self, node: Node):
        """Add a ready state checkpoint to the sparse SSM index."""
        key = self._make_state_checkpoint_node_key(node)
        nodes = self._state_checkpoint_index.setdefault(key, [])
        if not any(indexed_node is node for indexed_node in nodes):
            nodes.append(node)
        steps = self._state_checkpoint_steps.setdefault(node.adapter_name, set())
        steps.add(node.num_matched)

    def _unindex_state_checkpoint(self, node: Node):
        """Remove a state checkpoint from the sparse SSM index."""
        key = self._make_state_checkpoint_node_key(node)
        nodes = self._state_checkpoint_index.get(key)
        if nodes is not None:
            nodes[:] = [indexed_node for indexed_node in nodes if indexed_node is not node]
            if len(nodes) == 0:
                self._state_checkpoint_index.pop(key)

        steps = self._state_checkpoint_steps.get(node.adapter_name)
        if steps is not None and node.num_matched in steps:
            has_step = any(key[0] == node.adapter_name and key[1] == node.num_matched
                           for key in self._state_checkpoint_index)
            if not has_step:
                steps.remove(node.num_matched)
            if len(steps) == 0:
                self._state_checkpoint_steps.pop(node.adapter_name)

    def reserve_state_checkpoint(self, node: Node):
        """Reserve a state-cache slot owned by a trie node."""
        if not self.requires_state_checkpoint or node.parent is None:
            return -1
        if node.state_ready:
            if node.state_ref_count > 0:
                return -1
            self._unindex_state_checkpoint(node)
        if node.state_idx < 0:
            if self.state_manager.get_num_free_checkpoint() == 0:
                self.evict_state_checkpoints(1)
            node.state_idx = self.state_manager.allocate_checkpoint_state()
        node.state_ready = False
        return node.state_idx

    def _clear_pending_state_checkpoint(self, seq: SchedulerSequence):
        """Clear pending checkpoint save metadata from a sequence."""
        prefix_cache = seq.prefix_cache
        prefix_cache.save_state = -1
        prefix_cache.save_step = 0
        prefix_cache.save_is_decode = False
        prefix_cache.save_node = None

    def discard_state_checkpoint_for_seq(self, seq: SchedulerSequence):
        """Discard an unpublished state checkpoint reservation for a
        sequence."""
        prefix_cache = seq.prefix_cache
        state_idx = prefix_cache.save_state
        node = prefix_cache.save_node
        is_decode = prefix_cache.save_is_decode
        self._clear_pending_state_checkpoint(seq)
        if state_idx < 0 or node is None:
            return False
        if node.state_idx == state_idx and not node.state_ready:
            if is_decode and prefix_cache.decode_state_node is node:
                prefix_cache.decode_state_node = None
            self.release_state_checkpoint(node)
            return True
        return False

    def discard_state_checkpoints(self, seqs: list[SchedulerSequence]):
        """Discard unpublished sequence state checkpoint reservations."""
        for seq in seqs:
            self.discard_state_checkpoint_for_seq(seq)

    def _get_state_checkpoint_node_for_seq(self, seq: SchedulerSequence, step: int):
        """Get the trie node that exactly represents a sequence checkpoint
        step."""
        node = getattr(seq.logical_blocks, 'last_shared_node', None)
        while node is not None and node.num_matched > step:
            node = node.parent
        if node is None or node.parent is None or node.num_matched != step:
            return None
        return node

    def _is_attached_node(self, node: Node):
        """Check whether a node is still attached to the trie."""
        parent = node.parent
        return parent is not None and parent.children.get(node.hash_key) is node

    def reserve_state_checkpoint_for_seq(self,
                                         seq: SchedulerSequence,
                                         step: int = None,
                                         is_decode: bool = False):
        """Reserve a state checkpoint slot for a sequence checkpoint step."""
        self.discard_state_checkpoint_for_seq(seq)

        if not self.enable or not self.requires_state_checkpoint:
            return -1

        if step is None:
            step = seq.num_valid_ids
        if step <= 0 or step % self.block_size != 0:
            return -1
        if step > seq.num_valid_ids:
            return -1
        if seq.clamp_prefix_cache_match_step(step) != step:
            return -1

        node = self._get_state_checkpoint_node_for_seq(seq, step)
        if node is None:
            return -1
        if node.state_ready:
            return -1

        try:
            state_idx = self.reserve_state_checkpoint(node)
        except RuntimeError as e:
            if 'No free states' not in str(e):
                raise
            return -1

        prefix_cache = seq.prefix_cache
        prefix_cache.save_state = state_idx
        prefix_cache.save_step = step
        prefix_cache.save_is_decode = is_decode
        prefix_cache.save_node = node
        return state_idx

    def reserve_decode_state_checkpoint_for_seq(self,
                                                seq: SchedulerSequence,
                                                interval: int,
                                                step: int = None):
        """Reserve a bounded decode checkpoint for a sequence."""
        if step is None:
            step = seq.num_valid_ids
        if interval <= 0 or step % interval != 0:
            return -1
        if not self.enable or not self.requires_state_checkpoint:
            return -1
        if step <= 0 or step % self.block_size != 0:
            return -1
        if step > seq.num_valid_ids:
            return -1
        if seq.clamp_prefix_cache_match_step(step) != step:
            return -1
        node = self._get_state_checkpoint_node_for_seq(seq, step)
        if node is None or node.state_ready:
            return -1

        prefix_cache = seq.prefix_cache
        old_node = prefix_cache.decode_state_node
        if old_node is not None and old_node.state_idx < 0:
            prefix_cache.decode_state_node = None
            old_node = None
        if old_node is not None:
            if old_node.num_matched == step and old_node.state_ready:
                return -1
            if old_node.state_ref_count > 0:
                return -1
            self.release_state_checkpoint(old_node)
            prefix_cache.decode_state_node = None

        return self.reserve_state_checkpoint_for_seq(seq, step=step, is_decode=True)

    def mark_state_checkpoint_ready(self, node: Node):
        """Mark a node-owned state checkpoint as ready for SSM matching."""
        if node.state_idx < 0:
            raise RuntimeError('Cannot mark an unreserved state checkpoint as ready.')
        if node.state_ready:
            self._unindex_state_checkpoint(node)
        node.state_ready = True
        node.state_access_time = time.perf_counter()
        self._index_state_checkpoint(node)

    def commit_state_checkpoint_for_seq(self, seq: SchedulerSequence):
        """Publish a sequence state checkpoint after its state copy is
        enqueued."""
        prefix_cache = seq.prefix_cache
        state_idx = prefix_cache.save_state
        save_step = prefix_cache.save_step
        is_decode = prefix_cache.save_is_decode
        node = prefix_cache.save_node
        self._clear_pending_state_checkpoint(seq)
        if state_idx < 0:
            return False

        if (node is None or not self._is_attached_node(node) or node.state_idx != state_idx
                or node.num_matched != save_step):
            if node is not None and node.state_idx == state_idx and not node.state_ready:
                if is_decode and prefix_cache.decode_state_node is node:
                    prefix_cache.decode_state_node = None
                self.release_state_checkpoint(node)
            return False

        self.mark_state_checkpoint_ready(node)
        if is_decode:
            prefix_cache.decode_state_node = node
        return True

    def commit_state_checkpoints(self, seqs: list[SchedulerSequence]):
        """Publish pending sequence state checkpoints."""
        for seq in seqs:
            self.commit_state_checkpoint_for_seq(seq)

    def acquire_state_checkpoint_restore_for_seq(self, seq: SchedulerSequence):
        """Pin a matched state checkpoint until its restore copy has
        completed."""
        prefix_cache = seq.prefix_cache
        if prefix_cache.restore_state < 0 or prefix_cache.restore_state_acquired:
            return False
        node = getattr(seq.logical_blocks, 'last_shared_node', None)
        if node is None or node.state_idx != prefix_cache.restore_state or not node.state_ready:
            return False
        node.state_ref_count += 1
        node.state_access_time = time.perf_counter()
        prefix_cache.restore_state_acquired = True
        return True

    def acquire_state_checkpoint_restores(self, seqs: list[SchedulerSequence]):
        """Pin matched state checkpoints for a batch."""
        for seq in seqs:
            self.acquire_state_checkpoint_restore_for_seq(seq)

    def release_state_checkpoint_restore_for_seq(self, seq: SchedulerSequence):
        """Release a state checkpoint pinned for restore."""
        prefix_cache = seq.prefix_cache
        if not prefix_cache.restore_state_acquired:
            return False
        node = getattr(seq.logical_blocks, 'last_shared_node', None)
        if node is not None and node.state_idx == prefix_cache.restore_state and node.state_ref_count > 0:
            node.state_ref_count -= 1
        prefix_cache.restore_state = -1
        prefix_cache.restore_state_acquired = False
        return True

    def release_state_checkpoint_restores(self, seqs: list[SchedulerSequence]):
        """Release state checkpoints pinned for a batch restore."""
        for seq in seqs:
            self.release_state_checkpoint_restore_for_seq(seq)

    def release_state_checkpoint(self, node: Node):
        """Release a node-owned state checkpoint."""
        if node.state_idx < 0:
            return
        if node.state_ready:
            self._unindex_state_checkpoint(node)
        self.state_manager.free_checkpoint_state(node.state_idx)
        node.state_idx = -1
        node.state_ready = False
        node.state_ref_count = 0
        node.state_access_time = 0.0

    def evict_state_checkpoints(self, max_num_states: int):
        """Evict ready SSM state checkpoints without removing KV trie nodes."""
        if not self.requires_state_checkpoint or max_num_states <= 0:
            return 0

        candidates = []
        seen_nodes = set()
        for nodes in self._state_checkpoint_index.values():
            for node in nodes:
                node_id = id(node)
                if node_id in seen_nodes:
                    continue
                seen_nodes.add(node_id)
                if node.state_idx >= 0 and node.state_ready and node.state_ref_count == 0:
                    candidates.append((node.state_access_time, node))
        heapq.heapify(candidates)

        evicted = 0
        while len(candidates) > 0 and evicted < max_num_states:
            _, node = heapq.heappop(candidates)
            if node.state_idx < 0 or not node.state_ready or node.state_ref_count > 0:
                continue
            self.release_state_checkpoint(node)
            evicted += 1
        return evicted

    def _get_node_blocks(self, node: Node):
        """Get trie nodes from root to a target node."""
        nodes = []
        while node is not None and node.parent is not None:
            nodes.append(node)
            node = node.parent
        nodes.reverse()
        return nodes

    def _verify_state_checkpoint_node(self, seq: SchedulerSequence, node: Node):
        """Verify a sparse SSM checkpoint candidate exactly."""
        if node.adapter_name != seq.adapter_name or not node.state_ready or node.state_idx < 0:
            return None

        step = node.num_matched
        if step <= 0 or step > ((seq.num_valid_ids - 1) // self.block_size) * self.block_size:
            return None
        if seq.clamp_prefix_cache_match_step(step) != step:
            return None

        nodes = self._get_node_blocks(node)
        if len(nodes) * self.block_size != step:
            return None

        matched_blocks = []
        for idx, block_node in enumerate(nodes):
            start = idx * self.block_size
            end = start + self.block_size
            tokens = seq.history_cache[start:end]
            extra_hashes = self._get_block_extra_hashes(seq, start, end)
            if not self._match_node(block_node, tokens, extra_hashes):
                return None
            matched_blocks.append(block_node.block)

        return matched_blocks

    def _match_state_checkpoint(self, seq: SchedulerSequence):
        """Match SSM prefixes through sparse ready-checkpoint lookup."""
        seq.prefix_cache.restore_state = -1

        init_curr = getattr(seq.logical_blocks, 'last_shared_node', None)
        if init_curr is None:
            init_curr = self.get_root(seq.adapter_name)
        init_num_matched = init_curr.num_matched

        max_step = ((seq.num_valid_ids - 1) // self.block_size) * self.block_size
        steps = self._state_checkpoint_steps.get(seq.adapter_name, ())
        for step in sorted((step for step in steps if init_num_matched < step <= max_step), reverse=True):
            if seq.clamp_prefix_cache_match_step(step) != step:
                continue
            key = self._make_state_checkpoint_lookup_key(seq, step)
            for node in self._state_checkpoint_index.get(key, ()):
                matched_blocks = self._verify_state_checkpoint_node(seq, node)
                if matched_blocks is None:
                    continue

                matched_blocks = np.array(matched_blocks[init_num_matched // self.block_size:])
                self.allocator.update_access_time(matched_blocks)
                self.allocator.add_ref_count(matched_blocks, 1)
                seq.logical_blocks.append(matched_blocks)
                seq.set_step(step)
                seq.prefix_cache.restore_state = node.state_idx
                seq.logical_blocks.last_shared_node = node
                self.stats.num_query_tokens += seq.num_all_ids - init_num_matched
                self.stats.num_hit_tokens += step - init_num_matched
                return

        seq.logical_blocks.last_shared_node = init_curr
        self.stats.num_query_tokens += seq.num_all_ids - init_num_matched

    def match(self, seq: SchedulerSequence):
        """Match sequence and cache."""
        if not self.enable:
            return
        seq.prefix_cache.restore_state = -1
        if self.requires_state_checkpoint:
            self._match_state_checkpoint(seq)
            return

        block_size = self.block_size
        matched_blocks = []

        logical_blocks = seq.logical_blocks
        curr: Node = getattr(logical_blocks, 'last_shared_node', None)
        if curr is None:
            curr = self.get_root(seq.adapter_name)
        init_curr = curr
        init_num_matched = curr.num_matched
        num_matched = curr.num_matched

        def __match_success(node: Node):
            nonlocal curr, num_matched
            matched_blocks.append(node.block)
            curr = node
            num_matched += block_size

        matched_nodes: list[Node] = []

        while num_matched + block_size < seq.num_valid_ids:
            start = num_matched
            end = num_matched + block_size
            curr_tokens = seq.history_cache[start:end]
            extra_hashes = self._get_block_extra_hashes(seq, start, end)

            key = self._make_key(curr_tokens, extra_hashes)
            if key not in curr.children:
                break

            child = curr.children[key]
            if not self._match_node(child, curr_tokens, extra_hashes):
                break

            matched_nodes.append(child)
            __match_success(child)

        def __clamp_match_step(match_step: int):
            nonlocal curr, num_matched, matched_blocks, matched_nodes
            match_step = max(init_num_matched, match_step)
            if match_step >= num_matched:
                return
            keep = (match_step - init_num_matched) // block_size
            matched_nodes = matched_nodes[:keep]
            matched_blocks = matched_blocks[:keep]
            if keep > 0:
                curr = matched_nodes[-1]
                num_matched = curr.num_matched
            else:
                curr = init_curr
                num_matched = init_num_matched

        clamped_num_matched = seq.clamp_prefix_cache_match_step(num_matched)
        __clamp_match_step(clamped_num_matched)

        if len(matched_blocks) > 0:
            matched_blocks = np.array(matched_blocks)
            self.allocator.update_access_time(matched_blocks)
            self.allocator.add_ref_count(matched_blocks, 1)
            seq.logical_blocks.append(matched_blocks)
            seq.set_step(num_matched)
            if self.requires_state_checkpoint:
                seq.prefix_cache.restore_state = curr.state_idx

        # record prefix hit
        self.stats.num_query_tokens += seq.num_all_ids - init_num_matched
        self.stats.num_hit_tokens += num_matched - init_num_matched

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
        num_valid_ids = seq.num_valid_ids

        if num_matched + block_size > num_valid_ids:
            return

        if len(node.children) == 0 and node.parent is not None:
            self.leaves.remove(node)

        block_id = num_matched // block_size
        blocks = []
        free_blocks = []
        while num_matched + block_size <= num_valid_ids:
            start = num_matched
            end = num_matched + block_size
            curr_tokens = seq.history_cache[start:end]
            extra_hashes = self._get_block_extra_hashes(seq, start, end)

            block = logical_blocks[block_id]

            hash_key = self._make_key(curr_tokens, extra_hashes)
            parent = node
            if hash_key in parent.children:
                child = parent.children[hash_key]
                if not self._match_node(child, curr_tokens, extra_hashes):
                    break
                node = child
                free_blocks.append(block)
                logical_blocks[block_id] = node.block
            else:
                node = Node(hash_key=hash_key,
                            block=block,
                            tokens=curr_tokens,
                            num_matched=num_matched + block_size,
                            extra_hashes=extra_hashes,
                            adapter_name=seq.adapter_name)
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
            self.release_state_checkpoint(leaf)
            parent = leaf.parent
            leaf.parent = None
            self.leaves.remove(leaf)
            return parent

        def __add_leaf(leaves, parent):
            self.leaves.add(parent)
            if self.allocator.get_ref_count(parent.block) == 1:
                access_time = self.allocator.get_access_time(parent.block)
                heapq.heappush(leaves, (access_time, parent))

        if len(self.leaves) == 0:
            return 0

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
            parent = __remove_leaf(leaves, evicted_blocks)
            if parent.parent is None:
                # ignore root
                continue
            if len(parent.children) == 0:
                __add_leaf(leaves, parent)

        self.allocator.free(np.array(evicted_blocks))

        return len(evicted_blocks)
