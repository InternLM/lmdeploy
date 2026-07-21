# Copyright (c) OpenMMLab. All rights reserved.
"""Prefix-cache trie ownership and lifecycle.

``BlockTrie`` owns reusable prefix identity, trie-owned KV block references,
optional SSM state checkpoints, and optional routed-expert replay data.  Read
this module together with ``Scheduler._schedule_prefill``,
``InputsMaker.create_model_inputs*``, ``model_forward``, and
``EngineLoop._publish_forward_prefix_cache``.

Pipeline summary:

1. The scheduler calls ``match()`` before eviction/allocation.  A match mutates
   sequence state tentatively: it may append shared KV blocks, advance
   ``seq.num_history_ids``, set SSM restore metadata, and replay routed experts.
2. If scheduling later fails, the scheduler rolls that tentative match back.
   If it succeeds, ``block_manager.allocate()`` gives blocks for the uncached
   suffix, and ``allocate()`` attaches newly completed full blocks to the trie.
3. Text/VLM matching walks trie blocks by adapter root.  Each block key is
   token ids plus multimodal extra hashes; matches are clamped so forward never
   starts inside a multimodal span.
4. SSM matching cannot reuse KV alone.  It uses sparse ready checkpoint lookup,
   verifies the full ancestor chain, then asks ``ModelAgent`` to copy the
   frozen checkpoint state into the request runtime state on the forward stream.
5. SSM checkpoint saves are reserved here, copied by ``ModelAgent`` after
   forward, and published by ``EngineLoop`` once the producer forward is queued.
   Producer/restore refcounts pin checkpoint slots across async stream-ordering
   windows.

SSM checkpoint detail:

* ``seq.prefix_cache.last_shared_node`` stores the deepest trie node already
  shared by the sequence.  ``match()`` writes it, rollback/free clears it, and
  ``allocate()`` continues inserting newly computed full blocks from it.
* ``StateManager`` owns one state-cache pool split by role: active requests use
  runtime slots stored on ``seq.logical_state``; prefix-cache checkpoints use
  slots stored on trie ``Node.state_idx``.  A trie node may own KV only, KV plus
  an unready checkpoint reservation, or KV plus a ready checkpoint.
* Saving a checkpoint starts from an already-attached block-aligned trie node.
  ``reserve_state_checkpoint_for_seq()`` records ``save_state``, ``save_step``,
  ``save_node``, and ``save_is_decode`` on ``seq.prefix_cache``.  Prefill and
  long-context chunks save at the produced chunk end; decode saves are optional
  and bounded by ``prefix_cache_decode_state_interval``.
* ``InputsMaker`` converts those pending saves into compact host integer
  src/dst pairs.  ``ModelAgent`` then copies ``runtime_state -> checkpoint`` on
  the model forward stream after the model has produced the new SSM state.
  ``EngineLoop`` calls ``commit_state_checkpoint_for_seq()`` after the forward
  is queued; only then does ``state_ready`` become true and the sparse
  checkpoint index become matchable.  The producing forward holds a producer ref
  until the output/event boundary, so this early visibility cannot make the
  destination slot evictable before the save copy reaches the forward stream.
  Abandoned reservations are discarded.
* Matching a SSM prefix never walks KV blocks as the source of truth.
  ``_match_state_checkpoint()`` searches ready checkpoint steps, filters by
  ``(adapter, step, last_block_hash)``, then proves the complete prefix against
  immutable exact-match metadata built when the checkpoint is published.  A
  hit appends trie-owned KV blocks, advances ``seq.num_history_ids``, records
  ``restore_state``/``restore_node``, and may replay routed experts.
* Restore is two-phase.  The scheduler/input maker pins the ready checkpoint by
  incrementing ``state_ref_count``.  ``ModelAgent`` copies
  ``checkpoint -> runtime_state`` before the suffix forward.  ``EngineLoop``
  releases the pin once the copy has been queued, so LRU eviction cannot reuse
  the checkpoint source slot too early.
* Checkpoint eviction is state-only LRU over ready, unpinned nodes.  KV leaf
  eviction also releases any checkpoint owned by that leaf.  A KV match without
  an exact ready SSM checkpoint is intentionally a miss.
"""

import heapq
import logging
import time
from dataclasses import dataclass

import numpy as np

from lmdeploy.pytorch.messages import PrefixCacheExtraHashes, SchedulerSequence
from lmdeploy.utils import get_logger

from ...config import CacheConfig
from ..block_manager import BaseBlockManager
from .checkpoint import (
    StateCheckpointIndex,
    StateCheckpointMatchData,
    StateCheckpointVerifyStatus,
    freeze_state_checkpoint_match_data,
    make_node_multimodal_identity,
    make_request_multimodal_identity,
)

logger = get_logger('lmdeploy')


@dataclass
class PrefixCacheStats:
    """Prefix caching stats."""
    num_query_tokens: int = 0
    num_hit_tokens: int = 0

    def reset(self):
        self.num_query_tokens = 0
        self.num_hit_tokens = 0

    def copy(self):
        """Copy stats for tentative-match rollback."""
        return PrefixCacheStats(num_query_tokens=self.num_query_tokens, num_hit_tokens=self.num_hit_tokens)

    def hit_rate(self):
        return 0.0 if self.num_query_tokens <= 0 else float(self.num_hit_tokens) / self.num_query_tokens


class Node:
    """One full-token-block edge in the prefix-cache trie.

    A non-root node owns one trie KV-block reference.  ``hash_key``,
    ``tokens``, and ``extra_hashes`` define the block identity; ``extra_hashes``
    carries VLM content identity for blocks that overlap multimodal spans.

    The same node may also own an optional SSM checkpoint.  ``state_idx`` is
    the checkpoint slot, and ``state_ref_count`` pins the slot while an async
    restore may still read it or a producer save may still write it.
    ``state_ready`` together with non-``None`` ``state_match_data`` means the
    slot has been published and is matchable.  A topology change invalidates
    ``state_match_data`` immediately; lookup later removes that temporarily
    stale ready/index entry.
    ``state_match_data`` caches the immutable host identity and logical KV path
    used to prove and apply an exact checkpoint hit without repeated Python
    block scans.

    ``parent`` is intentionally stateful: assigning it updates the old and new
    parent ``children`` maps and invalidates cached checkpoint paths in the
    moved subtree.  Detached nodes can therefore still exist as stale
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
        self.children: dict[int, Node] = dict()
        self._parent: Node = None

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, val: 'Node'):
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


class BlockTrie:
    """Prefix-cache facade for trie KV reuse and SSM state reuse.

    Public scheduling flow stays small: ``match(seq)`` tentatively reuses a
    prefix, the scheduler admits resources, then ``allocate(seq)`` attaches the
    newly computed full blocks.  ``evict()`` frees trie-owned KV leaves under
    allocator pressure.

    Internally this facade owns several related indexes:

    * adapter-partitioned trie roots and leaf candidates for KV eviction;
    * multimodal-aware block keys;
    * optional SSM checkpoint reserve/commit/restore/save lifecycle;
    * sparse ready-checkpoint lookup for SSM prefix hits;
    * best-effort routed-expert replay data;
    * prefix-cache stats used by scheduler rollback.
    """

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
        # SSM checkpoints are sparse.  The trie still owns KV blocks, but ready
        # recurrent-state snapshots are indexed only at selected exact steps.
        self._state_checkpoints = StateCheckpointIndex(self.block_size, self._make_key)
        self.stats = PrefixCacheStats()

    # Prefix-cache stats and tentative-match rollback helpers.

    def hit_rate(self):
        """Get hit rate."""
        return self.stats.hit_rate()

    def snapshot_stats(self):
        """Snapshot prefix-cache stats before a tentative match."""
        if not self.enable:
            return None
        return self.stats.copy()

    def restore_stats(self, snapshot: PrefixCacheStats | None):
        """Restore prefix-cache stats for an unused tentative match."""
        if snapshot is None:
            return
        self.stats.num_query_tokens = snapshot.num_query_tokens
        self.stats.num_hit_tokens = snapshot.num_hit_tokens

    def _record_match_stats(self, seq: SchedulerSequence, query_tokens: int, hit_tokens: int = 0):
        """Record a user-visible prefix-cache match attempt."""
        if seq.prefix_cache.suppress_match_stats:
            return
        self.stats.num_query_tokens += query_tokens
        self.stats.num_hit_tokens += hit_tokens

    @staticmethod
    def _warn_unexpected_state(message: str):
        """Warn about contradictory internal trie/checkpoint state."""
        logger.warning('Unexpected prefix-cache trie state: %s', message)

    # Private recompute-overlap helpers for AR-spec/MTP prefix hits.

    @staticmethod
    def _clear_private_recompute_range(seq: SchedulerSequence):
        """Clear the one-shot private overlap allocation window."""
        prefix_cache = seq.prefix_cache
        prefix_cache.private_recompute_start_step = -1
        prefix_cache.private_recompute_end_step = -1

    def _set_private_recompute_range(self, seq: SchedulerSequence, start: int, end: int):
        """Mark matched-but-dropped trie blocks as private recompute overlap.

        ``BlockTrie.allocate()`` runs before the model forward.  If a dropped
        overlap block already exists in the trie, allocating normally would
        deduplicate the sequence back to the shared cached block and the forward
        would overwrite shared KV.  This range lets allocation traverse the trie
        path for identity while keeping the sequence's newly allocated blocks.
        """
        start = (start // self.block_size) * self.block_size
        end = (end // self.block_size) * self.block_size
        if end <= start:
            self._clear_private_recompute_range(seq)
            return
        prefix_cache = seq.prefix_cache
        prefix_cache.private_recompute_start_step = start
        prefix_cache.private_recompute_end_step = end

    @staticmethod
    def _is_private_recompute_step(seq: SchedulerSequence, step: int):
        prefix_cache = seq.prefix_cache
        start = prefix_cache.private_recompute_start_step
        end = prefix_cache.private_recompute_end_step
        return start >= 0 and start <= step < end

    # Trie keying and raw block matching helpers.

    def get_root(self, adapter_name: str):
        """Get root by adapter name."""
        if adapter_name not in self._roots:
            self._roots[adapter_name] = Node(-1, -1, None, adapter_name=adapter_name)
        return self._roots[adapter_name]

    @staticmethod
    def _get_block_extra_hashes(seq: SchedulerSequence, start: int, end: int) -> PrefixCacheExtraHashes:
        """Get multimodal identity entries that belong in a block key."""
        return seq.get_prefix_cache_extra_hashes(start, end)

    @staticmethod
    def _make_key(tokens: np.ndarray, extra_hashes: PrefixCacheExtraHashes):
        """Make the trie lookup key from tokens plus multimodal identity."""
        return hash(('random', tuple(tokens), extra_hashes))

    @staticmethod
    def _match_node(node: Node, tokens: np.ndarray, extra_hashes: PrefixCacheExtraHashes):
        """Check the exact key payload after the hash-table lookup."""
        return np.array_equal(tokens, node.tokens) and extra_hashes == node.extra_hashes

    def _find_raw_block_match_step(self, seq: SchedulerSequence, curr: Node):
        """Find the deepest KV trie match without acquiring block refs."""
        block_size = self.block_size
        num_matched = curr.num_matched
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

            curr = child
            num_matched += block_size
        return num_matched

    # Routed-expert cache/replay helpers.  Routed experts enrich a hit when
    # complete block rows are available, but they are not part of cache identity.

    @staticmethod
    def _get_routed_experts_for_range(seq: SchedulerSequence, start: int, end: int):
        """Get a copy of routed experts for a full token range, if present."""
        if not seq.return_routed_experts:
            return None
        all_routed_experts = seq.all_routed_experts
        if all_routed_experts is None:
            return None
        if len(all_routed_experts) < seq.num_history_ids or len(all_routed_experts) < end:
            return None
        routed_experts = all_routed_experts.get_real()
        if routed_experts is None or len(routed_experts) < end:
            return None
        return routed_experts[start:end].copy()

    def _try_cache_node_routed_experts(self, node: Node, seq: SchedulerSequence, start: int, end: int):
        """Attach routed experts to a trie node when a sequence has them."""
        if node.routed_experts is not None:
            return
        routed_experts = self._get_routed_experts_for_range(seq, start, end)
        if routed_experts is not None and len(routed_experts) == end - start:
            node.routed_experts = routed_experts

    def _append_matched_routed_experts(self, seq: SchedulerSequence, nodes: list[Node], start: int):
        """Replay cached routed experts for a matched trie range."""
        if not seq.return_routed_experts or len(nodes) == 0:
            return
        if len(seq.all_routed_experts) != start:
            return

        expert_slices = []
        for node in nodes:
            routed_experts = node.routed_experts
            if routed_experts is None or len(routed_experts) != self.block_size:
                return
            expert_slices.append(routed_experts)

        seq.append_routed_experts(np.concatenate(expert_slices, axis=0).copy())

    def _append_state_checkpoint_routed_experts(self, seq: SchedulerSequence, node: Node, start: int):
        """Replay checkpoint-path experts only when the request asks for
        them."""
        if not seq.return_routed_experts:
            return
        nodes = self._get_node_blocks(node)
        nodes = nodes[start // self.block_size:]
        self._append_matched_routed_experts(seq, nodes, start)

    def cache_routed_experts_for_seq(self, seq: SchedulerSequence):
        """Enrich attached trie nodes with routed experts from a sequence."""
        if not self.enable or not seq.return_routed_experts:
            return
        node = seq.prefix_cache.last_shared_node
        while node is not None and node.parent is not None:
            end = node.num_matched
            start = end - self.block_size
            self._try_cache_node_routed_experts(node, seq, start, end)
            node = node.parent

    def cache_routed_experts(self, seqs: list[SchedulerSequence]):
        """Enrich trie nodes with routed experts from multiple sequences."""
        if not self.enable:
            return
        for seq in seqs:
            self.cache_routed_experts_for_seq(seq)

    # Sparse SSM checkpoint index facade.  The extracted index owns lookup and
    # exact verification; this class validates trie ownership before mutation.

    def _make_state_checkpoint_lookup_key(self, seq: SchedulerSequence, step: int):
        """Make the coarse sparse-index key for a request prefix."""
        return self._state_checkpoints.make_request_key(seq, step)

    @staticmethod
    def _make_state_checkpoint_node_key(node: Node):
        """Make the canonical sparse-index key for a checkpoint node."""
        return StateCheckpointIndex.make_node_key(node)

    def _index_state_checkpoint(self, node: Node):
        """Add a ready state checkpoint to the sparse SSM index."""
        if node.state_idx < 0 or not node.state_ready:
            raise RuntimeError('Cannot index an unready SSM prefix-cache checkpoint.')
        if not self._is_attached_node(node):
            raise RuntimeError('Cannot index a detached SSM prefix-cache checkpoint node.')
        self._state_checkpoints.add(node)

    def _unindex_state_checkpoint(self, node: Node):
        """Remove a state checkpoint from every sparse-index bucket."""
        return self._state_checkpoints.remove(node)

    def _make_state_checkpoint_match_data_from_node(self, node: Node):
        """Build exact match data by reconstructing a node path.

        This fallback serves direct-node tests and diagnostics.  Production publication uses the producer sequence's
        contiguous buffers instead.
        """
        nodes = tuple(self._get_node_blocks(node))
        if len(nodes) * self.block_size != node.num_matched:
            raise RuntimeError('Cannot publish an SSM checkpoint with a detached ancestor chain.')
        if any(not self._is_attached_node(block_node) for block_node in nodes):
            raise RuntimeError('Cannot publish an SSM checkpoint with a stale ancestor link.')

        token_ids = np.concatenate([block_node.tokens for block_node in nodes])
        multimodal_hashes = make_node_multimodal_identity(nodes, self.block_size)
        blocks = np.fromiter((block_node.block for block_node in nodes), dtype=np.int64, count=len(nodes))
        return freeze_state_checkpoint_match_data(token_ids, multimodal_hashes, blocks)

    def _make_state_checkpoint_match_data_from_seq(self, node: Node, seq: SchedulerSequence):
        """Snapshot exact match data from a checkpoint producer.

        ``BlockTrie.allocate()`` already made the producer cursor authoritative
        for this prefix.  Its recorded private-recompute substitutions turn
        the contiguous logical-block copy back into the canonical trie path
        without rebuilding thousands of Python ``Node`` objects per save.
        """
        step = node.num_matched
        num_blocks = step // self.block_size
        token_ids = seq.history_cache[:step].copy()
        blocks = seq.logical_blocks.get_real_blocks()[:num_blocks].copy()
        for block_id, trie_block in seq.prefix_cache.private_recompute_trie_blocks.items():
            if block_id < num_blocks:
                blocks[block_id] = trie_block
        multimodal_hashes = make_request_multimodal_identity(seq, step)
        start = step - self.block_size
        extra_hashes = self._get_block_extra_hashes(seq, start, step)
        has_complete_identity = len(token_ids) == step and len(blocks) == num_blocks
        has_matching_owner = (seq.adapter_name == node.adapter_name and len(blocks) > 0
                              and blocks[-1] == node.block and self._match_node(node, token_ids[start:], extra_hashes))
        if not has_complete_identity or not has_matching_owner:
            raise RuntimeError('Cannot publish an SSM checkpoint from a mismatched sequence cursor.')
        return freeze_state_checkpoint_match_data(token_ids, multimodal_hashes, blocks)

    # SSM checkpoint reservation and publication lifecycle.

    def reserve_state_checkpoint(self, node: Node):
        """Reserve a state-cache slot owned by a trie node.

        Reusing a ready slot means replacing the checkpoint for the same node, which is allowed only while no restore
        copy has it pinned.  If the shared state pool is full, evict an old unpinned checkpoint without removing the
        trie/KV node itself.
        """
        if not self.requires_state_checkpoint or node.parent is None:
            return -1
        if node.state_ready:
            if node.state_ref_count > 0:
                return -1
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'Replace SSM prefix-cache checkpoint: adapter={node.adapter_name} '
                             f'step={node.num_matched} state_idx={node.state_idx}')
            self._unindex_state_checkpoint(node)
        elif node.state_idx >= 0:
            return -1
        if node.state_idx < 0:
            if self.state_manager.get_num_free_checkpoint() == 0 and self.evict_state_checkpoints(1) == 0:
                return -1
            node.state_idx = self.state_manager.allocate_checkpoint_state()
        node.state_ready = False
        node.state_topology_epoch = node._topology_epoch
        return node.state_idx

    def _clear_pending_state_checkpoint(self, seq: SchedulerSequence):
        """Clear pending checkpoint save metadata from a sequence."""
        prefix_cache = seq.prefix_cache
        prefix_cache.save_state = -1
        prefix_cache.save_step = 0
        prefix_cache.save_is_decode = False
        prefix_cache.save_node = None

    @staticmethod
    def _clear_save_checkpoint_ref(seq: SchedulerSequence):
        """Clear an in-flight producer checkpoint ref from a sequence."""
        prefix_cache = seq.prefix_cache
        prefix_cache.save_state_acquired = False
        prefix_cache.save_acquired_state = -1
        prefix_cache.save_acquired_node = None

    def discard_state_checkpoint_for_seq(self, seq: SchedulerSequence):
        """Discard an unpublished state checkpoint reservation for a sequence.

        Reservations happen before forward.  If the executor fails to produce output, or the sequence is rescheduled
        before the copy is committed, the unready state slot must be released rather than becoming matchable.
        """
        prefix_cache = seq.prefix_cache
        state_idx = prefix_cache.save_state
        node = prefix_cache.save_node
        is_decode = prefix_cache.save_is_decode
        self._clear_pending_state_checkpoint(seq)
        if state_idx < 0:
            return False
        if self._is_unpublished_state_checkpoint_reservation(node, state_idx):
            if is_decode and prefix_cache.decode_state_node is node:
                prefix_cache.decode_state_node = None
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'Discard SSM prefix-cache checkpoint reservation: session_id={seq.session_id} '
                             f'seq_id={seq.seq_id} step={node.num_matched} state_idx={state_idx} '
                             f'is_decode={is_decode}')
            self.release_state_checkpoint(node)
            return True
        return False

    def reserve_state_checkpoint_for_seq(self,
                                         seq: SchedulerSequence,
                                         step: int = None,
                                         is_decode: bool = False):
        """Reserve a state checkpoint slot for an exact trie step.

        SSM prefix hits are valid only when KV blocks and recurrent state refer to the same prefix.  Therefore saves are
        limited to block-aligned, multimodal-safe steps that already have an attached trie node.
        """
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
        if state_idx < 0:
            return -1

        prefix_cache = seq.prefix_cache
        prefix_cache.save_state = state_idx
        prefix_cache.save_step = step
        prefix_cache.save_is_decode = is_decode
        prefix_cache.save_node = node
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'Reserve SSM prefix-cache checkpoint: session_id={seq.session_id} '
                         f'seq_id={seq.seq_id} step={step} state_idx={state_idx} is_decode={is_decode}')
        return state_idx

    def reserve_decode_state_checkpoint_for_seq(self,
                                                seq: SchedulerSequence,
                                                interval: int,
                                                step: int = None):
        """Reserve a bounded decode checkpoint for a sequence.

        Decode checkpoints are opt-in and replaceable: keep at most one ready
        decode checkpoint per sequence so long generations do not consume the
        whole checkpoint budget.  The previous ready checkpoint is released
        only after the new step is proven eligible.
        """
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
        if node.state_idx >= 0:
            return -1

        prefix_cache = seq.prefix_cache
        old_node = prefix_cache.decode_state_node
        if old_node is not None and old_node.state_idx < 0:
            prefix_cache.decode_state_node = None
            old_node = None
        if old_node is not None:
            if self._is_same_ready_decode_state_checkpoint(old_node, step):
                return -1
            if old_node.state_ref_count > 0:
                return -1
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'Release previous decode SSM prefix-cache checkpoint: '
                             f'session_id={seq.session_id} seq_id={seq.seq_id} '
                             f'old_step={old_node.num_matched} old_state_idx={old_node.state_idx} '
                             f'new_step={step}')
            self.release_state_checkpoint(old_node)
            prefix_cache.decode_state_node = None

        return self.reserve_state_checkpoint_for_seq(seq, step=step, is_decode=True)

    def mark_state_checkpoint_ready(self, node: Node, seq: SchedulerSequence | None = None):
        """Mark a node-owned state checkpoint as ready for SSM matching."""
        if node.state_idx < 0:
            raise RuntimeError('Cannot mark an unreserved state checkpoint as ready.')
        if node.state_ref_count != 0:
            raise RuntimeError('Cannot publish a pinned SSM prefix-cache checkpoint.')
        if not self._is_attached_node(node):
            raise RuntimeError('Cannot publish a detached SSM prefix-cache checkpoint node.')
        if node.state_topology_epoch != node._topology_epoch:
            raise RuntimeError('Cannot publish an SSM checkpoint after its trie path changed.')
        if node.state_ready:
            if node.state_match_data is None:
                raise RuntimeError('Cannot republish an invalidated SSM prefix-cache checkpoint.')
            return

        if seq is None:
            match_data = self._make_state_checkpoint_match_data_from_node(node)
        else:
            match_data = self._make_state_checkpoint_match_data_from_seq(node, seq)
        node.state_match_data = match_data
        node.state_ready = True
        node.state_access_time = time.perf_counter()
        try:
            self._index_state_checkpoint(node)
        except Exception:
            # Publication is all-or-nothing.  The caller still owns this
            # unready reservation and can release its state slot safely.
            self._unindex_state_checkpoint(node)
            node.state_ready = False
            node.state_access_time = 0.0
            node.state_match_data = None
            raise

    @staticmethod
    def _is_same_ready_decode_state_checkpoint(node: Node, step: int):
        """Check whether a decode checkpoint for this exact step is ready."""
        return node.num_matched == step and node.state_ready

    def _state_checkpoint_commit_invalid_reason(self, node: Node | None, state_idx: int, save_step: int):
        """Return why a pending checkpoint commit is invalid, or ``None``."""
        if node is None:
            return 'missing node'
        if not self._is_attached_node(node):
            return 'detached node'
        if node.state_topology_epoch != node._topology_epoch:
            return 'trie path changed after reservation'
        if node.state_idx != state_idx:
            return f'state changed: current={node.state_idx}'
        if node.num_matched != save_step:
            return f'step changed: current={node.num_matched}'
        return None

    @staticmethod
    def _is_unpublished_state_checkpoint_reservation(node: Node | None, state_idx: int):
        """Check whether an invalid commit still owns an unready
        reservation."""
        return node is not None and node.state_idx == state_idx and not node.state_ready

    @staticmethod
    def _is_ready_state_checkpoint(node: Node | None, state_idx: int):
        """Check whether a node owns a ready checkpoint state slot."""
        return node is not None and node.state_idx == state_idx and node.state_ready

    @staticmethod
    def _is_ready_state_checkpoint_node(node: Node):
        """Check whether a node has any ready checkpoint state slot."""
        return node.state_idx >= 0 and node.state_ready

    @staticmethod
    def _has_state_checkpoint_ref(node: Node | None, state_idx: int):
        """Check whether a sequence still owns a checkpoint ref on this
        node."""
        return node is not None and node.state_idx == state_idx and node.state_ref_count > 0

    @staticmethod
    def _is_evictable_state_checkpoint(node: Node):
        """Check whether a ready checkpoint may be evicted by LRU."""
        return node.state_idx >= 0 and node.state_ready and node.state_ref_count == 0

    @staticmethod
    def _is_pinned_state_checkpoint(node: Node):
        """Check whether a checkpoint may still be read by an async restore."""
        return node.state_ref_count > 0

    def _release_invalid_state_checkpoint_reservation(self,
                                                      seq: SchedulerSequence,
                                                      node: Node | None,
                                                      state_idx: int,
                                                      is_decode: bool):
        """Release an invalid pending save only if it still owns the slot."""
        if not self._is_unpublished_state_checkpoint_reservation(node, state_idx):
            return
        if is_decode and seq.prefix_cache.decode_state_node is node:
            seq.prefix_cache.decode_state_node = None
        self.release_state_checkpoint(node)

    def _acquire_state_checkpoint_save_for_seq(self, seq: SchedulerSequence, node: Node, state_idx: int):
        """Pin a just-published checkpoint until its producer forward
        completes."""
        prefix_cache = seq.prefix_cache
        if prefix_cache.save_state_acquired:
            raise RuntimeError('SSM prefix-cache save checkpoint already has an in-flight producer ref.')
        if not self._is_ready_state_checkpoint(node, state_idx):
            return False
        node.state_ref_count += 1
        node.state_access_time = time.perf_counter()
        prefix_cache.save_state_acquired = True
        prefix_cache.save_acquired_state = state_idx
        prefix_cache.save_acquired_node = node
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'Acquire SSM prefix-cache save checkpoint: session_id={seq.session_id} '
                         f'seq_id={seq.seq_id} step={node.num_matched} state_idx={state_idx} '
                         f'ref_count={node.state_ref_count}')
        return True

    def commit_state_checkpoint_for_seq(self, seq: SchedulerSequence, acquire_save_ref: bool = False):
        """Publish a sequence state checkpoint.

        When ``acquire_save_ref`` is true, the checkpoint becomes matchable as
        soon as the producer forward is queued, but remains pinned until the
        output/event boundary confirms the stream has passed the save copy.

        Commit validates the remembered node directly.  This matters for decode saves because the sequence may have
        advanced by one sampled token before the output boundary publishes the checkpoint.
        """
        prefix_cache = seq.prefix_cache
        state_idx = prefix_cache.save_state
        save_step = prefix_cache.save_step
        is_decode = prefix_cache.save_is_decode
        node = prefix_cache.save_node
        if state_idx < 0:
            self._clear_pending_state_checkpoint(seq)
            return False

        invalid_reason = self._state_checkpoint_commit_invalid_reason(node, state_idx, save_step)
        if invalid_reason is not None:
            self._clear_pending_state_checkpoint(seq)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'Drop invalid SSM prefix-cache checkpoint commit: session_id={seq.session_id} '
                             f'seq_id={seq.seq_id} step={save_step} state_idx={state_idx} '
                             f'is_decode={is_decode} reason={invalid_reason}')
            self._release_invalid_state_checkpoint_reservation(seq, node, state_idx, is_decode)
            return False

        try:
            self.mark_state_checkpoint_ready(node, seq)
        except Exception:
            self._clear_pending_state_checkpoint(seq)
            self._release_invalid_state_checkpoint_reservation(seq, node, state_idx, is_decode)
            raise
        self._clear_pending_state_checkpoint(seq)
        if is_decode:
            prefix_cache.decode_state_node = node
        if acquire_save_ref:
            self._acquire_state_checkpoint_save_for_seq(seq, node, state_idx)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'Commit SSM prefix-cache checkpoint: session_id={seq.session_id} '
                         f'seq_id={seq.seq_id} step={save_step} state_idx={state_idx} is_decode={is_decode}')
        return True

    def commit_state_checkpoints(self, seqs: list[SchedulerSequence], acquire_save_ref: bool = False):
        """Publish pending sequence state checkpoints."""
        if not self.enable:
            return
        for seq in seqs:
            self.commit_state_checkpoint_for_seq(seq, acquire_save_ref=acquire_save_ref)

    def acquire_state_checkpoint_restore_for_seq(self, seq: SchedulerSequence):
        """Pin a matched state checkpoint until its restore copy has
        completed."""
        prefix_cache = seq.prefix_cache
        if prefix_cache.restore_state < 0 or prefix_cache.restore_state_acquired:
            return False
        node = prefix_cache.restore_node
        if not self._is_ready_state_checkpoint(node, prefix_cache.restore_state):
            return False
        node.state_ref_count += 1
        node.state_access_time = time.perf_counter()
        prefix_cache.restore_state_acquired = True
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'Acquire SSM prefix-cache restore checkpoint: session_id={seq.session_id} '
                         f'seq_id={seq.seq_id} step={node.num_matched} state_idx={node.state_idx} '
                         f'ref_count={node.state_ref_count}')
        return True

    def acquire_state_checkpoint_restores(self, seqs: list[SchedulerSequence]):
        """Pin matched state checkpoints for a batch."""
        for seq in seqs:
            self.acquire_state_checkpoint_restore_for_seq(seq)

    @staticmethod
    def _release_state_checkpoint_ref(node: Node | None, state_idx: int, err_msg: str):
        """Release one checkpoint ref held by a sequence."""
        if not BlockTrie._has_state_checkpoint_ref(node, state_idx):
            BlockTrie._warn_unexpected_state(f'{err_msg} state_idx={state_idx}')
            raise RuntimeError(err_msg)
        node.state_ref_count -= 1
        return node

    def release_state_checkpoint_restore_for_seq(self, seq: SchedulerSequence):
        """Release a state checkpoint pinned for restore."""
        prefix_cache = seq.prefix_cache
        if not prefix_cache.restore_state_acquired:
            return False
        node = self._release_state_checkpoint_ref(
            prefix_cache.restore_node,
            prefix_cache.restore_state,
            'Acquired SSM prefix-cache restore checkpoint lost its node reference.',
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'Release SSM prefix-cache restore checkpoint: session_id={seq.session_id} '
                         f'seq_id={seq.seq_id} step={node.num_matched} state_idx={node.state_idx} '
                         f'ref_count={node.state_ref_count}')
        prefix_cache.restore_state = -1
        prefix_cache.restore_node = None
        prefix_cache.restore_state_acquired = False
        return True

    def release_state_checkpoint_restores(self, seqs: list[SchedulerSequence]):
        """Release state checkpoints pinned for a batch restore."""
        if not self.enable:
            return
        for seq in seqs:
            self.release_state_checkpoint_restore_for_seq(seq)

    def release_state_checkpoint_save_for_seq(self, seq: SchedulerSequence):
        """Release a checkpoint pinned for its producer save copy."""
        prefix_cache = seq.prefix_cache
        if not prefix_cache.save_state_acquired:
            return False
        node = self._release_state_checkpoint_ref(
            prefix_cache.save_acquired_node, prefix_cache.save_acquired_state,
            'Acquired SSM prefix-cache save checkpoint lost its node reference.')
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'Release SSM prefix-cache save checkpoint: session_id={seq.session_id} '
                         f'seq_id={seq.seq_id} step={node.num_matched} state_idx={node.state_idx} '
                         f'ref_count={node.state_ref_count}')
        self._clear_save_checkpoint_ref(seq)
        return True

    def release_state_checkpoint_saves(self, seqs: list[SchedulerSequence]):
        """Release producer refs held by a batch of saved checkpoints."""
        if not self.enable:
            return
        for seq in seqs:
            self.release_state_checkpoint_save_for_seq(seq)

    def release_state_checkpoint(self, node: Node):
        """Release a node-owned state checkpoint while keeping KV ownership."""
        if node.state_ref_count > 0:
            raise RuntimeError('Cannot release a pinned SSM prefix-cache checkpoint.')
        if node.state_idx < 0:
            if node.state_ready:
                self._warn_unexpected_state(
                    f'ready SSM checkpoint has no state slot: adapter={node.adapter_name} '
                    f'step={node.num_matched}')
                self._unindex_state_checkpoint(node)
                node.state_ready = False
                node.state_ref_count = 0
                node.state_access_time = 0.0
            node.state_match_data = None
            node.state_topology_epoch = -1
            return
        if node.state_ready:
            self._unindex_state_checkpoint(node)
        self.state_manager.free_checkpoint_state(node.state_idx)
        node.state_idx = -1
        node.state_ready = False
        node.state_ref_count = 0
        node.state_access_time = 0.0
        node.state_match_data = None
        node.state_topology_epoch = -1

    def evict_state_checkpoints(self, max_num_states: int):
        """Evict ready SSM state checkpoints without removing KV trie nodes."""
        if not self.requires_state_checkpoint or max_num_states <= 0:
            return 0

        candidates = []
        for node in self._state_checkpoints.unique_nodes():
            if self._is_evictable_state_checkpoint(node):
                candidates.append((node.state_access_time, node))
        heapq.heapify(candidates)

        evicted = 0
        while len(candidates) > 0 and evicted < max_num_states:
            _, node = heapq.heappop(candidates)
            if not self._is_evictable_state_checkpoint(node):
                continue
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'Evict SSM prefix-cache checkpoint: adapter={node.adapter_name} '
                             f'step={node.num_matched} state_idx={node.state_idx}')
            self.release_state_checkpoint(node)
            evicted += 1
        return evicted

    # Trie attachment and leaf-index helpers.

    def _get_state_checkpoint_node_for_seq(self, seq: SchedulerSequence, step: int):
        """Get the trie node that exactly represents a sequence checkpoint
        step."""
        node = seq.prefix_cache.last_shared_node
        while node is not None and node.num_matched > step:
            node = node.parent
        if node is None or node.parent is None or node.num_matched != step:
            return None
        return node

    @staticmethod
    def _is_attached_node(node: Node):
        """Check whether a node is still attached to the trie."""
        parent = node.parent
        return parent is not None and parent.children.get(node.hash_key) is node

    def _has_current_state_checkpoint_path(self, node: Node):
        """Check a cached checkpoint path through its invalidation contract."""
        match_data = node.state_match_data
        if match_data is None or len(match_data.blocks) == 0:
            return False
        return (len(match_data.blocks) * self.block_size == node.num_matched and self._is_attached_node(node))

    @staticmethod
    def _is_attached_leaf(node: Node):
        """Check whether a node is a current attached trie leaf."""
        return BlockTrie._is_attached_node(node) and len(node.children) == 0

    @staticmethod
    def _is_evict_candidate_leaf(node: Node):
        """Check whether a leaf-set entry can be considered by KV eviction."""
        return (node.block >= 0 and len(node.children) == 0
                and (node.parent is None or BlockTrie._is_attached_node(node)))

    def _is_attached_cursor(self, node: Node):
        """Check whether a sequence cursor still reaches the adapter root."""
        if node.parent is None:
            return node.block < 0 and self._roots.get(node.adapter_name) is node
        if node.state_match_data is not None:
            return self._has_current_state_checkpoint_path(node)
        nodes = self._get_node_blocks(node)
        if len(nodes) * self.block_size != node.num_matched:
            return False
        return all(self._is_attached_node(node) for node in nodes)

    def _get_node_blocks(self, node: Node):
        """Get trie nodes from root to a target node."""
        nodes = []
        while node is not None and node.parent is not None:
            nodes.append(node)
            node = node.parent
        nodes.reverse()
        return nodes

    def _drop_stale_state_checkpoint_index_entry(self, node: Node, key, reason: str):
        """Remove a bad sparse-index entry without releasing a valid node."""
        removed = self._state_checkpoints.remove_entry(node, key)
        if removed and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'Drop stale SSM prefix-cache checkpoint index entry: adapter={key[0]} '
                         f'step={key[1]} node_adapter={node.adapter_name} '
                         f'node_step={node.num_matched} state_idx={node.state_idx} reason={reason}')
        return removed

    def _release_stale_state_checkpoint_candidate(self, node: Node, reason: str):
        """Release a globally stale checkpoint candidate if it is unpinned."""
        if self._is_pinned_state_checkpoint(node):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'Skip pinned stale SSM prefix-cache checkpoint candidate: '
                             f'adapter={node.adapter_name} step={node.num_matched} '
                             f'state_idx={node.state_idx} ref_count={node.state_ref_count} '
                             f'reason={reason}')
            return False

        state_idx = node.state_idx
        state_ready = node.state_ready
        self._unindex_state_checkpoint(node)
        if state_idx >= 0:
            self.state_manager.free_checkpoint_state(state_idx)
        node.state_idx = -1
        node.state_ready = False
        node.state_ref_count = 0
        node.state_access_time = 0.0
        node.state_match_data = None
        node.state_topology_epoch = -1
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'Release stale SSM prefix-cache checkpoint candidate: '
                         f'adapter={node.adapter_name} step={node.num_matched} '
                         f'state_idx={state_idx} was_ready={state_ready} reason={reason}')
        return state_idx >= 0 or state_ready

    def _handle_state_checkpoint_rejection(self, seq: SchedulerSequence, node: Node, key, match_result):
        """Clean up a rejected sparse candidate according to its status."""
        if match_result.status == StateCheckpointVerifyStatus.STALE_INDEX_ENTRY:
            self._drop_stale_state_checkpoint_index_entry(node, key, match_result.reason)
        elif match_result.status == StateCheckpointVerifyStatus.STALE_CHECKPOINT:
            self._release_stale_state_checkpoint_candidate(node, match_result.reason)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'Reject SSM prefix-cache checkpoint candidate: '
                         f'session_id={seq.session_id} seq_id={seq.seq_id} step={node.num_matched} '
                         f'state_idx={node.state_idx} status={match_result.status.name} '
                         f'reason={match_result.reason}')

    def _has_required_recompute_suffix(self, seq: SchedulerSequence, node: Node, recompute_blocks: int):
        """Check that an exact checkpoint has enough cached MTP overlap."""
        step = node.num_matched
        raw_match_step = self._find_raw_block_match_step(seq, node)
        required_step = step + recompute_blocks * self.block_size
        if raw_match_step >= required_step:
            return True, raw_match_step

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'Reject SSM prefix-cache checkpoint without MTP overlap: '
                         f'session_id={seq.session_id} seq_id={seq.seq_id} step={step} '
                         f'raw_match_step={raw_match_step} required_step={required_step}')
        return False, raw_match_step

    def _apply_state_checkpoint_hit(self,
                                    seq: SchedulerSequence,
                                    node: Node,
                                    matched_blocks: np.ndarray,
                                    initial_step: int,
                                    raw_match_step: int):
        """Acquire a verified checkpoint prefix and update sequence state."""
        step = node.num_matched
        new_blocks = matched_blocks[initial_step // self.block_size:]
        self.allocator.update_access_time(new_blocks)
        self.allocator.add_ref_count(new_blocks, 1)
        seq.logical_blocks.append(new_blocks)
        seq.set_step(step)
        self._append_state_checkpoint_routed_experts(seq, node, initial_step)

        prefix_cache = seq.prefix_cache
        prefix_cache.restore_state = node.state_idx
        prefix_cache.restore_node = node
        prefix_cache.last_shared_node = node
        if raw_match_step >= 0:
            self._set_private_recompute_range(seq, step, raw_match_step)

        self._record_match_stats(seq,
                                 query_tokens=seq.num_all_ids - initial_step,
                                 hit_tokens=step - initial_step)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'SSM prefix-cache hit: session_id={seq.session_id} seq_id={seq.seq_id} '
                         f'init_step={initial_step} matched_step={step} state_idx={node.state_idx}')

    def _match_state_checkpoint(self, seq: SchedulerSequence):
        """Match SSM prefixes through sparse ready-checkpoint lookup.

        KV-only reuse is unsafe for SSM models, so this path reports a hit only if a ready recurrent-state checkpoint
        exists at the exact matched step.
        """
        self._clear_private_recompute_range(seq)
        seq.prefix_cache.restore_state = -1
        seq.prefix_cache.restore_node = None

        init_curr = seq.prefix_cache.last_shared_node
        if init_curr is None:
            init_curr = self.get_root(seq.adapter_name)
        init_num_matched = init_curr.num_matched

        recompute_blocks = max(0, seq.prefix_cache.match_recompute_blocks)
        raw_match_step = -1
        max_step = ((seq.num_valid_ids - 1) // self.block_size) * self.block_size
        max_step = seq.clamp_prefix_cache_match_step(max_step)
        candidate_steps = self._state_checkpoints.candidate_steps(seq.adapter_name, init_num_matched, max_step)
        for step in candidate_steps:
            if seq.clamp_prefix_cache_match_step(step) != step:
                continue
            key = self._make_state_checkpoint_lookup_key(seq, step)
            for node in self._state_checkpoints.candidates(key):
                path_is_current = self._has_current_state_checkpoint_path(node)
                match_result = self._state_checkpoints.verify(seq, node, key, path_is_current)
                if match_result.status != StateCheckpointVerifyStatus.HIT:
                    self._handle_state_checkpoint_rejection(seq, node, key, match_result)
                    continue

                if recompute_blocks > 0:
                    has_recompute_suffix, raw_match_step = self._has_required_recompute_suffix(
                        seq, node, recompute_blocks)
                    if not has_recompute_suffix:
                        continue

                self._apply_state_checkpoint_hit(seq, node, match_result.matched_blocks, init_num_matched,
                                                 raw_match_step)
                return

        seq.prefix_cache.last_shared_node = init_curr
        self._clear_private_recompute_range(seq)
        self._record_match_stats(seq, query_tokens=seq.num_all_ids - init_num_matched)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'SSM prefix-cache miss: session_id={seq.session_id} seq_id={seq.seq_id} '
                         f'init_step={init_num_matched} raw_match_step={raw_match_step} '
                         f'max_step={max_step} ready_steps={self._state_checkpoints.num_steps(seq.adapter_name)}')

    def match(self, seq: SchedulerSequence):
        """Tentatively match reusable prefix blocks for a sequence.

        This method mutates sequence state before scheduler admission is final:
        it may append shared logical blocks, advance ``seq.num_history_ids``,
        record ``match_start_step``, replay routed experts, update stats, and
        set SSM restore metadata.  Callers must rollback the sequence if later
        resource admission rejects the request.

        Text/VLM models walk the trie block by block.  SSM models delegate to
        sparse checkpoint matching because a KV block match without the exact
        recurrent-state snapshot is a miss.
        """
        if not self.enable:
            return
        self._clear_private_recompute_range(seq)
        seq.prefix_cache.match_start_step = seq.num_history_ids
        seq.prefix_cache.restore_state = -1
        seq.prefix_cache.restore_node = None
        if self.requires_state_checkpoint:
            self._match_state_checkpoint(seq)
            return

        block_size = self.block_size
        matched_blocks = []

        curr: Node = seq.prefix_cache.last_shared_node
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
            if seq.return_routed_experts and child.routed_experts is None:
                break

            matched_nodes.append(child)
            __match_success(child)

        def __clamp_match_step(match_step: int):
            nonlocal curr, num_matched, matched_blocks, matched_nodes
            match_step = max(init_num_matched, match_step)
            if match_step >= num_matched:
                return
            # If a candidate hit stopped inside a multimodal span, drop any
            # blocks beyond the clamped safe boundary before acquiring refs.
            keep = (match_step - init_num_matched) // block_size
            matched_nodes = matched_nodes[:keep]
            matched_blocks = matched_blocks[:keep]
            if keep > 0:
                curr = matched_nodes[-1]
                num_matched = curr.num_matched
            else:
                curr = init_curr
                num_matched = init_num_matched

        max_match_step = seq.get_prefix_cache_max_match_step()
        candidate_num_matched = num_matched
        raw_num_matched = self._find_raw_block_match_step(seq, init_curr)
        effective_num_matched = seq.clamp_prefix_cache_match_step(min(candidate_num_matched, max_match_step))
        __clamp_match_step(effective_num_matched)
        self._set_private_recompute_range(seq, num_matched, raw_num_matched)

        if len(matched_blocks) > 0:
            matched_blocks = np.array(matched_blocks)
            self.allocator.update_access_time(matched_blocks)
            self.allocator.add_ref_count(matched_blocks, 1)
            seq.logical_blocks.append(matched_blocks)
            seq.set_step(num_matched)
            self._append_matched_routed_experts(seq, matched_nodes, init_num_matched)
            if self.requires_state_checkpoint:
                seq.prefix_cache.restore_state = curr.state_idx

        # record prefix hit
        self._record_match_stats(seq,
                                 query_tokens=seq.num_all_ids - init_num_matched,
                                 hit_tokens=num_matched - init_num_matched)

        seq.prefix_cache.last_shared_node = curr
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'Prefix-cache match: session_id={seq.session_id} seq_id={seq.seq_id} '
                         f'init_step={init_num_matched} matched_step={num_matched} '
                         f'candidate_step={raw_num_matched} '
                         f'clamped={effective_num_matched != raw_num_matched}')

    def allocate(self, seq: SchedulerSequence):
        """Attach newly allocated full blocks to the prefix-cache trie.

        Allocation starts from ``seq.prefix_cache.last_shared_node`` when that
        cursor still reaches the current trie.  Existing identical children are
        deduplicated back to the trie-owned block, except for the private
        recompute-overlap window where the sequence must keep fresh writable KV.
        New nodes take one trie-owned allocator ref.
        """
        if not self.enable:
            return

        block_size = self.block_size
        logical_blocks = seq.logical_blocks
        node: Node = seq.prefix_cache.last_shared_node
        if node is None:
            node = self.get_root(seq.adapter_name)
            seq.prefix_cache.last_shared_node = node
        elif not self._is_attached_cursor(node):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'Reset detached prefix-cache sequence cursor: session_id={seq.session_id} '
                             f'seq_id={seq.seq_id} adapter={seq.adapter_name} '
                             f'cursor_step={node.num_matched}')
            node = self.get_root(seq.adapter_name)
            seq.prefix_cache.last_shared_node = node

        num_matched = node.num_matched
        num_valid_ids = seq.num_valid_ids
        if seq.kv_token_limit is not None:
            num_valid_ids = min(num_valid_ids, seq.kv_token_limit)

        if num_matched + block_size > num_valid_ids:
            self._clear_private_recompute_range(seq)
            return

        if len(node.children) == 0 and node.parent is not None:
            self.leaves.discard(node)

        block_id = num_matched // block_size
        private_trie_blocks = seq.prefix_cache.private_recompute_trie_blocks
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
                if self._is_private_recompute_step(seq, start):
                    # This block was deliberately dropped from a prefix match so
                    # the target forward can regenerate hidden-state bridge data.
                    # Traverse the identity path, but keep the fresh writable
                    # sequence block instead of substituting the shared trie one.
                    private_trie_blocks[block_id] = child.block
                    node = child
                    num_matched += block_size
                    block_id += 1
                    continue
                # Another sequence inserted the same key before us.  Reuse the
                # trie-owned block and release this sequence's duplicate block.
                private_trie_blocks.pop(block_id, None)
                node = child
                self._try_cache_node_routed_experts(node, seq, start, end)
                if block != node.block:
                    free_blocks.append(block)
                    logical_blocks[block_id] = node.block
                    blocks.append(node.block)
            else:
                private_trie_blocks.pop(block_id, None)
                routed_experts = self._get_routed_experts_for_range(seq, start, end)
                node = Node(hash_key=hash_key,
                            block=block,
                            tokens=curr_tokens,
                            num_matched=num_matched + block_size,
                            extra_hashes=extra_hashes,
                            routed_experts=routed_experts,
                            adapter_name=seq.adapter_name)
                node.parent = parent
                blocks.append(node.block)
            num_matched += block_size
            block_id += 1

        seq.prefix_cache.last_shared_node = node
        if node.parent is not None and len(node.children) == 0:
            # ignore root
            self.leaves.add(node)
        if len(blocks) > 0:
            self.allocator.add_ref_count(np.array(blocks), 1)
        if len(free_blocks) > 0:
            self.allocator.free(np.array(free_blocks))
        self._clear_private_recompute_range(seq)

    def evict(self, max_num_blocks: int):
        """Evict trie-owned KV leaf blocks.

        ``self.leaves`` is an auxiliary candidate index, not the source of
        truth.  Each candidate is rechecked against the parent/children chain,
        allocator refcount, and checkpoint pin state before removing it.  When
        a leaf is removed, its parent can become the next leaf candidate.
        """
        if not self.enable:
            return 0

        def __remove_leaf(leaves, evicted_blocks):
            while len(leaves) > 0:
                _, leaf = heapq.heappop(leaves)
                if leaf not in self.leaves:
                    continue
                if not self._is_evict_candidate_leaf(leaf):
                    self.leaves.discard(leaf)
                    continue
                if self._is_pinned_state_checkpoint(leaf):
                    continue
                if int(self.allocator.get_ref_count(leaf.block)) != 1:
                    continue
                break
            else:
                return False, None

            evicted_blocks.append(leaf.block)
            self.release_state_checkpoint(leaf)
            parent = leaf.parent
            if parent is not None:
                leaf.parent = None
            self.leaves.discard(leaf)
            return True, parent

        def __add_leaf(leaves, parent):
            if not self._is_attached_leaf(parent):
                return
            if parent in self.leaves:
                return
            self.leaves.add(parent)
            if self.allocator.get_ref_count(parent.block) == 1:
                access_time = self.allocator.get_access_time(parent.block)
                heapq.heappush(leaves, (access_time, parent))

        if len(self.leaves) == 0:
            return 0

        evicted_blocks = []
        old_leaf_count = len(self.leaves)
        leaves = list(leaf for leaf in self.leaves if self._is_evict_candidate_leaf(leaf))
        if len(leaves) != len(self.leaves):
            self.leaves.intersection_update(leaves)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'Dropped stale prefix-cache leaf candidates before eviction: '
                             f'old_count={old_leaf_count} new_count={len(leaves)}')
        if len(leaves) == 0:
            return 0

        # filter ref-cnt == 1 (trie own one block ref)
        leaf_blocks = np.array(list(leaf.block for leaf in leaves))
        ref_cnt = self.allocator.get_ref_count(leaf_blocks)
        indices = (ref_cnt == 1).nonzero()[0]
        if len(indices) == 0:
            return 0

        # make heap
        leaves = list(leaves[i] for i in indices)
        access_times = self.allocator.get_access_time(leaf_blocks)
        access_times = list(access_times[i] for i in indices)
        leaves = list(zip(access_times, leaves))
        heapq.heapify(leaves)

        while len(leaves) > 0 and len(evicted_blocks) < max_num_blocks:
            removed, parent = __remove_leaf(leaves, evicted_blocks)
            if not removed:
                break
            if parent is None or parent.parent is None:
                # ignore root
                continue
            if len(parent.children) == 0:
                __add_leaf(leaves, parent)

        if len(evicted_blocks) == 0:
            return 0
        self.allocator.free(np.array(evicted_blocks))

        return len(evicted_blocks)
