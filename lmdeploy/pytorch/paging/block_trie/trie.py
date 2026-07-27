# Copyright (c) OpenMMLab. All rights reserved.
"""Prefix-cache trie topology, matching, and KV ownership.

``BlockTrie`` owns reusable prefix identity and optional routed-expert replay
data. It delegates trie KV references and leaf eviction to
``KVBlockLifecycle`` and exposes SSM checkpoint slots through the grouped
``state_checkpoints`` lifecycle API. Read
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
4. SSM matching cannot reuse KV alone. It uses sparse published-checkpoint lookup,
   verifies the full ancestor chain, then asks ``ModelAgent`` to copy the
   frozen checkpoint state into the request runtime state on the forward stream.
5. SSM checkpoint saves are reserved through ``state_checkpoints``, copied by
   ``ModelAgent`` after forward, and published by ``EngineLoop`` once the
   producer forward is queued.
   Producer/restore pins protect checkpoint slots across async stream-ordering
   windows.

SSM checkpoint detail:

* ``seq.prefix_cache.last_shared_node`` stores the deepest trie node already
  shared by the sequence.  ``match()`` writes it, rollback/free clears it, and
  ``allocate()`` continues inserting newly computed full blocks from it.
* ``StateManager`` owns one state-cache pool split by role: active requests use
  runtime slots stored on ``seq.logical_state``; prefix-cache checkpoints use
  slots stored in a trie's optional ``Node.state_checkpoint`` record. A trie
  node may own KV only, KV plus an unpublished checkpoint reservation, or KV
  plus a published checkpoint.
* Saving a checkpoint starts from an already-attached block-aligned trie node.
  ``state_checkpoints.reserve_save()`` records a ``pending_save`` reservation
  on ``seq.prefix_cache``.  Prefill and
  long-context chunks save at the produced chunk end; decode saves are optional
  and bounded by ``prefix_cache_decode_state_interval``.
* ``InputsMaker`` converts those pending saves into compact host integer
  src/dst pairs.  ``ModelAgent`` then copies ``runtime_state -> checkpoint`` on
  the model forward stream after the model has produced the new SSM state.
  ``EngineLoop`` calls ``state_checkpoints.publish_save()`` after the forward
  is queued; only then is the checkpoint published into the sparse
  match index. The producing forward holds a producer pin
  until the output/event boundary, so this early visibility cannot make the
  destination slot evictable before the save copy reaches the forward stream.
  Abandoned reservations are discarded.
* Matching a SSM prefix never walks KV blocks as the source of truth.
  ``_match_state_checkpoint_prefix()`` searches published checkpoint steps, filters by
  ``(adapter, step, last_block_hash)``, then proves the complete prefix against
  immutable exact-match metadata built when the checkpoint is published.  A
  hit appends trie-owned KV blocks, advances ``seq.num_history_ids``, records a
  selected ``restore``, and may replay routed experts.
* Restore is two-phase. The scheduler/input maker pins the published checkpoint
  by incrementing its pin count. ``ModelAgent`` copies
  ``checkpoint -> runtime_state`` before the suffix forward.  ``EngineLoop``
  releases the pin once the copy has been queued, so LRU eviction cannot reuse
  the checkpoint source slot too early.
* Checkpoint eviction is state-only LRU over published, unpinned nodes. KV leaf
  eviction also releases any checkpoint owned by that leaf.  A KV match without
  an exact published SSM checkpoint is intentionally a miss.
"""

from dataclasses import dataclass

import numpy as np

from lmdeploy.pytorch.messages import SchedulerSequence
from lmdeploy.pytorch.prefix_cache_state import PrefixCacheExtraHashes
from lmdeploy.utils import get_logger

from ..block_manager.base_block_manager import LogicalAllocator
from ..state_manager import StateManager
from .checkpoint import (
    StateCheckpointIndex,
    StateCheckpointVerifyStatus,
    freeze_state_checkpoint_match_data,
    make_request_multimodal_identity,
)
from .checkpoint_lifecycle import StateCheckpointLifecycle
from .kv_lifecycle import KVBlockLifecycle
from .node import Node

logger = get_logger('lmdeploy')


@dataclass
class PrefixCacheStats:
    """Prefix caching stats."""
    num_query_tokens: int = 0
    num_hit_tokens: int = 0

    def reset(self):
        self.num_query_tokens = 0
        self.num_hit_tokens = 0

    def snapshot(self):
        """Snapshot stats for tentative-match rollback."""
        return PrefixCacheStats(num_query_tokens=self.num_query_tokens, num_hit_tokens=self.num_hit_tokens)

    def restore(self, snapshot: 'PrefixCacheStats | None'):
        """Restore a tentative-match snapshot."""
        if snapshot is None:
            return
        self.num_query_tokens = snapshot.num_query_tokens
        self.num_hit_tokens = snapshot.num_hit_tokens

    def hit_rate(self):
        return 0.0 if self.num_query_tokens <= 0 else float(self.num_hit_tokens) / self.num_query_tokens


class BlockTrie:
    """Prefix-cache facade for trie KV reuse and SSM state reuse.

    Public scheduling flow stays small: ``match(seq)`` tentatively reuses a
    prefix, the scheduler admits resources, then ``allocate(seq)`` attaches the
    newly computed full blocks.  ``evict()`` frees trie-owned KV leaves under
    allocator pressure.

    Internally this facade coordinates several related owners:

    * adapter-partitioned trie roots and multimodal-aware block identity;
    * KV reference transactions and leaf eviction in ``KVBlockLifecycle``;
    * optional SSM checkpoint reserve/publish/restore/save lifecycle;
    * sparse published-checkpoint lookup for SSM prefix hits;
    * best-effort routed-expert replay data;
    * prefix-cache stats used by scheduler rollback.
    """

    def __init__(self,
                 *,
                 allocator: LogicalAllocator,
                 block_size: int,
                 enabled: bool,
                 checkpoint_state_manager: StateManager | None = None):
        self.allocator = allocator
        self.block_size = block_size
        self.enable = enabled
        self.requires_state_checkpoint = checkpoint_state_manager is not None

        # caches with different adapter should not be shared.
        self._roots: dict[str, Node] = dict()
        # SSM checkpoints are sparse. The trie still owns KV blocks, but
        # published recurrent-state snapshots are indexed only at selected
        # exact steps.
        self._checkpoint_index = StateCheckpointIndex(self.block_size, self._make_block_key)
        self._state_checkpoints = StateCheckpointLifecycle(
            prefix_cache_enabled=self.enable,
            state_checkpoints_enabled=self.requires_state_checkpoint,
            block_size=self.block_size,
            state_manager=checkpoint_state_manager,
            index=self._checkpoint_index,
            make_sequence_match_data=self._make_state_checkpoint_match_data_from_seq,
        )
        self._kv_lifecycle = KVBlockLifecycle(self.allocator, self._state_checkpoints)
        self.stats = PrefixCacheStats()

    @property
    def leaves(self):
        """Expose KV leaf candidates for compatibility and diagnostics."""
        return self._kv_lifecycle.leaves

    @property
    def state_checkpoints(self) -> StateCheckpointLifecycle:
        """Expose the state-checkpoint lifecycle as a grouped public API."""
        return self._state_checkpoints

    def _record_match_stats(self, seq: SchedulerSequence, query_tokens: int, hit_tokens: int = 0):
        """Record a user-visible prefix-cache match attempt."""
        if seq.prefix_cache.suppress_match_stats:
            return
        self.stats.num_query_tokens += query_tokens
        self.stats.num_hit_tokens += hit_tokens

    # Trie keying and raw block matching helpers.

    def _get_or_create_root(self, adapter_name: str):
        """Return the adapter-partitioned root, creating it on first use."""
        if adapter_name not in self._roots:
            self._roots[adapter_name] = Node(-1, -1, None, adapter_name=adapter_name)
        return self._roots[adapter_name]

    @staticmethod
    def _get_block_extra_hashes(seq: SchedulerSequence, start: int, end: int) -> PrefixCacheExtraHashes:
        """Get multimodal identity entries that belong in a block key."""
        return seq.get_prefix_cache_extra_hashes(start, end)

    @staticmethod
    def _make_block_key(tokens: np.ndarray, extra_hashes: PrefixCacheExtraHashes):
        """Make the trie lookup key from tokens plus multimodal identity."""
        return hash(('random', tuple(tokens), extra_hashes))

    @staticmethod
    def _node_matches_block(node: Node, tokens: np.ndarray, extra_hashes: PrefixCacheExtraHashes):
        """Check the exact key payload after the hash-table lookup."""
        return np.array_equal(tokens, node.tokens) and extra_hashes == node.extra_hashes

    def _find_deepest_block_match_step(self, seq: SchedulerSequence, start_node: Node):
        """Find the deepest KV trie match without acquiring block refs."""
        block_size = self.block_size
        node = start_node
        num_matched = node.num_matched
        while num_matched + block_size < seq.num_valid_ids:
            start = num_matched
            end = num_matched + block_size
            curr_tokens = seq.history_cache[start:end]
            extra_hashes = self._get_block_extra_hashes(seq, start, end)

            key = self._make_block_key(curr_tokens, extra_hashes)
            if key not in node.children:
                break

            child = node.children[key]
            if not self._node_matches_block(child, curr_tokens, extra_hashes):
                break

            node = child
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
        nodes = node.path_from_root()
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

    def _make_state_checkpoint_match_data_from_seq(self, node: Node, seq: SchedulerSequence):
        """Snapshot exact match data from a checkpoint producer.

        ``BlockTrie.allocate()`` already made the producer cursor authoritative
        for this prefix. Its recompute-overlap substitutions turn the
        contiguous logical-block copy back into the canonical trie path
        without rebuilding thousands of Python ``Node`` objects per save.
        """
        step = node.num_matched
        num_blocks = step // self.block_size
        token_ids = seq.history_cache[:step].copy()
        blocks = seq.logical_blocks.get_real_blocks()[:num_blocks].copy()
        seq.prefix_cache.recompute_overlap.rewrite_to_canonical_path(blocks)
        multimodal_hashes = make_request_multimodal_identity(seq, step)
        start = step - self.block_size
        extra_hashes = self._get_block_extra_hashes(seq, start, step)
        has_complete_identity = len(token_ids) == step and len(blocks) == num_blocks
        has_matching_owner = (seq.adapter_name == node.adapter_name and len(blocks) > 0
                              and blocks[-1] == node.block
                              and self._node_matches_block(node, token_ids[start:], extra_hashes))
        if not has_complete_identity or not has_matching_owner:
            raise RuntimeError('Cannot publish an SSM checkpoint from a mismatched sequence cursor.')
        return freeze_state_checkpoint_match_data(token_ids, multimodal_hashes, blocks)

    def _checkpoint_path_is_current(self, node: Node):
        """Check cached checkpoint data on an attached monotonic path."""
        checkpoint = node.state_checkpoint
        if checkpoint is None:
            return False
        match_data = checkpoint.exact_match_data
        if match_data is None or len(match_data.blocks) == 0:
            return False
        return len(match_data.blocks) * self.block_size == node.num_matched and node.is_attached()

    def _cursor_path_is_current(self, node: Node):
        """Check a cursor under the monotonic attach/detach contract."""
        if node.parent is None:
            return node.block < 0 and self._roots.get(node.adapter_name) is node
        return node.is_attached()

    def _handle_state_checkpoint_rejection(self, seq: SchedulerSequence, node: Node, key, match_result):
        """Clean up a rejected sparse candidate according to its status."""
        if match_result.status == StateCheckpointVerifyStatus.STALE_INDEX_ENTRY:
            self.state_checkpoints.discard_stale_index_entry(node, key, match_result.reason)
        elif match_result.status == StateCheckpointVerifyStatus.STALE_CHECKPOINT:
            self.state_checkpoints.release_stale_checkpoint(node, match_result.reason)
        checkpoint = node.state_checkpoint
        state_idx = -1 if checkpoint is None else checkpoint.slot
        logger.debug('Reject SSM prefix-cache checkpoint candidate: session_id=%s seq_id=%s step=%s '
                     'state_idx=%s status=%s reason=%s', seq.session_id, seq.seq_id, node.num_matched, state_idx,
                     match_result.status.name, match_result.reason)

    def _find_recompute_overlap_end(self, seq: SchedulerSequence, node: Node, recompute_blocks: int):
        """Return the cached overlap end, or ``-1`` when it is too short."""
        step = node.num_matched
        cached_end_step = self._find_deepest_block_match_step(seq, node)
        required_step = step + recompute_blocks * self.block_size
        if cached_end_step >= required_step:
            return cached_end_step

        logger.debug('Reject SSM prefix-cache checkpoint without MTP overlap: session_id=%s seq_id=%s step=%s '
                     'cached_end_step=%s required_step=%s', seq.session_id, seq.seq_id, step, cached_end_step,
                     required_step)
        return -1

    def _apply_state_checkpoint_hit(self,
                                    seq: SchedulerSequence,
                                    node: Node,
                                    matched_blocks: np.ndarray,
                                    initial_step: int,
                                    overlap_end_step: int):
        """Apply a verified checkpoint hit to sequence state."""
        step = node.num_matched
        new_blocks = matched_blocks[initial_step // self.block_size:]
        self.allocator.update_access_time(new_blocks)
        self.allocator.add_ref_count(new_blocks, 1)
        seq.logical_blocks.append(new_blocks)
        seq.set_step(step)
        self._append_state_checkpoint_routed_experts(seq, node, initial_step)

        prefix_cache = seq.prefix_cache
        checkpoint = node.state_checkpoint
        prefix_cache.restore.select(checkpoint.slot, node)
        prefix_cache.last_shared_node = node
        if overlap_end_step >= 0:
            prefix_cache.recompute_overlap.set_fresh_block_range(step // self.block_size,
                                                                 overlap_end_step // self.block_size)

        self._record_match_stats(seq,
                                 query_tokens=seq.num_all_ids - initial_step,
                                 hit_tokens=step - initial_step)
        logger.debug('SSM prefix-cache hit: session_id=%s seq_id=%s initial_step=%s matched_step=%s state_idx=%s',
                     seq.session_id, seq.seq_id, initial_step, step, checkpoint.slot)

    def _match_state_checkpoint_prefix(self, seq: SchedulerSequence):
        """Match SSM prefixes through sparse published-checkpoint lookup.

        KV-only reuse is unsafe for SSM models, so this path reports a hit only if a published recurrent-state
        checkpoint exists at the exact matched step.
        """
        initial_cursor = seq.prefix_cache.last_shared_node
        if initial_cursor is None:
            initial_cursor = self._get_or_create_root(seq.adapter_name)
        initial_step = initial_cursor.num_matched

        recompute_blocks = max(0, seq.prefix_cache.recompute_overlap.required_blocks)
        overlap_end_step = -1
        max_step = ((seq.num_valid_ids - 1) // self.block_size) * self.block_size
        max_step = seq.clamp_prefix_cache_match_step(max_step)
        candidate_steps = self._checkpoint_index.candidate_steps(seq.adapter_name, initial_step, max_step)
        for step in candidate_steps:
            if seq.clamp_prefix_cache_match_step(step) != step:
                continue
            key = self._checkpoint_index.make_request_key(seq, step)
            for node in self._checkpoint_index.candidates(key):
                path_is_current = self._checkpoint_path_is_current(node)
                match_result = self._checkpoint_index.verify_candidate(seq, node, key, path_is_current)
                if match_result.status != StateCheckpointVerifyStatus.HIT:
                    self._handle_state_checkpoint_rejection(seq, node, key, match_result)
                    continue

                if recompute_blocks > 0:
                    overlap_end_step = self._find_recompute_overlap_end(seq, node, recompute_blocks)
                    if overlap_end_step < 0:
                        continue

                self._apply_state_checkpoint_hit(seq, node, match_result.matched_blocks, initial_step,
                                                 overlap_end_step)
                return

        seq.prefix_cache.last_shared_node = initial_cursor
        seq.prefix_cache.recompute_overlap.clear_fresh_block_range()
        self._record_match_stats(seq, query_tokens=seq.num_all_ids - initial_step)
        logger.debug('SSM prefix-cache miss: session_id=%s seq_id=%s initial_step=%s overlap_end_step=%s '
                     'max_step=%s published_steps=%s', seq.session_id, seq.seq_id, initial_step, overlap_end_step,
                     max_step, self._checkpoint_index.num_steps(seq.adapter_name))

    def _match_block_prefix(self, seq: SchedulerSequence):
        """Match the reusable text/VLM block path for one sequence."""
        block_size = self.block_size
        initial_cursor = seq.prefix_cache.last_shared_node
        if initial_cursor is None:
            initial_cursor = self._get_or_create_root(seq.adapter_name)
        initial_step = initial_cursor.num_matched

        node = initial_cursor
        matched_nodes: list[Node] = []
        while node.num_matched + block_size < seq.num_valid_ids:
            start = node.num_matched
            end = start + block_size
            curr_tokens = seq.history_cache[start:end]
            extra_hashes = self._get_block_extra_hashes(seq, start, end)

            key = self._make_block_key(curr_tokens, extra_hashes)
            child = node.children.get(key)
            if child is None or not self._node_matches_block(child, curr_tokens, extra_hashes):
                break
            if seq.return_routed_experts and child.routed_experts is None:
                break

            matched_nodes.append(child)
            node = child

        candidate_step = node.num_matched
        max_match_step = seq.get_prefix_cache_max_match_step()
        clamped_step = seq.clamp_prefix_cache_match_step(min(candidate_step, max_match_step))
        accepted_step = max(initial_step, clamped_step)
        num_accepted_blocks = (accepted_step - initial_step) // block_size
        if num_accepted_blocks < len(matched_nodes):
            matched_nodes = matched_nodes[:num_accepted_blocks]
            node = matched_nodes[-1] if matched_nodes else initial_cursor
        matched_step = node.num_matched

        seq.prefix_cache.recompute_overlap.set_fresh_block_range(matched_step // block_size,
                                                                 candidate_step // block_size)
        if matched_nodes:
            matched_blocks = np.array([matched_node.block for matched_node in matched_nodes])
            self.allocator.update_access_time(matched_blocks)
            self.allocator.add_ref_count(matched_blocks, 1)
            seq.logical_blocks.append(matched_blocks)
            seq.set_step(matched_step)
            self._append_matched_routed_experts(seq, matched_nodes, initial_step)

        self._record_match_stats(seq,
                                 query_tokens=seq.num_all_ids - initial_step,
                                 hit_tokens=matched_step - initial_step)
        seq.prefix_cache.last_shared_node = node
        logger.debug('Prefix-cache match: session_id=%s seq_id=%s initial_step=%s matched_step=%s '
                     'candidate_step=%s clamped=%s', seq.session_id, seq.seq_id, initial_step, matched_step,
                     candidate_step, clamped_step != candidate_step)

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
        seq.prefix_cache.recompute_overlap.clear_fresh_block_range()
        seq.prefix_cache.match_start_step = seq.num_history_ids
        seq.prefix_cache.restore.clear()
        if self.requires_state_checkpoint:
            self._match_state_checkpoint_prefix(seq)
            return

        self._match_block_prefix(seq)

    def _ensure_attached_allocation_cursor(self, seq: SchedulerSequence):
        """Return an attached cursor, resetting a stale sequence cursor."""
        node = seq.prefix_cache.last_shared_node
        if node is None:
            node = self._get_or_create_root(seq.adapter_name)
        elif not self._cursor_path_is_current(node):
            logger.debug('Reset detached prefix-cache sequence cursor: session_id=%s seq_id=%s adapter=%s '
                         'cursor_step=%s', seq.session_id, seq.seq_id, seq.adapter_name, node.num_matched)
            node = self._get_or_create_root(seq.adapter_name)
        seq.prefix_cache.last_shared_node = node
        return node

    def _extend_trie_path(self, seq: SchedulerSequence, node: Node, num_complete_blocks: int):
        """Attach or deduplicate the sequence's completed block path."""
        block_size = self.block_size
        logical_blocks = seq.logical_blocks
        recompute_overlap = seq.prefix_cache.recompute_overlap
        additional_ref_blocks: list[int] = []
        duplicate_blocks: list[int] = []

        self._kv_lifecycle.begin_path_extension(node)
        start_block_id = node.num_matched // block_size
        for block_id in range(start_block_id, num_complete_blocks):
            start = block_id * block_size
            end = start + block_size
            curr_tokens = seq.history_cache[start:end]
            extra_hashes = self._get_block_extra_hashes(seq, start, end)
            block = logical_blocks[block_id]

            hash_key = self._make_block_key(curr_tokens, extra_hashes)
            child = node.children.get(hash_key)
            if child is not None and not self._node_matches_block(child, curr_tokens, extra_hashes):
                break

            if child is not None and recompute_overlap.requires_fresh_block(block_id):
                # Traverse the shared identity path while retaining the fresh,
                # writable sequence block needed to recompute bridge state.
                recompute_overlap.remember_canonical_block(block_id, child.block)
                node = child
                continue

            recompute_overlap.forget_canonical_block(block_id)
            if child is None:
                routed_experts = self._get_routed_experts_for_range(seq, start, end)
                child = Node(hash_key=hash_key,
                             block=block,
                             tokens=curr_tokens,
                             num_matched=end,
                             extra_hashes=extra_hashes,
                             routed_experts=routed_experts,
                             adapter_name=seq.adapter_name)
                child.attach_to(node)
                additional_ref_blocks.append(child.block)
            else:
                # Another sequence inserted this path first. Substitute its
                # canonical block and release the sequence's duplicate block.
                self._try_cache_node_routed_experts(child, seq, start, end)
                if block != child.block:
                    duplicate_blocks.append(block)
                    logical_blocks[block_id] = child.block
                    additional_ref_blocks.append(child.block)
            node = child

        seq.prefix_cache.last_shared_node = node
        self._kv_lifecycle.commit_path_extension(node,
                                                 additional_ref_blocks=additional_ref_blocks,
                                                 duplicate_blocks=duplicate_blocks)

    def allocate(self, seq: SchedulerSequence):
        """Attach newly allocated full blocks to the prefix-cache trie.

        Allocation starts from ``seq.prefix_cache.last_shared_node`` when that
        cursor still reaches the current trie.  Existing identical children are
        deduplicated back to the trie-owned block, except for the recompute
        overlap where the sequence must keep fresh writable KV.
        New nodes take one trie-owned allocator ref. The facade decides the
        identity path; ``KVBlockLifecycle`` commits the resulting batched
        reference changes and leaf bookkeeping.
        """
        if not self.enable:
            return

        block_size = self.block_size
        recompute_overlap = seq.prefix_cache.recompute_overlap
        node = self._ensure_attached_allocation_cursor(seq)
        num_valid_ids = seq.num_valid_ids
        if seq.kv_token_limit is not None:
            num_valid_ids = min(num_valid_ids, seq.kv_token_limit)

        start_block_id = node.num_matched // block_size
        num_complete_blocks = num_valid_ids // block_size
        if start_block_id >= num_complete_blocks:
            recompute_overlap.clear_fresh_block_range()
            return

        self._extend_trie_path(seq, node, num_complete_blocks)
        recompute_overlap.clear_fresh_block_range()

    def evict(self, max_num_blocks: int):
        """Evict trie-owned KV leaf blocks.

        ``KVBlockLifecycle`` owns the auxiliary leaf index and rechecks every
        candidate against topology, allocator refs, and checkpoint pins.
        """
        if not self.enable:
            return 0

        return self._kv_lifecycle.evict(max_num_blocks)
