# Copyright (c) OpenMMLab. All rights reserved.
"""Prefix-cache trie topology, matching, and KV ownership.

``BlockTrie`` owns reusable prefix identity and optional routed-expert replay
data. It delegates trie KV references and leaf eviction to
``KVBlockLifecycle`` and exposes SSM checkpoint slots through the grouped
``state_checkpoints`` lifecycle API. Read
this module together with ``Scheduler._schedule_prefill``,
``InputsMaker.create_model_inputs*``, ``model_forward``, and
``EngineLoop._publish_forward_checkpoints``.

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
4. SSM matching cannot reuse KV alone. It uses sparse published-checkpoint
   lookup, verifies exact host identity, then asks ``ModelAgent`` to restore
   any frozen partial KV block before the checkpoint state on the forward
   stream.
5. SSM checkpoint saves are reserved through ``state_checkpoints``, copied by
   ``ModelAgent`` after forward, and published by ``EngineLoop`` once the
   producer forward is queued.
   Producer/restore pins protect checkpoint slots across async stream-ordering
   windows.

SSM checkpoint detail:

* ``seq.prefix_cache.trie_cursor`` stores the deepest trie node already
  reached by the sequence.  ``match()`` writes it, rollback/free clears it, and
  ``allocate()`` continues inserting newly computed full blocks from it.
* ``StateManager`` owns one state-cache pool split by role: active requests use
  runtime slots stored on ``seq.logical_state``; prefix-cache checkpoints use
  slots stored in a trie's optional ``Node.state_checkpoint`` record. A trie
  node may own KV only, KV plus an unpublished checkpoint reservation, or KV
  plus a published checkpoint.
* Saving a checkpoint starts from the full-block trie node at or before the
  exact save step. A partial step also reserves one checkpoint-owned frozen KV
  block. ``state_checkpoints.reserve_save()`` records a ``pending_save``
  reservation on ``seq.prefix_cache``. Prefill and long-context chunks save at
  the produced chunk end; decode saves remain block-aligned and bounded by
  ``prefix_cache_decode_state_interval``.
* ``InputsMaker`` converts pending saves into compact host copy plans.
  ``ModelAgent`` copies a partial producer block into its frozen destination,
  then copies ``runtime_state -> checkpoint`` after forward on the same stream.
  ``EngineLoop`` calls ``state_checkpoints.publish_save()`` after the forward
  is queued; only then is the checkpoint published into the sparse
  match index. The producing forward holds a producer pin
  until the output/event boundary, so this early visibility cannot make the
  destination slot evictable before the save copy reaches the forward stream.
  Abandoned reservations are discarded.
* Matching a SSM prefix never walks KV blocks as the source of truth.
  ``_match_state_checkpoint_prefix()`` searches published checkpoint steps, filters by
  ``(adapter, step, tail_hash)``, then proves the complete prefix against
  immutable exact-match metadata built when the checkpoint is published.  A
  hit appends trie-owned KV blocks, advances ``seq.num_history_ids``, records a
  selected ``restore``, and may replay routed experts.
* Restore is two-phase. The scheduler/input maker pins the published checkpoint
  by incrementing its pin count. For a partial checkpoint, ``ModelAgent`` first
  copies the frozen tail into the request's private writable block, then copies
  ``checkpoint -> runtime_state`` before the suffix forward. ``EngineLoop``
  releases the pin once the copy has been queued, so LRU eviction cannot reuse
  the checkpoint source slot too early.
* State pressure uses checkpoint LRU. KV pressure first releases unpinned
  checkpoints with frozen blocks, then evicts trie leaves. A KV leaf eviction
  also releases any checkpoint owned by that leaf. A KV match without an exact
  published SSM checkpoint is intentionally a miss.
"""

from dataclasses import dataclass

import numpy as np

from lmdeploy.pytorch.messages import SchedulerSequence
from lmdeploy.pytorch.prefix_cache_state import PrefixCacheExtraIdentity
from lmdeploy.utils import get_logger

from ..block_manager.base_block_manager import LogicalAllocator
from ..state_manager import StateManager
from .checkpoint import (
    StateCheckpointIndex,
    StateCheckpointVerifyStatus,
    checkpoint_anchor_step,
    checkpoint_tail_start,
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
        self.enabled = enabled
        self._use_checkpoints = checkpoint_state_manager is not None

        # caches with different adapter should not be shared.
        self._roots: dict[str, Node] = dict()
        # SSM checkpoints are sparse. The trie still owns KV blocks, but
        # published recurrent-state snapshots are indexed only at selected
        # exact steps.
        self._checkpoint_index = StateCheckpointIndex(self.block_size, self._hash_block)
        self._state_checkpoints = StateCheckpointLifecycle(
            prefix_cache_enabled=self.enabled,
            state_checkpoints_enabled=self._use_checkpoints,
            block_size=self.block_size,
            allocator=self.allocator,
            state_manager=checkpoint_state_manager,
            index=self._checkpoint_index,
            snapshot_match_data=self._snapshot_checkpoint_match_data,
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
    def _hash_block(token_ids: np.ndarray, extra_identity: PrefixCacheExtraIdentity):
        """Hash token ids plus non-token cache identity for trie lookup."""
        return hash(('random', tuple(token_ids), extra_identity))

    @staticmethod
    def _node_matches_block(node: Node, token_ids: np.ndarray, extra_identity: PrefixCacheExtraIdentity):
        """Check the exact key payload after the hash-table lookup."""
        return np.array_equal(token_ids, node.token_ids) and extra_identity == node.extra_identity

    def _find_deepest_block_match_step(self, seq: SchedulerSequence, start_node: Node):
        """Find the deepest KV trie match without acquiring block refs."""
        block_size = self.block_size
        node = start_node
        prefix_len = node.prefix_len
        while prefix_len + block_size < seq.num_valid_ids:
            start = prefix_len
            end = prefix_len + block_size
            token_ids = seq.history_cache[start:end]
            extra_identity = seq.get_prefix_cache_extra_identity(start, end)

            block_hash = self._hash_block(token_ids, extra_identity)
            if block_hash not in node.children:
                break

            child = node.children[block_hash]
            if not self._node_matches_block(child, token_ids, extra_identity):
                break

            node = child
            prefix_len += block_size
        return prefix_len

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
        if not self.enabled or not seq.return_routed_experts:
            return
        node = seq.prefix_cache.trie_cursor
        while node is not None and node.parent is not None:
            end = node.prefix_len
            start = end - self.block_size
            self._try_cache_node_routed_experts(node, seq, start, end)
            node = node.parent

    def cache_routed_experts(self, seqs: list[SchedulerSequence]):
        """Enrich trie nodes with routed experts from multiple sequences."""
        if not self.enabled:
            return
        for seq in seqs:
            self.cache_routed_experts_for_seq(seq)

    def _snapshot_checkpoint_match_data(self, node: Node, seq: SchedulerSequence):
        """Snapshot exact match data from a checkpoint producer.

        The lifecycle already proved that ``node`` owns an unpublished
        reservation. This method proves that the producer still matches that
        owner before freezing its exact identity.

        ``BlockTrie.allocate()`` already made the producer cursor authoritative
        for this prefix. Its recompute-overlap substitutions turn the
        contiguous logical-block copy back into the shared trie path
        without rebuilding thousands of Python ``Node`` objects per save.
        """
        step = node.state_checkpoint.step
        anchor_step = checkpoint_anchor_step(step, self.block_size)
        num_full_blocks = anchor_step // self.block_size
        mismatch = 'Cannot publish an SSM checkpoint from a mismatched sequence cursor'
        if not self._cursor_belongs_to_trie(node):
            raise RuntimeError(f'{mismatch}: checkpoint owner does not belong to this trie.')
        if seq.adapter_name != node.adapter_name:
            raise RuntimeError(f'{mismatch}: adapter does not match the checkpoint owner.')
        if node.prefix_len != anchor_step:
            raise RuntimeError(f'{mismatch}: checkpoint owner is not the expected block anchor.')

        token_ids = seq.history_cache[:step].copy()
        if len(token_ids) != step:
            raise RuntimeError(f'{mismatch}: token identity is incomplete.')
        block_ids = seq.logical_blocks.get_real_blocks()[:num_full_blocks].copy()
        if len(block_ids) != num_full_blocks:
            raise RuntimeError(f'{mismatch}: logical block path is incomplete.')
        seq.prefix_cache.recompute_overlap.apply_trie_blocks(block_ids)
        if num_full_blocks > 0:
            owner_start = anchor_step - self.block_size
            block_extra_identity = seq.get_prefix_cache_extra_identity(owner_start, anchor_step)
            if block_ids[-1] != node.block_id:
                raise RuntimeError(f'{mismatch}: last logical block is not owned by the checkpoint node.')
            if not self._node_matches_block(node, token_ids[owner_start:anchor_step], block_extra_identity):
                raise RuntimeError(f'{mismatch}: last block identity does not match the checkpoint owner.')

        prefix_extra_identity = make_request_multimodal_identity(seq, step)
        tail_start = checkpoint_tail_start(step, self.block_size)
        tail_extra_identity = seq.get_prefix_cache_extra_identity(tail_start, step)
        tail_hash = self._hash_block(token_ids[tail_start:step], tail_extra_identity)
        return freeze_state_checkpoint_match_data(token_ids, prefix_extra_identity, block_ids, tail_hash)

    def _cursor_belongs_to_trie(self, node: Node):
        """Check whether a cursor is an attached node or registered root."""
        if node.parent is None:
            return node.block_id < 0 and self._roots.get(node.adapter_name) is node
        return node.is_attached()

    def _reject_state_checkpoint_candidate(self, seq: SchedulerSequence, node: Node, key, verification):
        """Clean up and log a rejected sparse candidate."""
        checkpoint = node.state_checkpoint
        state_idx = -1 if checkpoint is None else checkpoint.slot
        candidate_step = key[1]
        if verification.status == StateCheckpointVerifyStatus.STALE_INDEX_ENTRY:
            self.state_checkpoints.discard_stale_index_entry(node, key, verification.reason)
        elif verification.status == StateCheckpointVerifyStatus.STALE_CHECKPOINT:
            self.state_checkpoints.release_stale_checkpoint(node, verification.reason)
        logger.debug('Reject SSM prefix-cache checkpoint candidate: session_id=%s seq_id=%s step=%s '
                     'state_idx=%s status=%s reason=%s', seq.session_id, seq.seq_id, candidate_step, state_idx,
                     verification.status.name, verification.reason)

    def _find_recompute_overlap_end(self, seq: SchedulerSequence, node: Node, step: int, recompute_blocks: int):
        """Return the cached overlap end, or ``-1`` when it is too short."""
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
                                    matched_block_ids: np.ndarray,
                                    initial_step: int,
                                    overlap_end_step: int):
        """Apply a verified checkpoint hit to sequence state."""
        checkpoint = node.state_checkpoint
        step = checkpoint.step
        new_blocks = matched_block_ids[initial_step // self.block_size:]
        self.allocator.update_access_time(new_blocks)
        self.allocator.add_ref_count(new_blocks, 1)
        seq.logical_blocks.append(new_blocks)
        seq.set_step(step)
        self._append_state_checkpoint_routed_experts(seq, node, initial_step)

        prefix_cache = seq.prefix_cache
        prefix_cache.restore.select(checkpoint.slot, node)
        prefix_cache.trie_cursor = node
        fresh_start_idx = step // self.block_size
        fresh_end_idx = (step + self.block_size - 1) // self.block_size
        if checkpoint.frozen_block_id >= 0:
            # Forward resumes inside a private partial block. Complete suffix
            # blocks already present in the request must remain private too.
            fresh_end_idx = max(fresh_end_idx, seq.num_valid_ids // self.block_size)
        if overlap_end_step >= 0:
            fresh_end_idx = max(fresh_end_idx, overlap_end_step // self.block_size)
        prefix_cache.recompute_overlap.set_fresh_block_range(fresh_start_idx, fresh_end_idx)

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
        initial_cursor = seq.prefix_cache.trie_cursor
        if initial_cursor is None:
            initial_cursor = self._get_or_create_root(seq.adapter_name)
        initial_step = initial_cursor.prefix_len

        recompute_blocks = max(0, seq.prefix_cache.recompute_overlap.recompute_blocks)
        overlap_end_step = -1
        max_step = seq.get_prefix_cache_max_candidate_step()
        candidate_steps = self._checkpoint_index.candidate_steps(seq.adapter_name, initial_step, max_step)
        for step in candidate_steps:
            if not seq.is_prefix_cache_boundary_safe(step):
                continue
            key = self._checkpoint_index.make_request_key(seq, step)
            for node in self._checkpoint_index.candidates(key):
                verification = self._checkpoint_index.verify_candidate(seq, node, key)
                if verification.status != StateCheckpointVerifyStatus.HIT:
                    self._reject_state_checkpoint_candidate(seq, node, key, verification)
                    continue

                if recompute_blocks > 0:
                    overlap_end_step = self._find_recompute_overlap_end(seq, node, step, recompute_blocks)
                    if overlap_end_step < 0:
                        continue

                self._apply_state_checkpoint_hit(seq, node, verification.matched_block_ids, initial_step,
                                                 overlap_end_step)
                return

        seq.prefix_cache.trie_cursor = initial_cursor
        seq.prefix_cache.recompute_overlap.clear_fresh_block_range()
        self._record_match_stats(seq, query_tokens=seq.num_all_ids - initial_step)
        logger.debug('SSM prefix-cache miss: session_id=%s seq_id=%s initial_step=%s overlap_end_step=%s '
                     'max_step=%s published_steps=%s', seq.session_id, seq.seq_id, initial_step, overlap_end_step,
                     max_step, self._checkpoint_index.num_steps(seq.adapter_name))

    def _match_block_prefix(self, seq: SchedulerSequence):
        """Match the reusable text/VLM block path for one sequence."""
        block_size = self.block_size
        initial_cursor = seq.prefix_cache.trie_cursor
        if initial_cursor is None:
            initial_cursor = self._get_or_create_root(seq.adapter_name)
        initial_step = initial_cursor.prefix_len

        node = initial_cursor
        matched_nodes: list[Node] = []
        missing_routed_experts = False
        while node.prefix_len + block_size < seq.num_valid_ids:
            start = node.prefix_len
            end = start + block_size
            token_ids = seq.history_cache[start:end]
            extra_identity = seq.get_prefix_cache_extra_identity(start, end)

            block_hash = self._hash_block(token_ids, extra_identity)
            child = node.children.get(block_hash)
            if child is None or not self._node_matches_block(child, token_ids, extra_identity):
                break
            if seq.return_routed_experts and child.routed_experts is None:
                missing_routed_experts = True
                break

            matched_nodes.append(child)
            node = child

        candidate_step = node.prefix_len
        if missing_routed_experts:
            candidate_step = self._find_deepest_block_match_step(seq, initial_cursor)
        max_match_step = seq.get_prefix_cache_max_match_step()
        clamped_step = seq.clamp_prefix_cache_match_step(min(candidate_step, max_match_step))
        accepted_step = max(initial_step, clamped_step)
        num_accepted_blocks = (accepted_step - initial_step) // block_size
        if num_accepted_blocks < len(matched_nodes):
            matched_nodes = matched_nodes[:num_accepted_blocks]
            node = matched_nodes[-1] if matched_nodes else initial_cursor
        matched_step = node.prefix_len

        seq.prefix_cache.recompute_overlap.set_fresh_block_range(matched_step // block_size,
                                                                 candidate_step // block_size)
        if matched_nodes:
            matched_block_ids = np.array([matched_node.block_id for matched_node in matched_nodes])
            self.allocator.update_access_time(matched_block_ids)
            self.allocator.add_ref_count(matched_block_ids, 1)
            seq.logical_blocks.append(matched_block_ids)
            seq.set_step(matched_step)
            self._append_matched_routed_experts(seq, matched_nodes, initial_step)

        self._record_match_stats(seq,
                                 query_tokens=seq.num_all_ids - initial_step,
                                 hit_tokens=matched_step - initial_step)
        seq.prefix_cache.trie_cursor = node
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
        if not self.enabled:
            return
        seq.prefix_cache.recompute_overlap.clear_fresh_block_range()
        seq.prefix_cache.match_start_step = seq.num_history_ids
        seq.prefix_cache.restore.clear()
        if self._use_checkpoints:
            self._match_state_checkpoint_prefix(seq)
            return

        self._match_block_prefix(seq)

    def _ensure_attached_allocation_cursor(self, seq: SchedulerSequence):
        """Return an attached cursor, resetting a stale sequence cursor."""
        node = seq.prefix_cache.trie_cursor
        if node is None:
            node = self._get_or_create_root(seq.adapter_name)
        elif not self._cursor_belongs_to_trie(node):
            logger.debug('Reset detached prefix-cache sequence cursor: session_id=%s seq_id=%s adapter=%s '
                         'cursor_step=%s', seq.session_id, seq.seq_id, seq.adapter_name, node.prefix_len)
            node = self._get_or_create_root(seq.adapter_name)
        seq.prefix_cache.trie_cursor = node
        return node

    def _extend_trie_path(self, seq: SchedulerSequence, node: Node, num_complete_blocks: int):
        """Attach or deduplicate the sequence's completed block path."""
        block_size = self.block_size
        logical_blocks = seq.logical_blocks
        recompute_overlap = seq.prefix_cache.recompute_overlap
        fresh_block_range = recompute_overlap.fresh_block_range
        trie_block_map = recompute_overlap.trie_block_map
        ref_blocks: list[int] = []
        free_blocks: list[int] = []

        self._kv_lifecycle.begin_path_extension(node)
        start_block_idx = node.prefix_len // block_size
        for block_idx in range(start_block_idx, num_complete_blocks):
            start = block_idx * block_size
            end = start + block_size
            token_ids = seq.history_cache[start:end]
            extra_identity = seq.get_prefix_cache_extra_identity(start, end)
            block_id = logical_blocks[block_idx]

            block_hash = self._hash_block(token_ids, extra_identity)
            child = node.children.get(block_hash)
            if child is not None and not self._node_matches_block(child, token_ids, extra_identity):
                break

            if fresh_block_range is not None and block_idx in fresh_block_range:
                # Traverse an existing identity path while retaining the fresh,
                # writable sequence block. A missing child ends path extension:
                # the private block must not become trie-owned before forward.
                if child is None:
                    break
                trie_block_map[block_idx] = child.block_id
                node = child
                continue

            trie_block_map.pop(block_idx, None)
            if child is None:
                routed_experts = self._get_routed_experts_for_range(seq, start, end)
                child = Node(block_hash=block_hash,
                             block_id=block_id,
                             token_ids=token_ids,
                             prefix_len=end,
                             extra_identity=extra_identity,
                             routed_experts=routed_experts,
                             adapter_name=seq.adapter_name)
                child.attach_to(node)
                ref_blocks.append(child.block_id)
            else:
                # Another sequence inserted this path first. Substitute its
                # shared trie block and release the sequence's duplicate block.
                self._try_cache_node_routed_experts(child, seq, start, end)
                if block_id != child.block_id:
                    free_blocks.append(block_id)
                    logical_blocks[block_idx] = child.block_id
                    ref_blocks.append(child.block_id)
            node = child

        seq.prefix_cache.trie_cursor = node
        self._kv_lifecycle.commit_path_extension(node, ref_blocks=ref_blocks, free_blocks=free_blocks)

    def allocate(self, seq: SchedulerSequence):
        """Attach newly allocated full blocks to the prefix-cache trie.

        Allocation starts from ``seq.prefix_cache.trie_cursor`` when that
        cursor remains attached to this trie. Existing identical children are
        deduplicated back to the trie-owned block, except for the recompute
        overlap where the sequence must keep fresh writable KV.
        New nodes take one trie-owned allocator ref. The facade decides the
        identity path; ``KVBlockLifecycle`` commits the resulting batched
        reference changes and leaf bookkeeping.
        """
        if not self.enabled:
            return

        block_size = self.block_size
        recompute_overlap = seq.prefix_cache.recompute_overlap
        node = self._ensure_attached_allocation_cursor(seq)
        num_valid_ids = seq.num_valid_ids
        if seq.kv_token_limit is not None:
            num_valid_ids = min(num_valid_ids, seq.kv_token_limit)

        start_block_idx = node.prefix_len // block_size
        num_complete_blocks = num_valid_ids // block_size
        if start_block_idx >= num_complete_blocks:
            recompute_overlap.clear_fresh_block_range()
            return

        self._extend_trie_path(seq, node, num_complete_blocks)
        recompute_overlap.clear_fresh_block_range()

    def evict(self, max_num_blocks: int):
        """Evict checkpoint-frozen and trie-owned KV blocks.

        Frozen partial blocks are cheaper to release because their trie anchor
        remains cached. ``KVBlockLifecycle`` then owns normal leaf eviction and
        rechecks every candidate against topology, allocator refs, and
        checkpoint pins.
        """
        if not self.enabled:
            return 0

        evicted = self._state_checkpoints.evict_frozen_checkpoints(max_num_blocks)
        if evicted < max_num_blocks:
            evicted += self._kv_lifecycle.evict(max_num_blocks - evicted)
        return evicted
