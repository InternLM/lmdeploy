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
4. SSM matching cannot reuse KV alone.  It uses sparse ready checkpoint lookup,
   verifies the full ancestor chain, then asks ``ModelAgent`` to copy the
   frozen checkpoint state into the request runtime state on the forward stream.
5. SSM checkpoint saves are reserved through ``state_checkpoints``, copied by
   ``ModelAgent`` after forward, and published by ``EngineLoop`` once the
   producer forward is queued.
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
  ``state_checkpoints.reserve_for_seq()`` records a ``pending_save`` reservation
  on ``seq.prefix_cache``.  Prefill and
  long-context chunks save at the produced chunk end; decode saves are optional
  and bounded by ``prefix_cache_decode_state_interval``.
* ``InputsMaker`` converts those pending saves into compact host integer
  src/dst pairs.  ``ModelAgent`` then copies ``runtime_state -> checkpoint`` on
  the model forward stream after the model has produced the new SSM state.
  ``EngineLoop`` calls ``state_checkpoints.commit_for_seq()`` after the forward
  is queued; only then does ``state_ready`` become true and the sparse
  checkpoint index become matchable.  The producing forward holds a producer ref
  until the output/event boundary, so this early visibility cannot make the
  destination slot evictable before the save copy reaches the forward stream.
  Abandoned reservations are discarded.
* Matching a SSM prefix never walks KV blocks as the source of truth.
  ``_match_state_checkpoint()`` searches ready checkpoint steps, filters by
  ``(adapter, step, last_block_hash)``, then proves the complete prefix against
  immutable exact-match metadata built when the checkpoint is published.  A
  hit appends trie-owned KV blocks, advances ``seq.num_history_ids``, records a
  selected ``restore``, and may replay routed experts.
* Restore is two-phase.  The scheduler/input maker pins the ready checkpoint by
  incrementing ``state_ref_count``.  ``ModelAgent`` copies
  ``checkpoint -> runtime_state`` before the suffix forward.  ``EngineLoop``
  releases the pin once the copy has been queued, so LRU eviction cannot reuse
  the checkpoint source slot too early.
* Checkpoint eviction is state-only LRU over ready, unpinned nodes.  KV leaf
  eviction also releases any checkpoint owned by that leaf.  A KV match without
  an exact ready SSM checkpoint is intentionally a miss.
"""

import logging
from dataclasses import dataclass

import numpy as np

from lmdeploy.pytorch.messages import PrefixCacheExtraHashes, SchedulerSequence
from lmdeploy.utils import get_logger

from ...config import CacheConfig
from ..block_manager import BaseBlockManager
from .checkpoint import (
    StateCheckpointIndex,
    StateCheckpointVerifyStatus,
    freeze_state_checkpoint_match_data,
    make_node_multimodal_identity,
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
    * optional SSM checkpoint reserve/commit/restore/save lifecycle;
    * sparse ready-checkpoint lookup for SSM prefix hits;
    * best-effort routed-expert replay data;
    * prefix-cache stats used by scheduler rollback.
    """

    def __init__(self, cache_config: CacheConfig, block_manager: BaseBlockManager, state_manager=None):
        self.allocator = block_manager.allocator
        self.block_size = cache_config.block_size
        self.enable = cache_config.enable_prefix_caching
        self.requires_state_checkpoint = state_manager is not None and len(cache_config.states_shapes) > 0

        # caches with different adapter should not be shared.
        self._roots: dict[str, Node] = dict()
        # SSM checkpoints are sparse.  The trie still owns KV blocks, but ready
        # recurrent-state snapshots are indexed only at selected exact steps.
        self._checkpoint_index = StateCheckpointIndex(self.block_size, self._make_key)
        self._state_checkpoints = StateCheckpointLifecycle(
            prefix_cache_enabled=self.enable,
            state_checkpoints_enabled=self.requires_state_checkpoint,
            block_size=self.block_size,
            state_manager=state_manager,
            index=self._checkpoint_index,
            is_attached_node=Node.is_attached,
            find_checkpoint_node=self._get_state_checkpoint_node_for_seq,
            make_node_match_data=self._make_state_checkpoint_match_data_from_node,
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

    def _make_state_checkpoint_match_data_from_node(self, node: Node):
        """Build exact match data by reconstructing a node path.

        This fallback serves direct-node tests and diagnostics.  Production publication uses the producer sequence's
        contiguous buffers instead.
        """
        nodes = tuple(node.path_from_root())
        if len(nodes) * self.block_size != node.num_matched:
            raise RuntimeError('Cannot publish an SSM checkpoint with a detached ancestor chain.')
        if any(not block_node.is_attached() for block_node in nodes):
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

    def _has_current_state_checkpoint_path(self, node: Node):
        """Check a cached checkpoint path through its invalidation contract."""
        match_data = node.state_match_data
        if match_data is None or len(match_data.blocks) == 0:
            return False
        return len(match_data.blocks) * self.block_size == node.num_matched and node.is_attached()

    def _is_attached_cursor(self, node: Node):
        """Check whether a sequence cursor still reaches the adapter root."""
        if node.parent is None:
            return node.block < 0 and self._roots.get(node.adapter_name) is node
        if node.state_match_data is not None:
            return self._has_current_state_checkpoint_path(node)
        nodes = node.path_from_root()
        if len(nodes) * self.block_size != node.num_matched:
            return False
        return all(path_node.is_attached() for path_node in nodes)

    def _handle_state_checkpoint_rejection(self, seq: SchedulerSequence, node: Node, key, match_result):
        """Clean up a rejected sparse candidate according to its status."""
        if match_result.status == StateCheckpointVerifyStatus.STALE_INDEX_ENTRY:
            self.state_checkpoints._drop_stale_index_entry(node, key, match_result.reason)
        elif match_result.status == StateCheckpointVerifyStatus.STALE_CHECKPOINT:
            self.state_checkpoints._release_stale_candidate(node, match_result.reason)
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
        prefix_cache.restore.select(node.state_idx, node)
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
        init_curr = seq.prefix_cache.last_shared_node
        if init_curr is None:
            init_curr = self._get_or_create_root(seq.adapter_name)
        init_num_matched = init_curr.num_matched

        recompute_blocks = max(0, seq.prefix_cache.match_recompute_blocks)
        raw_match_step = -1
        max_step = ((seq.num_valid_ids - 1) // self.block_size) * self.block_size
        max_step = seq.clamp_prefix_cache_match_step(max_step)
        candidate_steps = self._checkpoint_index.candidate_steps(seq.adapter_name, init_num_matched, max_step)
        for step in candidate_steps:
            if seq.clamp_prefix_cache_match_step(step) != step:
                continue
            key = self._checkpoint_index.make_request_key(seq, step)
            for node in self._checkpoint_index.candidates(key):
                path_is_current = self._has_current_state_checkpoint_path(node)
                match_result = self._checkpoint_index.verify(seq, node, key, path_is_current)
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
                         f'max_step={max_step} ready_steps={self._checkpoint_index.num_steps(seq.adapter_name)}')

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
        seq.prefix_cache.restore.clear()
        if self.requires_state_checkpoint:
            self._match_state_checkpoint(seq)
            return

        block_size = self.block_size
        matched_blocks = []

        curr: Node = seq.prefix_cache.last_shared_node
        if curr is None:
            curr = self._get_or_create_root(seq.adapter_name)
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
                seq.prefix_cache.restore.select(curr.state_idx, curr)

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
        New nodes take one trie-owned allocator ref. The facade decides the
        identity path; ``KVBlockLifecycle`` commits the resulting batched
        reference changes and leaf bookkeeping.
        """
        if not self.enable:
            return

        block_size = self.block_size
        logical_blocks = seq.logical_blocks
        node: Node = seq.prefix_cache.last_shared_node
        if node is None:
            node = self._get_or_create_root(seq.adapter_name)
            seq.prefix_cache.last_shared_node = node
        elif not self._is_attached_cursor(node):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'Reset detached prefix-cache sequence cursor: session_id={seq.session_id} '
                             f'seq_id={seq.seq_id} adapter={seq.adapter_name} '
                             f'cursor_step={node.num_matched}')
            node = self._get_or_create_root(seq.adapter_name)
            seq.prefix_cache.last_shared_node = node

        num_matched = node.num_matched
        num_valid_ids = seq.num_valid_ids
        if seq.kv_token_limit is not None:
            num_valid_ids = min(num_valid_ids, seq.kv_token_limit)

        if num_matched + block_size > num_valid_ids:
            self._clear_private_recompute_range(seq)
            return

        self._kv_lifecycle.begin_path_extension(node)

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
        self._kv_lifecycle.commit_path_extension(node,
                                                 additional_ref_blocks=blocks,
                                                 duplicate_blocks=free_blocks)
        self._clear_private_recompute_range(seq)

    def evict(self, max_num_blocks: int):
        """Evict trie-owned KV leaf blocks.

        ``KVBlockLifecycle`` owns the auxiliary leaf index and rechecks every
        candidate against topology, allocator refs, and checkpoint pins.
        """
        if not self.enable:
            return 0

        return self._kv_lifecycle.evict(max_num_blocks)
