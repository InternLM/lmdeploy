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
   forward, and published by ``EngineLoop`` only after the copy is enqueued.
   Restore refcounts pin checkpoint source slots across that async window.

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
  checkpoint index become matchable.  Abandoned reservations are discarded.
* Matching a SSM prefix never walks KV blocks as the source of truth.
  ``_match_state_checkpoint()`` searches ready checkpoint steps, filters by
  ``(adapter, step, last_block_hash)``, then verifies every ancestor block's
  tokens and multimodal extra hashes before mutating the sequence.  A hit
  appends trie-owned KV blocks, advances ``seq.num_history_ids``, records
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

import enum
import heapq
import logging
import time
from dataclasses import dataclass

import numpy as np

from lmdeploy.pytorch.messages import SchedulerSequence
from lmdeploy.utils import get_logger

from ..config import CacheConfig
from .block_manager import BaseBlockManager

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


class StateCheckpointVerifyStatus(enum.Enum):
    """Outcome of sparse SSM checkpoint verification."""
    HIT = enum.auto()
    REQUEST_MISMATCH = enum.auto()
    STALE_INDEX_ENTRY = enum.auto()
    STALE_CHECKPOINT = enum.auto()


class Node:
    """One full-token-block edge in the prefix-cache trie.

    ``extra_hashes`` augments the token block key with VLM content identity.
    ``state_idx`` / ``state_ready`` / ``state_ref_count`` are optional SSM
    state-checkpoint ownership fields; they are meaningful only when the cache
    config has state shapes.  ``state_ready`` controls whether the checkpoint
    has been published and may be matched.  ``state_ref_count`` pins a ready
    checkpoint while a restore copy may still read it, so LRU eviction or
    checkpoint reuse cannot overwrite the source slot too early.
    """

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
        if old_parent is not None:
            old_parent.children.pop(self.hash_key)
        if val is not None:
            val.children[self.hash_key] = self
        self._parent = val

    def __lt__(self, other):
        return True

    def __le__(self, other):
        return True


@dataclass
class StateCheckpointVerifyResult:
    """Verified checkpoint candidate details."""
    status: StateCheckpointVerifyStatus
    reason: str = ''
    matched_blocks: list[int] | None = None
    matched_nodes: list[Node] | None = None


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
        # SSM checkpoints are sparse.  The trie still owns KV blocks, but ready
        # recurrent-state snapshots are indexed only at selected exact steps.
        self._state_checkpoint_index: dict[tuple, list[Node]] = dict()
        self._state_checkpoint_steps: dict[str, set[int]] = dict()
        self.stats = PrefixCacheStats()

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

    def record_recompute_after_rollback(self, seq: SchedulerSequence, snapshot):
        """Record a recompute after a tentative match was rolled back."""
        if snapshot is None:
            return
        self.stats.num_query_tokens += seq.num_all_ids

    def get_root(self, adapter_name: str):
        """Get root by adapter name."""
        if adapter_name not in self._roots:
            self._roots[adapter_name] = Node(-1, -1, None, adapter_name=adapter_name)
        return self._roots[adapter_name]

    @staticmethod
    def _get_block_extra_hashes(seq: SchedulerSequence, start: int, end: int):
        """Get multimodal identity entries that belong in a block key."""
        return seq.get_prefix_cache_extra_hashes(start, end)

    @staticmethod
    def _make_key(tokens: np.ndarray, extra_hashes: tuple):
        """Make the trie lookup key from tokens plus multimodal identity."""
        return hash(('random', tuple(tokens), extra_hashes))

    @staticmethod
    def _match_node(node: Node, tokens: np.ndarray, extra_hashes: tuple):
        """Check the exact key payload after the hash-table lookup."""
        return np.array_equal(tokens, node.tokens) and extra_hashes == node.extra_hashes

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
        for seq in seqs:
            self.cache_routed_experts_for_seq(seq)

    def _make_state_checkpoint_lookup_key(self, seq: SchedulerSequence, step: int):
        """Make the sparse SSM checkpoint lookup key for a sequence prefix.

        The last block key is only a filter into the sparse index.  Candidate nodes are still verified by walking the
        full ancestor chain so hash collisions or stale index entries cannot produce a false state hit.
        """
        start = step - self.block_size
        end = step
        tokens = seq.history_cache[start:end]
        extra_hashes = self._get_block_extra_hashes(seq, start, end)
        return (seq.adapter_name, step, self._make_key(tokens, extra_hashes))

    @staticmethod
    def _make_state_checkpoint_node_key(node: Node):
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

    def _refresh_state_checkpoint_step(self, adapter_name: str, step: int):
        """Drop an adapter step when no indexed checkpoint still owns it."""
        steps = self._state_checkpoint_steps.get(adapter_name)
        if steps is None or step not in steps:
            return
        has_step = any(key[0] == adapter_name and key[1] == step for key in self._state_checkpoint_index)
        if not has_step:
            steps.remove(step)
        if len(steps) == 0:
            self._state_checkpoint_steps.pop(adapter_name)

    def _remove_state_checkpoint_index_entry(self, node: Node, key: tuple):
        """Remove a node from one sparse-index bucket."""
        nodes = self._state_checkpoint_index.get(key)
        if nodes is None:
            return False

        old_len = len(nodes)
        nodes[:] = [indexed_node for indexed_node in nodes if indexed_node is not node]
        if len(nodes) == old_len:
            return False
        if len(nodes) == 0:
            self._state_checkpoint_index.pop(key)
        self._refresh_state_checkpoint_step(key[0], key[1])
        return True

    def _unindex_state_checkpoint(self, node: Node):
        """Remove a state checkpoint from every sparse-index bucket."""
        removed = False
        for key in list(self._state_checkpoint_index):
            removed = self._remove_state_checkpoint_index_entry(node, key) or removed
        return removed

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

    def mark_state_checkpoint_ready(self, node: Node):
        """Mark a node-owned state checkpoint as ready for SSM matching."""
        if node.state_idx < 0:
            raise RuntimeError('Cannot mark an unreserved state checkpoint as ready.')
        if node.state_ready:
            self._unindex_state_checkpoint(node)
        node.state_ready = True
        node.state_access_time = time.perf_counter()
        self._index_state_checkpoint(node)

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
    def _has_restore_checkpoint_ref(node: Node | None, state_idx: int):
        """Check whether a sequence still owns a restore ref on this node."""
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

    def commit_state_checkpoint_for_seq(self, seq: SchedulerSequence):
        """Publish a sequence state checkpoint after its state copy is
        enqueued.

        Commit validates the remembered node directly.  This matters for decode saves because the sequence may have
        advanced by one sampled token before the output boundary publishes the checkpoint.
        """
        prefix_cache = seq.prefix_cache
        state_idx = prefix_cache.save_state
        save_step = prefix_cache.save_step
        is_decode = prefix_cache.save_is_decode
        node = prefix_cache.save_node
        self._clear_pending_state_checkpoint(seq)
        if state_idx < 0:
            return False

        invalid_reason = self._state_checkpoint_commit_invalid_reason(node, state_idx, save_step)
        if invalid_reason is not None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'Drop invalid SSM prefix-cache checkpoint commit: session_id={seq.session_id} '
                             f'seq_id={seq.seq_id} step={save_step} state_idx={state_idx} '
                             f'is_decode={is_decode} reason={invalid_reason}')
            self._release_invalid_state_checkpoint_reservation(seq, node, state_idx, is_decode)
            return False

        self.mark_state_checkpoint_ready(node)
        if is_decode:
            prefix_cache.decode_state_node = node
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'Commit SSM prefix-cache checkpoint: session_id={seq.session_id} '
                         f'seq_id={seq.seq_id} step={save_step} state_idx={state_idx} is_decode={is_decode}')
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

    def release_state_checkpoint_restore_for_seq(self, seq: SchedulerSequence):
        """Release a state checkpoint pinned for restore."""
        prefix_cache = seq.prefix_cache
        if not prefix_cache.restore_state_acquired:
            return False
        node = prefix_cache.restore_node
        if self._has_restore_checkpoint_ref(node, prefix_cache.restore_state):
            node.state_ref_count -= 1
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
        for seq in seqs:
            self.release_state_checkpoint_restore_for_seq(seq)

    def release_state_checkpoint(self, node: Node):
        """Release a node-owned state checkpoint while keeping KV ownership."""
        if node.state_idx < 0:
            if node.state_ready:
                self._unindex_state_checkpoint(node)
                node.state_ready = False
                node.state_ref_count = 0
                node.state_access_time = 0.0
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

    def _get_node_blocks(self, node: Node):
        """Get trie nodes from root to a target node."""
        nodes = []
        while node is not None and node.parent is not None:
            nodes.append(node)
            node = node.parent
        nodes.reverse()
        return nodes

    def _drop_stale_state_checkpoint_index_entry(self, node: Node, key: tuple, reason: str):
        """Remove a bad sparse-index entry without releasing a valid node."""
        removed = self._remove_state_checkpoint_index_entry(node, key)
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
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'Release stale SSM prefix-cache checkpoint candidate: '
                         f'adapter={node.adapter_name} step={node.num_matched} '
                         f'state_idx={state_idx} was_ready={state_ready} reason={reason}')
        return state_idx >= 0 or state_ready

    def _verify_state_checkpoint_node(self, seq: SchedulerSequence, node: Node, index_key: tuple):
        """Verify a sparse SSM checkpoint candidate exactly.

        Matching only the sparse index key is not enough: we require every
        ancestor block to match tokens and multimodal extra hashes before
        restoring the frozen recurrent state.
        """
        if not self._is_ready_state_checkpoint_node(node):
            return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.STALE_CHECKPOINT,
                                               reason='checkpoint is not ready')

        step = node.num_matched
        if step <= 0:
            return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.STALE_CHECKPOINT,
                                               reason=f'invalid checkpoint step: {step}')

        nodes = self._get_node_blocks(node)
        if len(nodes) * self.block_size != step:
            return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.STALE_CHECKPOINT,
                                               reason='checkpoint ancestor chain is detached')
        for block_node in nodes:
            if not self._is_attached_node(block_node):
                return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.STALE_CHECKPOINT,
                                                   reason='checkpoint ancestor link is stale')

        node_key = self._make_state_checkpoint_node_key(node)
        if index_key != node_key:
            return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.STALE_INDEX_ENTRY,
                                               reason='checkpoint is indexed under a stale key')

        if node.adapter_name != seq.adapter_name:
            return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.STALE_INDEX_ENTRY,
                                               reason='checkpoint adapter differs from lookup adapter')

        max_step = ((seq.num_valid_ids - 1) // self.block_size) * self.block_size
        if step > max_step:
            return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.REQUEST_MISMATCH,
                                               reason='checkpoint is longer than this request')
        if seq.clamp_prefix_cache_match_step(step) != step:
            return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.REQUEST_MISMATCH,
                                               reason='checkpoint would stop inside a multimodal span')

        matched_blocks = []
        for idx, block_node in enumerate(nodes):
            start = idx * self.block_size
            end = start + self.block_size
            tokens = seq.history_cache[start:end]
            extra_hashes = self._get_block_extra_hashes(seq, start, end)
            if not self._match_node(block_node, tokens, extra_hashes):
                return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.REQUEST_MISMATCH,
                                                   reason=f'block payload mismatch at block {idx}')
            matched_blocks.append(block_node.block)

        return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.HIT,
                                           matched_blocks=matched_blocks,
                                           matched_nodes=nodes)

    def _match_state_checkpoint(self, seq: SchedulerSequence):
        """Match SSM prefixes through sparse ready-checkpoint lookup.

        KV-only reuse is unsafe for SSM models, so this path reports a hit only if a ready recurrent-state checkpoint
        exists at the exact matched step.
        """
        seq.prefix_cache.restore_state = -1
        seq.prefix_cache.restore_node = None

        init_curr = seq.prefix_cache.last_shared_node
        if init_curr is None:
            init_curr = self.get_root(seq.adapter_name)
        init_num_matched = init_curr.num_matched

        max_step = ((seq.num_valid_ids - 1) // self.block_size) * self.block_size
        steps = self._state_checkpoint_steps.get(seq.adapter_name, ())
        for step in sorted((step for step in steps if init_num_matched < step <= max_step), reverse=True):
            if seq.clamp_prefix_cache_match_step(step) != step:
                continue
            key = self._make_state_checkpoint_lookup_key(seq, step)
            for node in tuple(self._state_checkpoint_index.get(key, ())):
                match_result = self._verify_state_checkpoint_node(seq, node, key)
                if match_result.status != StateCheckpointVerifyStatus.HIT:
                    if match_result.status == StateCheckpointVerifyStatus.STALE_INDEX_ENTRY:
                        self._drop_stale_state_checkpoint_index_entry(node, key, match_result.reason)
                    elif match_result.status == StateCheckpointVerifyStatus.STALE_CHECKPOINT:
                        self._release_stale_state_checkpoint_candidate(node, match_result.reason)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f'Reject SSM prefix-cache checkpoint candidate: '
                                     f'session_id={seq.session_id} seq_id={seq.seq_id} step={step} '
                                     f'state_idx={node.state_idx} status={match_result.status.name} '
                                     f'reason={match_result.reason}')
                    continue

                matched_blocks = match_result.matched_blocks
                matched_nodes = match_result.matched_nodes
                matched_nodes = matched_nodes[init_num_matched // self.block_size:]
                matched_blocks = np.array(matched_blocks[init_num_matched // self.block_size:])
                self.allocator.update_access_time(matched_blocks)
                self.allocator.add_ref_count(matched_blocks, 1)
                seq.logical_blocks.append(matched_blocks)
                seq.set_step(step)
                self._append_matched_routed_experts(seq, matched_nodes, init_num_matched)
                seq.prefix_cache.restore_state = node.state_idx
                seq.prefix_cache.restore_node = node
                seq.prefix_cache.last_shared_node = node
                self.stats.num_query_tokens += seq.num_all_ids - init_num_matched
                self.stats.num_hit_tokens += step - init_num_matched
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'SSM prefix-cache hit: session_id={seq.session_id} seq_id={seq.seq_id} '
                                 f'init_step={init_num_matched} matched_step={step} '
                                 f'state_idx={node.state_idx}')
                return

        seq.prefix_cache.last_shared_node = init_curr
        self.stats.num_query_tokens += seq.num_all_ids - init_num_matched
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'SSM prefix-cache miss: session_id={seq.session_id} seq_id={seq.seq_id} '
                         f'init_step={init_num_matched} max_step={max_step} ready_steps={len(steps)}')

    def match(self, seq: SchedulerSequence):
        """Match reusable prefix blocks for a sequence.

        Text/VLM models walk the trie block by block.  SSM models delegate to the sparse checkpoint matcher above
        because a KV block match without an exact recurrent-state snapshot must be treated as a miss.
        """
        if not self.enable:
            return
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

        clamped_num_matched = seq.clamp_prefix_cache_match_step(num_matched)
        unclamped_num_matched = num_matched
        __clamp_match_step(clamped_num_matched)

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
        self.stats.num_query_tokens += seq.num_all_ids - init_num_matched
        self.stats.num_hit_tokens += num_matched - init_num_matched

        seq.prefix_cache.last_shared_node = curr
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'Prefix-cache match: session_id={seq.session_id} seq_id={seq.seq_id} '
                         f'init_step={init_num_matched} matched_step={num_matched} '
                         f'candidate_step={unclamped_num_matched} '
                         f'clamped={clamped_num_matched != unclamped_num_matched}')

    def allocate(self, seq: SchedulerSequence):
        """Attach newly allocated full blocks to the prefix-cache trie."""
        if not self.enable:
            return

        block_size = self.block_size
        logical_blocks = seq.logical_blocks
        node: Node = seq.prefix_cache.last_shared_node
        if node is None:
            node = self.get_root(seq.adapter_name)
            seq.prefix_cache.last_shared_node = node

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
                # Another sequence inserted the same key before us.  Reuse the
                # trie-owned block and release this sequence's duplicate block.
                node = child
                self._try_cache_node_routed_experts(node, seq, start, end)
                free_blocks.append(block)
                logical_blocks[block_id] = node.block
            else:
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
