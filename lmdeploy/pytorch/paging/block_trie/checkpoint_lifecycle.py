# Copyright (c) OpenMMLab. All rights reserved.
"""Ownership lifecycle for node-backed SSM state checkpoints.

The lifecycle owns state-slot reservation, publication, async-copy pins, and
checkpoint-only eviction.  It deliberately does not own trie topology or KV
blocks.  Instead, :class:`BlockTrie` supplies narrow callbacks for the few
topology and exact-identity decisions needed while publishing a checkpoint.
"""

from __future__ import annotations

import heapq
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from lmdeploy.pytorch.messages import SchedulerSequence
from lmdeploy.utils import get_logger

from .checkpoint import StateCheckpointIndex

if TYPE_CHECKING:
    from ..state_manager import StateManager
    from .checkpoint import StateCheckpointKey, StateCheckpointMatchData
    from .node import Node

logger = get_logger('lmdeploy')

NodePredicate = Callable[['Node'], bool]
CheckpointNodeFinder = Callable[[SchedulerSequence, int], 'Node | None']
NodeMatchDataBuilder = Callable[['Node'], 'StateCheckpointMatchData']
SequenceMatchDataBuilder = Callable[['Node', SchedulerSequence], 'StateCheckpointMatchData']


class StateCheckpointLifecycle:
    """Manage recurrent-state checkpoint ownership independently of KV.

    Trie nodes remain the source of truth: each node records its checkpoint
    slot, readiness, and reference count.  This component coordinates those
    fields with ``StateManager`` and ``StateCheckpointIndex``.  The callbacks
    keep topology and exact-identity policy in ``BlockTrie`` without giving the
    lifecycle a back-reference to the whole trie.
    """

    def __init__(self,
                 *,
                 prefix_cache_enabled: bool,
                 state_checkpoints_enabled: bool,
                 block_size: int,
                 state_manager: StateManager | None,
                 index: StateCheckpointIndex,
                 is_attached_node: NodePredicate,
                 find_checkpoint_node: CheckpointNodeFinder,
                 make_node_match_data: NodeMatchDataBuilder,
                 make_sequence_match_data: SequenceMatchDataBuilder):
        self.prefix_cache_enabled = prefix_cache_enabled
        self.state_checkpoints_enabled = state_checkpoints_enabled
        self.block_size = block_size
        self.state_manager = state_manager
        self.index = index
        self._is_attached_node = is_attached_node
        self._find_checkpoint_node = find_checkpoint_node
        self._make_node_match_data = make_node_match_data
        self._make_sequence_match_data = make_sequence_match_data

    def index_checkpoint(self, node: Node):
        """Add a ready checkpoint to the sparse index."""
        if node.state_idx < 0 or not node.state_ready:
            raise RuntimeError('Cannot index an unready SSM prefix-cache checkpoint.')
        if not self._is_attached_node(node):
            raise RuntimeError('Cannot index a detached SSM prefix-cache checkpoint node.')
        self.index.add(node)

    def unindex_checkpoint(self, node: Node):
        """Remove a checkpoint from every sparse-index bucket."""
        return self.index.remove(node)

    def reserve_state_checkpoint(self, node: Node):
        """Reserve a state-cache slot owned by a trie node.

        Replacing a ready checkpoint is allowed only while no async copy pins it.  If the shared state pool is full, one
        old unpinned checkpoint may be evicted without removing its trie/KV node.
        """
        if not self.state_checkpoints_enabled or node.parent is None:
            return -1
        if node.state_ready:
            if node.state_ref_count > 0:
                return -1
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'Replace SSM prefix-cache checkpoint: adapter={node.adapter_name} '
                             f'step={node.num_matched} state_idx={node.state_idx}')
            self.unindex_checkpoint(node)
        elif node.state_idx >= 0:
            return -1
        if node.state_idx < 0:
            if self.state_manager.get_num_free_checkpoint() == 0 and self.evict_state_checkpoints(1) == 0:
                return -1
            node.state_idx = self.state_manager.allocate_checkpoint_state()
        node.state_ready = False
        node.state_topology_epoch = node._topology_epoch
        return node.state_idx

    def discard_state_checkpoint_for_seq(self, seq: SchedulerSequence):
        """Discard an unpublished checkpoint reservation for a sequence."""
        prefix_cache = seq.prefix_cache
        pending_save = prefix_cache.pending_save
        if not pending_save.is_pending:
            pending_save.clear()
            return False
        state_idx = pending_save.slot
        node = pending_save.node
        is_decode = pending_save.is_decode
        pending_save.clear()
        if self._is_unpublished_reservation(node, state_idx):
            if is_decode and prefix_cache.decode_checkpoint_node is node:
                prefix_cache.decode_checkpoint_node = None
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
        """Reserve a checkpoint at an attached, block-aligned trie step."""
        self.discard_state_checkpoint_for_seq(seq)

        if not self.prefix_cache_enabled or not self.state_checkpoints_enabled:
            return -1

        if step is None:
            step = seq.num_valid_ids
        if step <= 0 or step % self.block_size != 0:
            return -1
        if step > seq.num_valid_ids:
            return -1
        if seq.clamp_prefix_cache_match_step(step) != step:
            return -1

        node = self._find_checkpoint_node(seq, step)
        if node is None or node.state_ready:
            return -1

        try:
            state_idx = self.reserve_state_checkpoint(node)
        except RuntimeError as e:
            if 'No free states' not in str(e):
                raise
            return -1
        if state_idx < 0:
            return -1

        seq.prefix_cache.pending_save.reserve(state_idx, step, node, is_decode)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'Reserve SSM prefix-cache checkpoint: session_id={seq.session_id} '
                         f'seq_id={seq.seq_id} step={step} state_idx={state_idx} is_decode={is_decode}')
        return state_idx

    def reserve_decode_state_checkpoint_for_seq(self,
                                                seq: SchedulerSequence,
                                                interval: int,
                                                step: int = None):
        """Reserve a bounded, replaceable decode checkpoint."""
        if step is None:
            step = seq.num_valid_ids
        if interval <= 0 or step % interval != 0:
            return -1
        if not self.prefix_cache_enabled or not self.state_checkpoints_enabled:
            return -1
        if step <= 0 or step % self.block_size != 0:
            return -1
        if step > seq.num_valid_ids:
            return -1
        if seq.clamp_prefix_cache_match_step(step) != step:
            return -1
        node = self._find_checkpoint_node(seq, step)
        if node is None or node.state_ready or node.state_idx >= 0:
            return -1

        prefix_cache = seq.prefix_cache
        old_node = prefix_cache.decode_checkpoint_node
        if old_node is not None and old_node.state_idx < 0:
            prefix_cache.decode_checkpoint_node = None
            old_node = None
        if old_node is not None:
            if self._is_same_ready_decode_checkpoint(old_node, step):
                return -1
            if old_node.state_ref_count > 0:
                return -1
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'Release previous decode SSM prefix-cache checkpoint: '
                             f'session_id={seq.session_id} seq_id={seq.seq_id} '
                             f'old_step={old_node.num_matched} old_state_idx={old_node.state_idx} '
                             f'new_step={step}')
            self.release_state_checkpoint(old_node)
            prefix_cache.decode_checkpoint_node = None

        return self.reserve_state_checkpoint_for_seq(seq, step=step, is_decode=True)

    def mark_state_checkpoint_ready(self, node: Node, seq: SchedulerSequence | None = None):
        """Publish a node-owned checkpoint after its state copy is queued."""
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
            match_data = self._make_node_match_data(node)
        else:
            match_data = self._make_sequence_match_data(node, seq)
        node.state_match_data = match_data
        node.state_ready = True
        node.state_access_time = time.perf_counter()
        try:
            self.index_checkpoint(node)
        except Exception:
            # Publication is transactional: the caller still owns an unready
            # reservation and may release its slot after this rollback.
            self.unindex_checkpoint(node)
            node.state_ready = False
            node.state_access_time = 0.0
            node.state_match_data = None
            raise

    @staticmethod
    def _is_same_ready_decode_checkpoint(node: Node, step: int):
        return node.num_matched == step and node.state_ready

    def _commit_invalid_reason(self, node: Node | None, state_idx: int, save_step: int):
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
    def _is_unpublished_reservation(node: Node | None, state_idx: int):
        return node is not None and node.state_idx == state_idx and not node.state_ready

    @staticmethod
    def is_ready_checkpoint(node: Node | None, state_idx: int):
        """Whether a node owns the specified ready checkpoint slot."""
        return node is not None and node.state_idx == state_idx and node.state_ready

    @staticmethod
    def _has_checkpoint_ref(node: Node | None, state_idx: int):
        return node is not None and node.state_idx == state_idx and node.state_ref_count > 0

    @staticmethod
    def _is_evictable_checkpoint(node: Node):
        return node.state_idx >= 0 and node.state_ready and node.state_ref_count == 0

    @staticmethod
    def is_pinned_checkpoint(node: Node):
        """Whether an async save or restore still pins a checkpoint."""
        return node.state_ref_count > 0

    def _release_invalid_reservation(self,
                                     seq: SchedulerSequence,
                                     node: Node | None,
                                     state_idx: int,
                                     is_decode: bool):
        if not self._is_unpublished_reservation(node, state_idx):
            return
        if is_decode and seq.prefix_cache.decode_checkpoint_node is node:
            seq.prefix_cache.decode_checkpoint_node = None
        self.release_state_checkpoint(node)

    def _acquire_save_pin(self, seq: SchedulerSequence, node: Node, state_idx: int):
        producer_pin = seq.prefix_cache.producer_save_pin
        if producer_pin.is_acquired:
            raise RuntimeError('SSM prefix-cache save checkpoint already has an in-flight producer ref.')
        if not self.is_ready_checkpoint(node, state_idx):
            return False
        node.state_ref_count += 1
        node.state_access_time = time.perf_counter()
        producer_pin.acquire(state_idx, node)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'Acquire SSM prefix-cache save checkpoint: session_id={seq.session_id} '
                         f'seq_id={seq.seq_id} step={node.num_matched} state_idx={state_idx} '
                         f'ref_count={node.state_ref_count}')
        return True

    def commit_state_checkpoint_for_seq(self, seq: SchedulerSequence, acquire_save_ref: bool = False):
        """Publish a pending checkpoint and optionally pin its producer."""
        prefix_cache = seq.prefix_cache
        pending_save = prefix_cache.pending_save
        if not pending_save.is_pending:
            pending_save.clear()
            return False
        state_idx = pending_save.slot
        save_step = pending_save.step
        is_decode = pending_save.is_decode
        node = pending_save.node

        invalid_reason = self._commit_invalid_reason(node, state_idx, save_step)
        if invalid_reason is not None:
            pending_save.clear()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'Drop invalid SSM prefix-cache checkpoint commit: session_id={seq.session_id} '
                             f'seq_id={seq.seq_id} step={save_step} state_idx={state_idx} '
                             f'is_decode={is_decode} reason={invalid_reason}')
            self._release_invalid_reservation(seq, node, state_idx, is_decode)
            return False

        try:
            self.mark_state_checkpoint_ready(node, seq)
        except Exception:
            pending_save.clear()
            self._release_invalid_reservation(seq, node, state_idx, is_decode)
            raise
        pending_save.clear()
        if is_decode:
            prefix_cache.decode_checkpoint_node = node
        if acquire_save_ref:
            self._acquire_save_pin(seq, node, state_idx)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'Commit SSM prefix-cache checkpoint: session_id={seq.session_id} '
                         f'seq_id={seq.seq_id} step={save_step} state_idx={state_idx} is_decode={is_decode}')
        return True

    def commit_state_checkpoints(self, seqs: list[SchedulerSequence], acquire_save_ref: bool = False):
        """Publish pending checkpoints for a batch."""
        if not self.prefix_cache_enabled:
            return
        for seq in seqs:
            self.commit_state_checkpoint_for_seq(seq, acquire_save_ref=acquire_save_ref)

    def acquire_restore_for_seq(self, seq: SchedulerSequence):
        """Pin a matched checkpoint until its restore copy is queued."""
        restore = seq.prefix_cache.restore
        if not restore.is_selected or restore.pinned:
            return False
        node = restore.node
        if not self.is_ready_checkpoint(node, restore.slot):
            return False
        node.state_ref_count += 1
        node.state_access_time = time.perf_counter()
        restore.pinned = True
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'Acquire SSM prefix-cache restore checkpoint: session_id={seq.session_id} '
                         f'seq_id={seq.seq_id} step={node.num_matched} state_idx={node.state_idx} '
                         f'ref_count={node.state_ref_count}')
        return True

    def acquire_restores(self, seqs: list[SchedulerSequence]):
        """Pin matched checkpoints for a batch."""
        for seq in seqs:
            self.acquire_restore_for_seq(seq)

    @classmethod
    def _release_checkpoint_ref(cls, node: Node | None, state_idx: int, err_msg: str):
        if not cls._has_checkpoint_ref(node, state_idx):
            cls._warn_unexpected_state(f'{err_msg} state_idx={state_idx}')
            raise RuntimeError(err_msg)
        node.state_ref_count -= 1
        return node

    def release_restore_for_seq(self, seq: SchedulerSequence):
        """Release a checkpoint pinned for restore."""
        restore = seq.prefix_cache.restore
        if not restore.pinned:
            return False
        node = self._release_checkpoint_ref(
            restore.node,
            restore.slot,
            'Acquired SSM prefix-cache restore checkpoint lost its node reference.',
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'Release SSM prefix-cache restore checkpoint: session_id={seq.session_id} '
                         f'seq_id={seq.seq_id} step={node.num_matched} state_idx={node.state_idx} '
                         f'ref_count={node.state_ref_count}')
        restore.clear()
        return True

    def release_restores(self, seqs: list[SchedulerSequence]):
        """Release checkpoints pinned for a batch restore."""
        if not self.prefix_cache_enabled:
            return
        for seq in seqs:
            self.release_restore_for_seq(seq)

    def release_save_for_seq(self, seq: SchedulerSequence):
        """Release a checkpoint pinned for its producer save copy."""
        producer_pin = seq.prefix_cache.producer_save_pin
        if not producer_pin.is_acquired:
            return False
        node = self._release_checkpoint_ref(
            producer_pin.node,
            producer_pin.slot,
            'Acquired SSM prefix-cache save checkpoint lost its node reference.',
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'Release SSM prefix-cache save checkpoint: session_id={seq.session_id} '
                         f'seq_id={seq.seq_id} step={node.num_matched} state_idx={node.state_idx} '
                         f'ref_count={node.state_ref_count}')
        producer_pin.clear()
        return True

    def release_saves(self, seqs: list[SchedulerSequence]):
        """Release producer pins held by a batch of saved checkpoints."""
        if not self.prefix_cache_enabled:
            return
        for seq in seqs:
            self.release_save_for_seq(seq)

    def release_state_checkpoint(self, node: Node):
        """Release a node-owned state checkpoint while keeping its KV node."""
        if node.state_ref_count > 0:
            raise RuntimeError('Cannot release a pinned SSM prefix-cache checkpoint.')
        if node.state_idx < 0:
            if node.state_ready:
                self._warn_unexpected_state(
                    f'ready SSM checkpoint has no state slot: adapter={node.adapter_name} '
                    f'step={node.num_matched}')
                self.unindex_checkpoint(node)
                node.state_ready = False
                node.state_ref_count = 0
                node.state_access_time = 0.0
            node.state_match_data = None
            node.state_topology_epoch = -1
            return
        if node.state_ready:
            self.unindex_checkpoint(node)
        self.state_manager.free_checkpoint_state(node.state_idx)
        node.state_idx = -1
        node.state_ready = False
        node.state_ref_count = 0
        node.state_access_time = 0.0
        node.state_match_data = None
        node.state_topology_epoch = -1

    def evict_state_checkpoints(self, max_num_states: int):
        """Evict ready state checkpoints without removing KV trie nodes."""
        if not self.state_checkpoints_enabled or max_num_states <= 0:
            return 0

        candidates = [(node.state_access_time, node) for node in self.index.unique_nodes()
                      if self._is_evictable_checkpoint(node)]
        heapq.heapify(candidates)

        evicted = 0
        while candidates and evicted < max_num_states:
            _, node = heapq.heappop(candidates)
            if not self._is_evictable_checkpoint(node):
                continue
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'Evict SSM prefix-cache checkpoint: adapter={node.adapter_name} '
                             f'step={node.num_matched} state_idx={node.state_idx}')
            self.release_state_checkpoint(node)
            evicted += 1
        return evicted

    def drop_stale_index_entry(self, node: Node, key: StateCheckpointKey, reason: str):
        """Remove a stale index entry without releasing a valid checkpoint."""
        removed = self.index.remove_entry(node, key)
        if removed and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'Drop stale SSM prefix-cache checkpoint index entry: adapter={key[0]} '
                         f'step={key[1]} node_adapter={node.adapter_name} '
                         f'node_step={node.num_matched} state_idx={node.state_idx} reason={reason}')
        return removed

    def release_stale_candidate(self, node: Node, reason: str):
        """Release a globally stale checkpoint candidate when it is
        unpinned."""
        if self.is_pinned_checkpoint(node):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'Skip pinned stale SSM prefix-cache checkpoint candidate: '
                             f'adapter={node.adapter_name} step={node.num_matched} '
                             f'state_idx={node.state_idx} ref_count={node.state_ref_count} '
                             f'reason={reason}')
            return False

        state_idx = node.state_idx
        state_ready = node.state_ready
        self.unindex_checkpoint(node)
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

    @staticmethod
    def _warn_unexpected_state(message: str):
        logger.warning('Unexpected prefix-cache trie state: %s', message)
