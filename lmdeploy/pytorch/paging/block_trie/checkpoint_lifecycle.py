# Copyright (c) OpenMMLab. All rights reserved.
"""Ownership lifecycle for node-backed SSM state checkpoints.

The lifecycle owns state-slot reservation, publication, async-copy pins, and
checkpoint-only eviction. It deliberately does not own trie topology or KV
blocks. :class:`BlockTrie` supplies the exact-identity builder needed while
publishing a checkpoint, while ``KVBlockLifecycle`` consults this owner before
evicting a checkpoint-bearing KV leaf.
"""

from __future__ import annotations

import heapq
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from lmdeploy.pytorch.messages import SchedulerSequence
from lmdeploy.utils import get_logger

from .checkpoint import StateCheckpointIndex
from .node import Node, NodeStateCheckpoint

if TYPE_CHECKING:
    from ..state_manager import StateManager
    from .checkpoint import StateCheckpointKey, StateCheckpointMatchData

logger = get_logger('lmdeploy')

SequenceMatchDataBuilder = Callable[['Node', SchedulerSequence], 'StateCheckpointMatchData']


class StateCheckpointLifecycle:
    """Manage recurrent-state checkpoint ownership independently of KV.

    Trie nodes remain the source of truth: a node with state ownership holds a
    lazily allocated :class:`NodeStateCheckpoint`. This component coordinates
    that record with ``StateManager`` and ``StateCheckpointIndex``. Nodes own
    their monotonic attachment invariant; the injected builder keeps
    exact-identity policy in ``BlockTrie`` without giving the lifecycle a
    back-reference to the whole trie.
    """

    def __init__(self,
                 *,
                 prefix_cache_enabled: bool,
                 state_checkpoints_enabled: bool,
                 block_size: int,
                 state_manager: StateManager | None,
                 index: StateCheckpointIndex,
                 make_sequence_match_data: SequenceMatchDataBuilder):
        self._prefix_cache_enabled = prefix_cache_enabled
        self._state_checkpoints_enabled = state_checkpoints_enabled
        self._block_size = block_size
        self._state_manager = state_manager
        self._index = index
        self._make_sequence_match_data = make_sequence_match_data

    def reserve_save(self, seq: SchedulerSequence, step: int = None, is_decode: bool = False):
        """Reserve a checkpoint at an attached, block-aligned trie step."""
        self.discard_save(seq)

        if not self._prefix_cache_enabled or not self._state_checkpoints_enabled:
            return -1

        if step is None:
            step = seq.num_valid_ids
        if step <= 0 or step % self._block_size != 0:
            return -1
        if step > seq.num_valid_ids:
            return -1
        if seq.clamp_prefix_cache_match_step(step) != step:
            return -1

        node = self._find_checkpoint_node(seq, step)
        if node is None or self._node_has_slot(node):
            return -1

        try:
            slot = self._reserve_slot(node)
        except RuntimeError as e:
            if 'No free states' not in str(e):
                raise
            return -1
        if slot < 0:
            return -1

        seq.prefix_cache.pending_save.reserve(slot, step, node, is_decode)
        logger.debug('Reserve SSM prefix-cache checkpoint: session_id=%s seq_id=%s step=%s state_idx=%s '
                     'is_decode=%s', seq.session_id, seq.seq_id, step, slot, is_decode)
        return slot

    def reserve_decode_save(self, seq: SchedulerSequence, interval: int, step: int = None):
        """Reserve a bounded, replaceable decode checkpoint."""
        if step is None:
            step = seq.num_valid_ids
        if interval <= 0 or step % interval != 0:
            return -1
        if not self._prefix_cache_enabled or not self._state_checkpoints_enabled:
            return -1
        if step <= 0 or step % self._block_size != 0:
            return -1
        if step > seq.num_valid_ids:
            return -1
        if seq.clamp_prefix_cache_match_step(step) != step:
            return -1
        node = self._find_checkpoint_node(seq, step)
        if node is None or self._node_has_slot(node):
            return -1

        if not self._release_replaceable_decode_checkpoint(seq, step):
            return -1

        return self.reserve_save(seq, step=step, is_decode=True)

    def discard_save(self, seq: SchedulerSequence):
        """Discard an unpublished checkpoint reservation for a sequence."""
        prefix_cache = seq.prefix_cache
        pending_save = prefix_cache.pending_save
        if not pending_save.is_pending:
            pending_save.clear()
            return False
        slot = pending_save.slot
        node = pending_save.node
        is_decode = pending_save.is_decode
        pending_save.clear()
        if self._is_unpublished_reservation(node, slot):
            if is_decode and prefix_cache.decode_checkpoint_node is node:
                prefix_cache.decode_checkpoint_node = None
            logger.debug('Discard SSM prefix-cache checkpoint reservation: session_id=%s seq_id=%s step=%s '
                         'state_idx=%s is_decode=%s', seq.session_id, seq.seq_id, node.prefix_len, slot,
                         is_decode)
            self.release_checkpoint(node)
            return True
        return False

    def publish_save(self, seq: SchedulerSequence, pin_save: bool = False):
        """Publish a pending checkpoint and optionally pin its producer."""
        prefix_cache = seq.prefix_cache
        pending_save = prefix_cache.pending_save
        if not pending_save.is_pending:
            pending_save.clear()
            return False
        slot = pending_save.slot
        save_step = pending_save.step
        is_decode = pending_save.is_decode
        node = pending_save.node

        invalid_reason = self._publication_invalid_reason(node, slot, save_step)
        if invalid_reason is not None:
            pending_save.clear()
            logger.debug('Drop invalid SSM prefix-cache checkpoint publication: session_id=%s seq_id=%s step=%s '
                         'state_idx=%s is_decode=%s reason=%s', seq.session_id, seq.seq_id, save_step, slot,
                         is_decode, invalid_reason)
            self._release_invalid_reservation(seq, node, slot, is_decode)
            return False

        try:
            self._publish_checkpoint(node, seq)
        except Exception:
            pending_save.clear()
            self._release_invalid_reservation(seq, node, slot, is_decode)
            raise
        pending_save.clear()
        if is_decode:
            prefix_cache.decode_checkpoint_node = node
        if pin_save:
            self._pin_save(seq, node, slot)
        logger.debug('Publish SSM prefix-cache checkpoint: session_id=%s seq_id=%s step=%s state_idx=%s '
                     'is_decode=%s', seq.session_id, seq.seq_id, save_step, slot, is_decode)
        return True

    def publish_saves(self, seqs: list[SchedulerSequence], pin_saves: bool = False):
        """Publish pending checkpoints for a batch."""
        if not self._prefix_cache_enabled:
            return
        for seq in seqs:
            self.publish_save(seq, pin_save=pin_saves)

    def pin_restore(self, seq: SchedulerSequence):
        """Pin a matched checkpoint until its restore copy is queued."""
        restore = seq.prefix_cache.restore
        if not restore.is_selected or restore.pinned:
            return False
        node = restore.node
        if not self._is_published(node, restore.slot):
            return False
        checkpoint = node.state_checkpoint
        checkpoint.pin_count += 1
        checkpoint.last_access_time = time.perf_counter()
        restore.pinned = True
        logger.debug('Pin SSM prefix-cache restore checkpoint: session_id=%s seq_id=%s step=%s state_idx=%s '
                     'pin_count=%s', seq.session_id, seq.seq_id, node.prefix_len, checkpoint.slot,
                     checkpoint.pin_count)
        return True

    def pin_restores(self, seqs: list[SchedulerSequence]):
        """Pin matched checkpoints for a batch."""
        for seq in seqs:
            self.pin_restore(seq)

    def unpin_restore(self, seq: SchedulerSequence):
        """Unpin a checkpoint after its restore copy is queued."""
        restore = seq.prefix_cache.restore
        if not restore.pinned:
            return False
        node = self._unpin_checkpoint(
            restore.node,
            restore.slot,
            'Pinned SSM prefix-cache restore checkpoint lost its node reference.',
        )
        checkpoint = node.state_checkpoint
        logger.debug('Unpin SSM prefix-cache restore checkpoint: session_id=%s seq_id=%s step=%s state_idx=%s '
                     'pin_count=%s', seq.session_id, seq.seq_id, node.prefix_len, checkpoint.slot,
                     checkpoint.pin_count)
        restore.clear()
        return True

    def unpin_restores(self, seqs: list[SchedulerSequence]):
        """Unpin checkpoints after a batch restore copy is queued."""
        if not self._prefix_cache_enabled:
            return
        for seq in seqs:
            self.unpin_restore(seq)

    def unpin_save(self, seq: SchedulerSequence):
        """Unpin a checkpoint after its producer save copy is queued."""
        producer_pin = seq.prefix_cache.producer_save_pin
        if not producer_pin.is_acquired:
            return False
        node = self._unpin_checkpoint(
            producer_pin.node,
            producer_pin.slot,
            'Pinned SSM prefix-cache save checkpoint lost its node reference.',
        )
        checkpoint = node.state_checkpoint
        logger.debug('Unpin SSM prefix-cache save checkpoint: session_id=%s seq_id=%s step=%s state_idx=%s '
                     'pin_count=%s', seq.session_id, seq.seq_id, node.prefix_len, checkpoint.slot,
                     checkpoint.pin_count)
        producer_pin.clear()
        return True

    def unpin_saves(self, seqs: list[SchedulerSequence]):
        """Unpin producer checkpoints after batched save copies are queued."""
        if not self._prefix_cache_enabled:
            return
        for seq in seqs:
            self.unpin_save(seq)

    @staticmethod
    def is_pinned(node: Node):
        """Whether an async save or restore still pins a checkpoint."""
        checkpoint = node.state_checkpoint
        return checkpoint is not None and checkpoint.pin_count > 0

    def release_checkpoint(self, node: Node):
        """Release a node-owned state checkpoint while keeping its KV node."""
        checkpoint = node.state_checkpoint
        if checkpoint is None:
            return
        if checkpoint.pin_count > 0:
            raise RuntimeError('Cannot release a pinned SSM prefix-cache checkpoint.')
        if checkpoint.slot < 0:
            if checkpoint.published:
                self._warn_unexpected_state(
                    f'published SSM checkpoint has no state slot: adapter={node.adapter_name} '
                    f'step={node.prefix_len}')
                self._unindex_checkpoint(node)
            node.state_checkpoint = None
            return
        if checkpoint.published:
            self._unindex_checkpoint(node)
        self._state_manager.free_checkpoint_state(checkpoint.slot)
        node.state_checkpoint = None

    def evict(self, max_num_states: int):
        """Evict published state checkpoints without removing KV trie nodes."""
        if not self._state_checkpoints_enabled or max_num_states <= 0:
            return 0

        candidates = [(node.state_checkpoint.last_access_time, id(node), node) for node in self._index.unique_nodes()
                      if self._is_evictable_checkpoint(node)]
        heapq.heapify(candidates)

        evicted = 0
        while candidates and evicted < max_num_states:
            _, _, node = heapq.heappop(candidates)
            if not self._is_evictable_checkpoint(node):
                continue
            checkpoint = node.state_checkpoint
            logger.debug('Evict SSM prefix-cache checkpoint: adapter=%s step=%s state_idx=%s', node.adapter_name,
                         node.prefix_len, checkpoint.slot)
            self.release_checkpoint(node)
            evicted += 1
        return evicted

    def discard_stale_index_entry(self, node: Node, key: StateCheckpointKey, reason: str):
        """Remove a stale index entry without releasing a valid checkpoint."""
        removed = self._index.remove_entry(node, key)
        if removed:
            checkpoint = node.state_checkpoint
            slot = -1 if checkpoint is None else checkpoint.slot
            logger.debug('Drop stale SSM prefix-cache checkpoint index entry: adapter=%s step=%s node_adapter=%s '
                         'node_step=%s state_idx=%s reason=%s', key[0], key[1], node.adapter_name, node.prefix_len,
                         slot, reason)
        return removed

    def release_stale_checkpoint(self, node: Node, reason: str):
        """Release a globally stale checkpoint candidate when it is
        unpinned."""
        if self.is_pinned(node):
            checkpoint = node.state_checkpoint
            logger.debug('Skip pinned stale SSM prefix-cache checkpoint candidate: adapter=%s step=%s '
                         'state_idx=%s pin_count=%s reason=%s', node.adapter_name, node.prefix_len, checkpoint.slot,
                         checkpoint.pin_count, reason)
            return False

        checkpoint = node.state_checkpoint
        if checkpoint is None:
            self._unindex_checkpoint(node)
            return False
        slot = checkpoint.slot
        was_published = checkpoint.published
        self._unindex_checkpoint(node)
        if slot >= 0:
            self._state_manager.free_checkpoint_state(slot)
        node.state_checkpoint = None
        logger.debug('Release stale SSM prefix-cache checkpoint candidate: adapter=%s step=%s state_idx=%s '
                     'was_published=%s reason=%s', node.adapter_name, node.prefix_len, slot, was_published,
                     reason)
        return slot >= 0 or was_published

    @staticmethod
    def _find_checkpoint_node(seq: SchedulerSequence, step: int):
        """Find the attached trie node at an exact sequence step."""
        node = seq.prefix_cache.trie_cursor
        while node is not None and node.prefix_len > step:
            node = node.parent
        if node is None or not node.is_attached() or node.prefix_len != step:
            return None
        return node

    def _release_replaceable_decode_checkpoint(self, seq: SchedulerSequence, new_step: int):
        """Release the previous decode checkpoint when replacement is safe."""
        prefix_cache = seq.prefix_cache
        old_node = prefix_cache.decode_checkpoint_node
        if old_node is None:
            return True
        old_checkpoint = old_node.state_checkpoint
        if old_checkpoint is None or old_checkpoint.slot < 0:
            prefix_cache.decode_checkpoint_node = None
            return True
        if old_node.prefix_len == new_step and old_checkpoint.published:
            return False

        if old_checkpoint.pin_count > 0:
            return False
        logger.debug('Release previous decode SSM prefix-cache checkpoint: session_id=%s seq_id=%s '
                     'old_step=%s old_state_idx=%s new_step=%s', seq.session_id, seq.seq_id, old_node.prefix_len,
                     old_checkpoint.slot, new_step)
        self.release_checkpoint(old_node)
        prefix_cache.decode_checkpoint_node = None
        return True

    def _reserve_slot(self, node: Node):
        """Reserve a state-cache slot owned by a trie node.

        Replacing a published checkpoint is allowed only while no async copy pins it. If the shared state pool is full,
        one old unpinned checkpoint may be evicted without removing its trie/KV node.
        """
        if not self._state_checkpoints_enabled or node.parent is None:
            return -1
        checkpoint = node.state_checkpoint
        if checkpoint is not None and checkpoint.published:
            if checkpoint.pin_count > 0:
                return -1
            logger.debug('Replace SSM prefix-cache checkpoint: adapter=%s step=%s state_idx=%s', node.adapter_name,
                         node.prefix_len, checkpoint.slot)
            self._unindex_checkpoint(node)
        elif checkpoint is not None:
            return -1
        if checkpoint is None:
            if self._state_manager.get_num_free_checkpoint() == 0 and self.evict(1) == 0:
                return -1
            slot = self._state_manager.allocate_checkpoint_state()
            checkpoint = NodeStateCheckpoint(slot=slot)
            node.state_checkpoint = checkpoint
        checkpoint.published = False
        checkpoint.exact_match_data = None
        checkpoint.last_access_time = 0.0
        return checkpoint.slot

    def _publish_checkpoint(self, node: Node, seq: SchedulerSequence):
        """Publish a node-owned checkpoint after its state copy is queued."""
        checkpoint = node.state_checkpoint
        if checkpoint is None or checkpoint.slot < 0:
            raise RuntimeError('Cannot publish an unreserved state checkpoint.')
        if checkpoint.pin_count != 0:
            raise RuntimeError('Cannot publish a pinned SSM prefix-cache checkpoint.')
        if not node.is_attached():
            raise RuntimeError('Cannot publish a detached SSM prefix-cache checkpoint node.')
        if checkpoint.published:
            if checkpoint.exact_match_data is None:
                raise RuntimeError('Cannot republish an SSM checkpoint with missing exact-match metadata.')
            return

        match_data = self._make_sequence_match_data(node, seq)
        checkpoint.exact_match_data = match_data
        checkpoint.published = True
        checkpoint.last_access_time = time.perf_counter()
        try:
            self._index.add(node)
        except Exception:
            # Publication is transactional: the caller still owns an
            # unpublished reservation and may release its slot after rollback.
            self._unindex_checkpoint(node)
            checkpoint.published = False
            checkpoint.exact_match_data = None
            checkpoint.last_access_time = 0.0
            raise

    def _pin_save(self, seq: SchedulerSequence, node: Node, slot: int):
        producer_pin = seq.prefix_cache.producer_save_pin
        if producer_pin.is_acquired:
            raise RuntimeError('SSM prefix-cache save checkpoint already has an in-flight producer pin.')
        if not self._is_published(node, slot):
            return False
        checkpoint = node.state_checkpoint
        checkpoint.pin_count += 1
        checkpoint.last_access_time = time.perf_counter()
        producer_pin.acquire(slot, node)
        logger.debug('Pin SSM prefix-cache save checkpoint: session_id=%s seq_id=%s step=%s state_idx=%s '
                     'pin_count=%s', seq.session_id, seq.seq_id, node.prefix_len, slot, checkpoint.pin_count)
        return True

    @classmethod
    def _unpin_checkpoint(cls, node: Node | None, slot: int, err_msg: str):
        checkpoint = None if node is None else node.state_checkpoint
        if checkpoint is None or checkpoint.slot != slot or checkpoint.pin_count <= 0:
            cls._warn_unexpected_state(f'{err_msg} state_idx={slot}')
            raise RuntimeError(err_msg)
        checkpoint.pin_count -= 1
        return node

    def _release_invalid_reservation(self,
                                     seq: SchedulerSequence,
                                     node: Node | None,
                                     slot: int,
                                     is_decode: bool):
        if not self._is_unpublished_reservation(node, slot):
            return
        if is_decode and seq.prefix_cache.decode_checkpoint_node is node:
            seq.prefix_cache.decode_checkpoint_node = None
        self.release_checkpoint(node)

    def _publication_invalid_reason(self, node: Node | None, slot: int, save_step: int):
        if node is None:
            return 'missing node'
        if not node.is_attached():
            return 'detached node'
        checkpoint = node.state_checkpoint
        if checkpoint is None:
            return 'missing state checkpoint'
        if checkpoint.slot != slot:
            return f'slot changed: expected={slot} actual={checkpoint.slot}'
        if node.prefix_len != save_step:
            return f'step changed: expected={save_step} actual={node.prefix_len}'
        return None

    @staticmethod
    def _node_has_slot(node: Node):
        checkpoint = node.state_checkpoint
        return checkpoint is not None and checkpoint.slot >= 0

    @staticmethod
    def _is_unpublished_reservation(node: Node | None, slot: int):
        if node is None:
            return False
        checkpoint = node.state_checkpoint
        return checkpoint is not None and checkpoint.slot == slot and not checkpoint.published

    @staticmethod
    def _is_published(node: Node | None, slot: int):
        if node is None:
            return False
        checkpoint = node.state_checkpoint
        return checkpoint is not None and checkpoint.slot == slot and checkpoint.published

    @staticmethod
    def _is_evictable_checkpoint(node: Node):
        checkpoint = node.state_checkpoint
        return checkpoint is not None and checkpoint.slot >= 0 and checkpoint.published and checkpoint.pin_count == 0

    def _unindex_checkpoint(self, node: Node):
        """Remove a checkpoint from every sparse-index bucket."""
        return self._index.remove(node)

    @staticmethod
    def _warn_unexpected_state(message: str):
        logger.warning('Unexpected prefix-cache trie state: %s', message)
