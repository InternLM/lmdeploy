# Copyright (c) OpenMMLab. All rights reserved.
"""Trie ownership of direct compact units; no state snapshot copy interface."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lmdeploy.pytorch.messages import SchedulerSequence

from ..compact_state import SSMStateManager, StateAccess, StateBank, StateReference
from .checkpoint import checkpoint_anchor_step
from .checkpoint_lifecycle import StateCheckpointLifecycle
from .node import Node, NodeStateCheckpoint


@dataclass
class _DecodeCheckpointReservation:
    """Future aligned output, before its token identity reaches the CPU
    trie."""
    seq: SchedulerSequence
    bank: StateBank
    reference: StateReference
    old_node: Node | None
    old_checkpoint: NodeStateCheckpoint | None
    published_node: Node | None = None


@dataclass(frozen=True)
class _ForwardCheckpointHolds:
    """Physical pin identities whose lifetime differs from mutable
    sequences."""
    access: StateAccess
    restores: tuple[tuple[StateBank, StateReference, NodeStateCheckpoint], ...]
    saves: tuple[tuple[SchedulerSequence, Node, NodeStateCheckpoint], ...]
    inputs: tuple[tuple[StateBank, StateReference], ...]
    decode_saves: tuple[_DecodeCheckpointReservation, ...] = ()


class SSMStateCheckpointLifecycle(StateCheckpointLifecycle):
    """Reuse exact-identity/KV policy, replacing only state resource ownership.

    A node owns a reservation until its whole forward is submitted, then an
    immutable trie reference. Consumers acquire their own physical references.
    Unpublished failed producers can retire while still executing: their state
    access holds and deferred frozen-KV tails survive until real completion.
    """

    def __init__(self, *, attach_decode_checkpoint=None, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(self._state_manager, SSMStateManager):
            raise TypeError('Compact checkpoints require the unified state owner.')
        self._attach_decode_checkpoint = attach_decode_checkpoint
        self._retired_tails: list[NodeStateCheckpoint] = []
        self._forwards: dict[int, _ForwardCheckpointHolds] = {}
        self._failed_nodes: dict[Node, NodeStateCheckpoint] = {}

    def bind_restore(self, seq):
        """Bind a selected unit without copying; keep its KV pin until
        completion.

        The returned ID is a one-forward source override, not a repeated decode bind command. Call unpin_restore only
        after this forward's all-rank completion, unlike the legacy queued-copy dispatch shortcut.
        """
        restore = seq.prefix_cache.restore
        if not restore.is_selected:
            return -1
        if not StateCheckpointLifecycle._is_published(restore.node, restore.slot):
            raise RuntimeError('Compact restore checkpoint is no longer published.')
        checkpoint = restore.node.state_checkpoint
        reference = checkpoint.state_reference
        if reference is None or reference.unit_id != restore.slot or not reference.active:
            raise RuntimeError('Compact restore lost its physical unit identity.')
        if reference.phase not in ('submitted', 'ready'):
            raise RuntimeError('Compact restore checkpoint producer failed or is not submitted.')
        bank = self._state_manager.bank_for(seq)
        acquired = not restore.pinned
        if acquired and not self.pin_restore(seq):
            raise RuntimeError('Compact restore checkpoint is no longer published.')
        try:
            self._state_manager.bind_input(bank, reference)
        except Exception:
            # Preserve any earlier admission pin and the selected identity.
            # No await occurs here: this is the exact checkpoint just pinned.
            if acquired:
                checkpoint.pin_count -= 1
                restore.pinned = False
            raise
        return reference.unit_id

    def attach_forward(self, access, seqs, *, final_checkpoints=(), decode_reservations=()):
        """Transfer restore pins from admission into an exact submitted access.

        Called last in CPU input preparation. Clearing the request's selection prevents prefetch from rebinding it and
        cancellation from unpinning an in-flight consumer. The physical pin is released only by completion.
        """
        if access.finished or access.access_id in self._forwards:
            raise ValueError('Compact checkpoint forward must be attached exactly once before completion.')
        restores, saves, selections = [], [], []
        for seq in seqs:
            restore, pending = seq.prefix_cache.restore, seq.prefix_cache.pending_save
            if restore.is_selected:
                bank = self._state_manager.bank_for(seq)
                checkpoint = restore.node.state_checkpoint
                if (not restore.pinned or checkpoint is None or checkpoint.slot != restore.slot
                        or bank.input_ref is None or bank.input_ref.unit_id != restore.slot):
                    raise ValueError('Compact forward lost its acquired restore pin or input reference.')
                restores.append((bank, bank.input_ref, checkpoint))
                selections.append(restore)
            if pending.is_pending:
                checkpoint = pending.node.state_checkpoint
                if checkpoint is None or checkpoint.slot != pending.slot:
                    raise ValueError('Compact forward lost its checkpoint destination.')
                saves.append((seq, pending.node, checkpoint))
        # Decode consumption is retired by the owner's completed target-token
        # mask; unlike positive prefill, completion alone must preserve a=0.
        inputs = (() if decode_reservations else
                  tuple((bank, bank.input_ref) for bank in access._banks if bank.input_ref is not None))
        if final_checkpoints:
            banks = tuple(self._state_manager.bank_for(seq) for seq in seqs)
            self._state_manager.bind_produced_inputs(access, banks, final_checkpoints)
        self._forwards[access.access_id] = _ForwardCheckpointHolds(
            access, tuple(restores), tuple(saves), inputs, tuple(decode_reservations))
        for restore in selections:
            restore.clear()

    def publish_forward(self, access_id):
        """Publish only this ordered dispatch's still-owned reservations."""
        hold = self._forwards[access_id]
        for seq, node, checkpoint in hold.saves:
            pending = seq.prefix_cache.pending_save
            if (node.state_checkpoint is checkpoint and pending.node is node and pending.slot == checkpoint.slot):
                self.publish_save(seq)
        for reservation in hold.decode_saves:
            self._publish_decode(reservation)

    def complete_forward(self, access_id):
        """Release exact pins after the owner has observed all-rank
        completion."""
        hold = self._forwards.get(access_id)
        if hold is None:
            return
        if not hold.access.finished:
            raise RuntimeError('Compact checkpoint pins require actual owner completion.')
        del self._forwards[access_id]
        if hold.decode_saves:
            # A failed/discarded dispatch may never have reached publication.
            # Actual completion now permits releasing its reservation/old pin.
            self.discard_decode(hold.decode_saves)
        for _, _, checkpoint in hold.restores:
            checkpoint.pin_count -= 1
        for bank, reference in hold.inputs:
            # Positive-token prefill has produced private progress. A later
            # submitted read owns its own hold; a later binding must survive.
            if not bank.closing and bank.input_ref is reference:
                self._state_manager.clear_input(bank)
        self.collect_completed()

    def finish_cancelled_forwards_after_worker_drain(self):
        """Unwind exact pins after quiescence, never publish discarded
        outputs."""
        self._state_manager.finish_cancelled_forwards_after_worker_drain()
        holds = tuple(self._forwards.items())
        for access_id, _ in holds:
            self.complete_forward(access_id)
        # Session cancellation normally discarded these reservations already.
        # Keep exact identity checks: another node publication must not be
        # released through a stale request-local pending_save object.
        for _, hold in holds:
            for seq, node, checkpoint in hold.saves:
                if node.state_checkpoint is checkpoint and not checkpoint.published:
                    self.release_checkpoint(node)
                pending = seq.prefix_cache.pending_save
                if pending.node is node and pending.slot == checkpoint.slot:
                    pending.clear()
        self.collect_completed()

    def prepare_prefill(self, seqs, steps):
        """Reserve direct outputs and expose only the necessary KV-tail copies.

        Caller validates kernel export boundaries before this transaction and rolls back reservations if later metadata
        construction fails. Source pins and request references are acquired separately, at final binding.
        """
        if len(seqs) != len(steps):
            raise ValueError('Checkpoint boundaries must match prefill rows.')
        restores, saves = [], []
        try:
            for seq, step in zip(seqs, steps):
                restore = seq.prefix_cache.restore
                if restore.is_selected:
                    if not self._is_published(restore.node, restore.slot):
                        raise RuntimeError('Compact restore checkpoint is no longer published.')
                    checkpoint = restore.node.state_checkpoint
                    if checkpoint.frozen_block_id >= 0:
                        block = checkpoint.step // self._block_size
                        if block >= len(seq.logical_blocks):
                            raise RuntimeError('Compact restore destination KV block is missing.')
                        restores.append((checkpoint.frozen_block_id, int(seq.logical_blocks[block])))
                # One live reservation per bank. Once publication transfers it
                # to Trie/Store ownership, prefetch may reserve another unit;
                # earlier producer holds still protect their own destinations.
                if self._state_manager.bank_for(seq).pending_checkpoint is not None:
                    continue
                if self.reserve_save(seq, step=step) < 0:
                    continue
                checkpoint = seq.prefix_cache.pending_save.node.state_checkpoint
                if checkpoint.frozen_block_id >= 0:
                    block = checkpoint.step // self._block_size
                    if block >= len(seq.logical_blocks):
                        raise RuntimeError('Compact save source KV block is missing.')
                    saves.append((int(seq.logical_blocks[block]), checkpoint.frozen_block_id))
        except Exception:
            for seq in seqs:
                self.discard_save(seq)
            raise
        return restores, saves

    def prepare_decode(self, seqs, steps):
        """Reserve upcoming AR finals without guessing their not-yet-known
        tokens.

        Prefetch can precede the previous output's CPU token append. Reserve only the physical destination here; attach
        the exact trie identity at ordered publication after that append. The old decode node stays pinned throughout
        reservation and cannot be evicted to fake replacement space.
        """
        if len(seqs) != len(steps) or self._state_manager.capacity.max_update_tokens != 1:
            raise ValueError('Compact decode saves require matching single-token AR rows.')
        reservations = []
        try:
            for seq, step in zip(seqs, steps):
                bank = self._state_manager.bank_for(seq)
                alignment = seq._seq_meta.prefix_cache_checkpoint_block_size
                if (step <= 0 or step % self._block_size or (alignment and step % alignment)
                        or not self._prefix_cache_enabled or not self._state_checkpoints_enabled
                        or seq._seq_meta.prefix_cache_token_lookahead
                        or seq.clamp_prefix_cache_match_step(step) != step
                        or bank.pending_checkpoint is not None):
                    continue
                old_node = seq.prefix_cache.decode_checkpoint_node
                old = None if old_node is None else old_node.state_checkpoint
                if old is not None and (self.is_pinned(old_node) or old.step >= step):
                    continue
                if old is not None:
                    old.pin_count += 1
                reference = None
                try:
                    while self._state_manager.get_num_free_optional() == 0:
                        if self.evict(1) == 0:
                            break
                    if self._state_manager.get_num_free_optional():
                        reference = self._state_manager.reserve_checkpoint(bank, step)
                finally:
                    if reference is None and old is not None:
                        old.pin_count -= 1
                if reference is not None:
                    reservations.append(_DecodeCheckpointReservation(seq, bank, reference, old_node, old))
        except Exception:
            self.discard_decode(reservations)
            raise
        return tuple(reservations)

    def publish_decode_snapshot(self, seq, step, reference):
        """Publish an accepted exact-boundary unit after its forward
        completes."""
        if not reference.active or reference.phase != 'ready' or reference.boundary != step:
            return False
        node = self._attach_decode_checkpoint(seq, step)
        if node is None or self._node_has_slot(node):
            return False
        if not self._release_replaceable_decode_checkpoint(seq, step):
            return False
        node.state_checkpoint = NodeStateCheckpoint(slot=reference.unit_id, step=step, state_reference=reference)
        try:
            self._publish_checkpoint(node, seq)
        except Exception:
            node.state_checkpoint = None
            raise
        seq.prefix_cache.decode_checkpoint_node = node
        seq.prefix_cache.last_decode_save_step = max(seq.prefix_cache.last_decode_save_step, step)
        return True

    def discard_decode(self, reservations):
        """Drop tentative publication, never its submitted GPU/reader holds."""
        for reservation in reservations:
            reference = reservation.reference
            if reference.active:
                if reference.phase == 'reserved':
                    self._state_manager.cancel_checkpoint(reference)
                else:
                    self._state_manager.release_reference(reference)
            if reservation.old_checkpoint is not None:
                reservation.old_checkpoint.pin_count -= 1
                reservation.old_checkpoint = None

    def _publish_decode(self, reservation):
        reference, seq = reservation.reference, reservation.seq
        if not reference.active:
            self.discard_decode((reservation,))
            return
        old_node, old = reservation.old_node, reservation.old_checkpoint
        try:
            bank = reservation.bank
            # Cancellation can free the sequence while a prefetched producer
            # remains in flight. Its physical output/read holds still survive.
            if bank.closing or reference.phase not in ('submitted', 'ready'):
                return
            if seq.num_valid_ids < reference.boundary:
                raise RuntimeError('Decode checkpoint token identity has not reached its reserved boundary.')
            if old is not None and old.pin_count != 1:
                return  # A new reader selected the previous checkpoint meanwhile.
            if self._attach_decode_checkpoint is None:
                raise RuntimeError('Compact decode publication requires its trie attachment owner.')
            node = self._attach_decode_checkpoint(seq, reference.boundary)
            if node is None or node.state_checkpoint is not None:
                return  # Existing identical publication wins without being overwritten.
            checkpoint = NodeStateCheckpoint(slot=reference.unit_id, step=reference.boundary,
                                             frozen_block_id=-1, state_reference=reference)
            node.state_checkpoint = checkpoint
            try:
                self._publish_checkpoint(node, seq)
            except Exception:
                if node.state_checkpoint is checkpoint:
                    self.release_checkpoint(node)
                raise
            # Only the new publication can retire the old logical node. Its
            # independent physical readers remain protected by the state owner.
            if old is not None:
                old.pin_count -= 1
                reservation.old_checkpoint = None
                if old_node.state_checkpoint is old:
                    self.release_checkpoint(old_node)
            seq.prefix_cache.decode_checkpoint_node = node
            reservation.published_node = node
        finally:
            self.discard_decode((reservation,))

    def fail_forward(self, access_id):
        """Withdraw failed publications and queued descendants, retaining
        holds."""
        affected = self._state_manager.invalidate(access_id)
        for failed_id in affected:
            hold = self._forwards.get(failed_id)
            if hold is None:
                continue
            for _, node, checkpoint in hold.saves:
                if node.state_checkpoint is checkpoint:
                    self._withdraw_failed_checkpoint(node)
            for reservation in hold.decode_saves:
                node = reservation.published_node
                checkpoint = None if node is None else node.state_checkpoint
                if (checkpoint is not None
                        and checkpoint.state_reference.shares_storage(reservation.reference)):
                    self._withdraw_failed_checkpoint(node)
        return affected

    def _withdraw_failed_checkpoint(self, node):
        checkpoint = node.state_checkpoint
        checkpoint.published = False
        self._unindex_checkpoint(node)
        self._failed_nodes[node] = checkpoint
        # Producer/read pins and physical holds are intentionally unchanged.

    @staticmethod
    def _is_published(node, slot):
        if not StateCheckpointLifecycle._is_published(node, slot):
            return False
        reference = node.state_checkpoint.state_reference
        return reference is not None and reference.active and reference.phase in ('submitted', 'ready')

    def release_stale_checkpoint(self, node, reason):
        checkpoint = node.state_checkpoint
        reference = None if checkpoint is None else checkpoint.state_reference
        if reference is not None and (not reference.active or reference.phase not in ('submitted', 'ready')):
            self._withdraw_failed_checkpoint(node)
            self.collect_completed()
            return node.state_checkpoint is None
        return super().release_stale_checkpoint(node, reason)

    def unpin_restore(self, seq):
        changed = super().unpin_restore(seq)
        self.collect_completed()
        return changed

    def collect_completed(self):
        """Release retired partial-KV tails only after owner-reported
        completion."""
        pending = []
        for checkpoint in self._retired_tails:
            if checkpoint.state_reference.producing:
                pending.append(checkpoint)
            else:
                self._allocator.free(np.array([checkpoint.frozen_block_id], dtype=np.int64))
        self._retired_tails = pending
        for node, checkpoint in tuple(self._failed_nodes.items()):
            if node.state_checkpoint is not checkpoint:
                del self._failed_nodes[node]
            elif not self.is_pinned(node):
                self.release_checkpoint(node)

    def make_runtime_state_available(self):
        """Reclaim enough physical units for a full bank, not one legacy
        slot."""
        owner = self._state_manager
        if owner.get_num_free_runtime_slots() == 0:
            return False  # Checkpoint eviction cannot release request slots.
        while owner.get_num_free_runtime() == 0:
            if self.evict(1) == 0:
                return False
        return True

    def prepare_restore_batch(self, seqs):
        raise NotImplementedError('Compact restore binds physical references; it cannot produce state-copy plans.')

    def reserve_prefill_save_batch(self, seqs, steps=None):
        raise NotImplementedError('Compact prefill produces units directly, not state-copy plans.')

    def reserve_decode_save_batch(self, seqs, interval):
        raise NotImplementedError('Compact decode reserves direct final IDs, not state-copy plans.')

    def finish_forward_dispatch(self, seqs, *, has_save_plan):
        raise NotImplementedError('Compact restore/producer pins require actual completion, not queued-copy dispatch.')

    @staticmethod
    def is_pinned(node):
        checkpoint = node.state_checkpoint
        return (StateCheckpointLifecycle.is_pinned(node)
                or (checkpoint is not None and checkpoint.state_reference is not None
                    and checkpoint.state_reference.producing))

    def _allocate_checkpoint(self, node, step, seq):
        """Reserve the indivisible state/tail bundle before changing trie
        truth."""
        if seq is None:
            raise ValueError('Compact checkpoint reservation requires its admitted producer sequence.')
        owner = self._state_manager
        bank = owner.bank_for(seq)
        self.collect_completed()
        while owner.get_num_free_optional() == 0:
            # Replacement cannot overwrite/reuse a unit that a detached node's
            # consumer still holds. Eviction may drop only a logical reference,
            # so keep checking physical capacity. Never evict our old node to
            # fake replacement atomicity.
            candidates = (candidate for candidate in self._index.unique_nodes() if candidate is not node)
            if self._evict_checkpoints(candidates, 1) == 0:
                return None
        reference = owner.reserve_checkpoint(bank, step)
        try:
            tail = (-1 if checkpoint_anchor_step(step, self._block_size) == step else
                    int(self._allocator.allocate(1, 'gpu')[0]))
        except Exception:
            owner.cancel_checkpoint(reference)
            raise
        return NodeStateCheckpoint(slot=reference.unit_id, step=step, frozen_block_id=tail,
                                   state_reference=reference)

    def _create_checkpoint(self, node, step, *, seq=None):
        checkpoint = self._allocate_checkpoint(node, step, seq)
        if checkpoint is None:
            return -1
        node.state_checkpoint = checkpoint
        return checkpoint.slot

    def _replace_checkpoint(self, node, checkpoint, step, *, seq=None):
        replacement = self._allocate_checkpoint(node, step, seq)
        if replacement is None:
            return -1
        self.release_checkpoint(node)
        node.state_checkpoint = replacement
        return replacement.slot

    def _publish_checkpoint(self, node, seq):
        checkpoint = node.state_checkpoint
        if checkpoint is None or checkpoint.state_reference is None:
            raise RuntimeError('Compact publication requires a physical checkpoint reservation.')
        reference = checkpoint.state_reference
        if reference.phase not in ('submitted', 'ready'):
            raise RuntimeError('Compact publication requires a submitted producer or completed boundary.')
        if checkpoint.published:
            return super()._publish_checkpoint(node, seq)
        owner = self._state_manager
        trie_reference = owner.retain(reference, 'trie')
        try:
            super()._publish_checkpoint(node, seq)
        except Exception:
            owner.release_reference(trie_reference)
            raise
        checkpoint.state_reference = trie_reference
        owner.release_reference(reference)

    def release_checkpoint(self, node):
        checkpoint = node.state_checkpoint
        if checkpoint is None:
            return
        if checkpoint.pin_count:
            raise RuntimeError('Cannot release a pinned SSM prefix-cache checkpoint.')
        reference = checkpoint.state_reference
        if reference is None:
            raise RuntimeError('Compact trie checkpoint lost its physical reference.')
        self._unindex_checkpoint(node)
        self._failed_nodes.pop(node, None)
        node.state_checkpoint = None
        if reference.phase == 'reserved':
            self._state_manager.cancel_checkpoint(reference)
        else:
            self._state_manager.release_reference(reference)
        if checkpoint.frozen_block_id >= 0:
            if reference.producing:
                self._retired_tails.append(checkpoint)
            else:
                self._allocator.free(np.array([checkpoint.frozen_block_id], dtype=np.int64))
