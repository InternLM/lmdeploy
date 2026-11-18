# Copyright (c) OpenMMLab. All rights reserved.
"""Paging ownership of PD exports and receives; callbacks never own memory."""
import hashlib
import json
from dataclasses import dataclass, field
from time import monotonic, time
from uuid import uuid4

import numpy as np

from lmdeploy.pytorch.disagg.config import PD_RELEASE_TIMEOUT, PD_TRANSFER_TIMEOUT
from lmdeploy.pytorch.disagg.conn.protocol import PDControlRequest, PDHandoff
from lmdeploy.pytorch.disagg.messages import PDReceiveRequest

from .compact_state import SSMStateManager, StateAccess


def prompt_digest(seq, tokens) -> str:
    """Bind a handoff to tokens, adapter and multimodal content identity."""
    identity = (seq.adapter_name, seq.get_prefix_cache_multimodal_identity(0, len(tokens)))
    digest = hashlib.sha256(np.asarray(tokens, dtype='<i8').tobytes())
    digest.update(json.dumps(identity, default=str, sort_keys=True).encode())
    return digest.hexdigest()


@dataclass
class PDExport:
    sequence: object
    handoff: PDHandoff
    logical_blocks: np.ndarray
    state_access: StateAccess | None
    receiver_epoch: str = ''
    receiver_engine_id: str = ''
    revoked: bool = False
    renewed_at: float = field(default_factory=monotonic)
    reason: str = ''


@dataclass
class PDReceive:
    sequence: object
    request: PDReceiveRequest
    logical_blocks: np.ndarray
    state_access: StateAccess | None
    cancelled: str = ''
    failed_at: float | None = None
    error: str = ''

    @property
    def quarantined(self):
        return self.failed_at is not None


class PDCoordinator:
    """Serialize grants/cancellation on the scheduler loop and pin allocations.

    Failed receives and abandoned grants have a bounded hold. Expiry deliberately favors availability over proof of DMA
    completion; it releases only this transaction's references, never another Store or request owner's holds.
    """

    def __init__(self, block_manager, state_manager):
        self.block_manager = block_manager
        self.state_manager = state_manager
        self.epoch = uuid4().hex
        self.exports: dict[str, PDExport] = {}
        self._export_by_sequence = {}
        self.receives: dict[int, PDReceive] = {}
        self.release_timeout = PD_RELEASE_TIMEOUT
        self.transfer_timeout = PD_TRANSFER_TIMEOUT
        self.forced_releases = 0
        self.notifications: list[tuple[str, PDControlRequest]] = []

    def validate_receive(self, seq, cache_schema):
        handoff = seq.migration_request.handoff
        if (handoff.cache_schema != cache_schema or handoff.block_size != seq.block_size
                or handoff.prompt_token_count != seq.num_valid_ids
                or handoff.prompt_digest != prompt_digest(seq, seq.valid_ids)):
            raise ValueError('PD handoff does not match this prompt or execution schema.')

    def prepare_receive(self, seq):
        """Pin the matched prefix and private destinations before granting
        I/O."""
        migration = seq.migration_request
        handoff = migration.handoff
        matched = seq.num_history_ids // seq.block_size
        local = self.block_manager.get_block_table(seq)
        pairs = list(zip(handoff.block_ids[matched:], local[matched:len(handoff.block_ids)].tolist()))
        if len(pairs) != len(handoff.block_ids) - matched and handoff.block_ids:
            raise ValueError('PD receiver has insufficient private block destinations.')
        access = None
        state_id = auxiliary_id = legacy_id = None
        if handoff.valid_token_count and seq.logical_state >= 0:
            if isinstance(self.state_manager, SSMStateManager):
                bank = self.state_manager.bank_for(seq)
                access = self.state_manager.submit_load(bank, handoff.valid_token_count)
                state_id, auxiliary_id = bank.unit_ids[0], bank.auxiliary_id
            else:
                legacy_id = seq.logical_state
                self.state_manager.retain_state(legacy_id)
        request = PDReceiveRequest(remote_engine_id=migration.remote_engine_id, handoff=handoff,
                                    block_pairs=pairs, state_id=state_id, auxiliary_id=auxiliary_id,
                                    legacy_state_id=legacy_id, deadline=time() + self.transfer_timeout)
        blocks = seq.logical_blocks.get_real_blocks().copy()
        self.block_manager.pin_logical_blocks(blocks)
        self.receives[seq.seq_id] = PDReceive(seq, request, blocks, access)

    def finish_receive(self, seq, result) -> str:
        """A late completion may release its own hold, but never a newer
        receive."""
        receive = self.receives.get(seq.seq_id)
        if receive is None or result.transfer_id != receive.request.handoff.transfer_id:
            return 'stale'
        if result.status == 'completion_unknown':
            self.quarantine_receive(seq, result.error or 'PD native completion is unknown.')
            return 'quarantined'
        success = result.status == 'completed' and not receive.cancelled and not receive.quarantined
        self._release_receive(receive, success=success)
        if success:
            seq.set_step(receive.request.handoff.valid_token_count)
            seq.migration_request = None
            seq.kv_token_limit = None
            return 'ready'
        return receive.cancelled or 'error'

    def quarantine_receive(self, seq, reason):
        receive = self.receives[seq.seq_id]
        if receive.failed_at is None:
            receive.failed_at = monotonic()
            receive.error = reason

    def expire_receive(self, seq, transfer_id, now):
        """Apply the availability deadline without claiming native DMA
        drained."""
        receive = self.receives.get(seq.seq_id)
        if (receive is None or receive.request.handoff.transfer_id != transfer_id
                or receive.failed_at is None or now - receive.failed_at < self.release_timeout):
            return False
        self._release_receive(receive, success=False)
        self.forced_releases += 1
        return True

    def _release_receive(self, receive, *, success):
        if receive.state_access is not None:
            self.state_manager.complete(receive.state_access, success=success)
        elif receive.request.legacy_state_id is not None:
            self.state_manager.free_state(receive.request.legacy_state_id)
        self.block_manager.release_logical_blocks(receive.logical_blocks)
        del self.receives[receive.sequence.seq_id]

    def publish(self, seq, cache_schema, state_ids) -> PDHandoff:
        existing = self._export_by_sequence.get(seq.seq_id)
        if existing is not None:
            return self.exports[existing].handoff
        tokens = seq.pd_prompt_tokens
        boundary = seq.pd_prefix_length
        count = (boundary + seq.block_size - 1) // seq.block_size
        blocks = seq.logical_blocks.get_real_blocks()[:count].copy()
        physical = self.block_manager.resolve_gpu_block_offsets(blocks).tolist()
        compact = isinstance(self.state_manager, SSMStateManager)
        auxiliary = seq.logical_state if compact and boundary else None
        legacy = seq.logical_state if not compact and seq.logical_state >= 0 and boundary else None
        handoff = PDHandoff(transfer_id=uuid4().hex, engine_epoch=self.epoch, session_id=seq.session_id,
                            cache_schema=cache_schema,
                            prompt_digest=prompt_digest(seq, tokens), prompt_token_count=len(tokens),
                            valid_token_count=boundary, block_size=seq.block_size, block_ids=physical,
                            state_ids=state_ids if compact and boundary else [], auxiliary_id=auxiliary,
                            legacy_state_id=legacy)
        access = None
        if auxiliary is not None:
            bank = self.state_manager.bank_for(seq)
            access = self.state_manager.submit_export(bank)
            valid_ids = set(bank.unit_ids)
            if bank.input_ref is not None:
                valid_ids.add(bank.input_ref.unit_id)
            if not set(state_ids).issubset(valid_ids):
                self.state_manager.complete(access)
                raise ValueError('Worker export state is not owned by the frozen request bank.')
        elif legacy is not None:
            self.state_manager.retain_state(legacy)
        self.block_manager.pin_logical_blocks(blocks)
        self.exports[handoff.transfer_id] = PDExport(seq, handoff, blocks, access)
        self._export_by_sequence[seq.seq_id] = handoff.transfer_id
        return handoff

    def control(self, request: PDControlRequest):
        """Return the released sequence, or None while its resources stay
        held."""
        if request.action == 'abort':
            if request.receiver_epoch != self.epoch:
                raise ValueError('PD receiver incarnation has changed.')
            for receive in self.receives.values():
                handoff = receive.request.handoff
                if handoff.transfer_id == request.transfer_id and handoff.engine_epoch == request.engine_epoch:
                    self.quarantine_receive(receive.sequence, request.reason or 'Prefill abandoned this transfer.')
                    break
            return None
        if request.engine_epoch != self.epoch:
            raise ValueError('PD engine incarnation has changed.')
        export = self.exports.get(request.transfer_id)
        if export is None:
            if request.action == 'claim':
                raise ValueError('PD export is no longer available.')
            return None
        if request.action == 'claim':
            if not request.receiver_epoch or export.revoked:
                raise ValueError(export.reason or 'PD export was revoked or has no receiver incarnation.')
            if export.receiver_epoch and export.receiver_epoch != request.receiver_epoch:
                raise ValueError('PD export is already claimed by another receiver.')
            export.receiver_epoch = request.receiver_epoch
            export.receiver_engine_id = request.receiver_engine_id
            export.renewed_at = monotonic()
            return None
        if request.action == 'revoke':
            if not export.revoked:
                export.revoked = True
                export.renewed_at = monotonic()
                export.reason = request.reason or 'Prefill request was cancelled.'
                if export.receiver_engine_id:
                    self.notifications.append((export.receiver_engine_id, PDControlRequest(
                        action='abort', transfer_id=request.transfer_id, engine_epoch=self.epoch,
                        receiver_epoch=export.receiver_epoch, reason=export.reason)))
            if export.receiver_epoch:
                return None
        elif not export.receiver_epoch or export.receiver_epoch != request.receiver_epoch:
            raise ValueError('Only the granted receiver can release a PD export.')
        return self._release_export(export)

    def holds(self, seq) -> bool:
        return seq.seq_id in self._export_by_sequence or seq.seq_id in self.receives

    def defer_end(self, seq, *, stop=False) -> bool:
        """Cancel a published request without making its memory evictable."""
        receive = self.receives.get(seq.seq_id)
        if receive is not None:
            if receive.cancelled != 'end':
                receive.cancelled = 'stop' if stop else 'end'
            self.quarantine_receive(seq, 'Decode request was cancelled.')
            return True
        transfer_id = self._export_by_sequence.get(seq.seq_id)
        if transfer_id is None:
            return False
        export = self.exports[transfer_id]
        self.control(PDControlRequest(action='revoke', transfer_id=transfer_id, engine_epoch=self.epoch))
        return bool(export.receiver_epoch)

    def has_pending(self) -> bool:
        return bool(self.exports or self.receives)

    def expire_exports(self, now: float) -> list:
        """Reclaim unclaimed exports and grants whose receiver stopped
        renewing."""
        expired = [export for export in self.exports.values()
                   if now - export.renewed_at >= self.release_timeout]
        self.forced_releases += sum(bool(export.receiver_epoch) for export in expired)
        return [self._release_export(export) for export in expired]

    def require_drained(self):
        if self.has_pending():
            raise RuntimeError('PD cache is still owned by exports or receives; refusing cache release.')

    def reset(self):
        self.require_drained()
        self.epoch = uuid4().hex
        self.notifications.clear()

    def snapshot(self):
        now = monotonic()
        failures = [r.failed_at for r in self.receives.values() if r.quarantined]
        return dict(exports=len(self.exports), grants=sum(bool(x.receiver_epoch) for x in self.exports.values()),
                    receives=len(self.receives), quarantined=len(failures),
                    oldest_quarantine_seconds=now - min(failures) if failures else 0.0,
                    forced_releases=self.forced_releases,
                    pinned_blocks=sum(len(x.logical_blocks) for x in (*self.exports.values(), *self.receives.values())))

    def _release_export(self, export):
        self.block_manager.release_logical_blocks(export.logical_blocks)
        if export.state_access is not None:
            self.state_manager.complete(export.state_access)
        elif export.handoff.legacy_state_id is not None:
            self.state_manager.free_state(export.handoff.legacy_state_id)
        del self.exports[export.handoff.transfer_id]
        del self._export_by_sequence[export.sequence.seq_id]
        return export.sequence
