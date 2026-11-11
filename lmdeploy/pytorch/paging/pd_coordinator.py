# Copyright (c) OpenMMLab. All rights reserved.
"""Paging ownership of PD exports and receives; callbacks never own memory."""
import hashlib
import json
from dataclasses import dataclass, field
from time import monotonic
from uuid import uuid4

import numpy as np

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
    revoked: bool = False
    created_at: float = field(default_factory=monotonic)


@dataclass
class PDReceive:
    sequence: object
    request: PDReceiveRequest
    logical_blocks: np.ndarray
    state_access: StateAccess | None
    cancelled: str = ''
    quarantined: bool = False


class PDCoordinator:
    """Serialize grants/cancellation on the scheduler loop and pin allocations.

    A granted export survives request cancellation until its receiver reports physical completion. Session IDs and
    timeout alone never release a grant.
    """

    def __init__(self, block_manager, state_manager):
        self.block_manager = block_manager
        self.state_manager = state_manager
        self.epoch = uuid4().hex
        self.exports: dict[str, PDExport] = {}
        self._export_by_sequence = {}
        self.receives: dict[int, PDReceive] = {}
        self.failed = False

    def validate_receive(self, seq, cache_schema):
        handoff = seq.migration_request.handoff
        if (handoff.cache_schema != cache_schema or handoff.block_size != seq.block_size
                or handoff.prompt_token_count != seq.num_valid_ids
                or handoff.prompt_digest != prompt_digest(seq, seq.valid_ids)):
            raise ValueError('PD handoff does not match this prompt or execution schema.')
        if self.failed:
            raise RuntimeError('PD receiver is quarantined and requires restart.')

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
                                    legacy_state_id=legacy_id)
        blocks = seq.logical_blocks.get_real_blocks().copy()
        self.block_manager.pin_logical_blocks(blocks)
        self.receives[seq.seq_id] = PDReceive(seq, request, blocks, access)

    def finish_receive(self, seq, result) -> str:
        """Commit only complete results; unknown native work keeps every
        hold."""
        receive = self.receives[seq.seq_id]
        if result.transfer_id != receive.request.handoff.transfer_id:
            raise ValueError('PD completion belongs to another receive operation.')
        if result.status == 'completion_unknown':
            receive.quarantined = self.failed = True
            return 'quarantined'
        success = result.status == 'completed' and not receive.cancelled
        if receive.state_access is not None:
            self.state_manager.complete(receive.state_access, success=success)
        elif receive.request.legacy_state_id is not None:
            self.state_manager.free_state(receive.request.legacy_state_id)
        self.block_manager.release_logical_blocks(receive.logical_blocks)
        del self.receives[seq.seq_id]
        if success:
            seq.set_step(receive.request.handoff.valid_token_count)
            seq.migration_request = None
            seq.kv_token_limit = None
            return 'ready'
        return receive.cancelled or 'error'

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
        if request.engine_epoch != self.epoch:
            raise ValueError('PD engine incarnation has changed.')
        export = self.exports.get(request.transfer_id)
        if export is None:
            if request.action == 'claim':
                raise ValueError('PD export is no longer available.')
            return None
        if request.action == 'claim':
            if not request.receiver_epoch or export.revoked:
                raise ValueError('PD export was revoked or has no receiver incarnation.')
            if export.receiver_epoch and export.receiver_epoch != request.receiver_epoch:
                raise ValueError('PD export is already claimed by another receiver.')
            export.receiver_epoch = request.receiver_epoch
            return None
        if request.action == 'revoke':
            export.revoked = True
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
            return True
        transfer_id = self._export_by_sequence.get(seq.seq_id)
        if transfer_id is None:
            return False
        export = self.exports[transfer_id]
        self.control(PDControlRequest(action='revoke', transfer_id=transfer_id, engine_epoch=self.epoch))
        return bool(export.receiver_epoch)

    def has_pending(self) -> bool:
        return bool(self.exports or self.receives)

    def expire_unclaimed(self, now: float, ttl: float = 300) -> list:
        """Reclaim only prepared exports for which no reader was granted."""
        return [self._release_export(export) for export in list(self.exports.values())
                if not export.receiver_epoch and now - export.created_at >= ttl]

    def require_drained(self):
        if self.has_pending():
            raise RuntimeError('PD cache is still owned by exports or receives; refusing cache release.')

    def reset(self):
        self.require_drained()
        self.epoch = uuid4().hex
        self.failed = False

    def snapshot(self):
        return dict(exports=len(self.exports), grants=sum(bool(x.receiver_epoch) for x in self.exports.values()),
                    receives=len(self.receives), quarantined=self.failed,
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
