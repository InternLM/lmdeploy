# Copyright (c) OpenMMLab. All rights reserved.
"""Paging ownership of PD exports and receives; callbacks never own memory."""
import hashlib
import json
from dataclasses import dataclass
from uuid import uuid4

import numpy as np

from lmdeploy.pytorch.disagg.conn.protocol import PDControlRequest, PDHandoff

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

    def publish(self, seq, cache_schema, state_ids) -> PDHandoff:
        existing = self._export_by_sequence.get(seq.seq_id)
        if existing is not None:
            return self.exports[existing].handoff
        tokens = seq.pd_prompt_tokens
        boundary = len(tokens) - 1
        count = (boundary + seq.block_size - 1) // seq.block_size
        blocks = seq.logical_blocks.get_real_blocks()[:count].copy()
        physical = self.block_manager.resolve_gpu_block_offsets(blocks).tolist()
        compact = isinstance(self.state_manager, SSMStateManager)
        auxiliary = seq.logical_state if compact and boundary else None
        legacy = seq.logical_state if not compact and seq.logical_state >= 0 and boundary else None
        handoff = PDHandoff(transfer_id=uuid4().hex, engine_epoch=self.epoch, cache_schema=cache_schema,
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
        return seq.seq_id in self._export_by_sequence

    def defer_end(self, seq) -> bool:
        """Cancel a published request without making its memory evictable."""
        transfer_id = self._export_by_sequence.get(seq.seq_id)
        if transfer_id is None:
            return False
        export = self.exports[transfer_id]
        self.control(PDControlRequest(action='revoke', transfer_id=transfer_id, engine_epoch=self.epoch))
        return bool(export.receiver_epoch)

    def has_pending(self) -> bool:
        return bool(self.exports)

    def _release_export(self, export):
        self.block_manager.release_logical_blocks(export.logical_blocks)
        if export.state_access is not None:
            self.state_manager.complete(export.state_access)
        elif export.handoff.legacy_state_id is not None:
            self.state_manager.free_state(export.handoff.legacy_state_id)
        del self.exports[export.handoff.transfer_id]
        del self._export_by_sequence[export.sequence.seq_id]
        return export.sequence
