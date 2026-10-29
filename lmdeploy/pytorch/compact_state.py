# Copyright (c) OpenMMLab. All rights reserved.
"""Compact state execution tables, worker cursors and prefill metadata."""
from dataclasses import dataclass, fields

import torch


@dataclass(frozen=True)
class SSMStateMetadata:
    """One immutable submitted bank/input/output table version.

    Tensors contain IDs only. Bank promotion creates a new table; submitted tables must not be mutated in place.
    Auxiliary request IDs remain in ModelInputs.state_offsets. Checkpoint reservation IDs and relative offsets
    deliberately do not belong here.
    """
    state_ids: torch.Tensor
    input_state_ids: torch.Tensor
    output_state_ids: torch.Tensor
    # Trailing -1 IDs describe no allocation. Decode must rebind a fully
    # admitted bank after prefill, even after device-side state advancement.
    partial_bank: bool = False

    def __post_init__(self):
        minimum = 1
        if self.state_ids.ndim != 2 or self.state_ids.shape[1] < minimum:
            raise ValueError(f'Expected compact bank table [B,V], V>={minimum}.')
        batch, width = self.state_ids.shape
        if (self.input_state_ids.shape != (batch, ) or self.output_state_ids.ndim != 2
                or self.output_state_ids.shape[0] != batch):
            raise ValueError('Compact input/output batch dimensions must match bank rows.')
        maximum = width
        if not 1 <= self.output_state_ids.shape[1] <= maximum:
            raise ValueError('Compact output width must satisfy 1<=W<=V.')
        for tensor in (self.state_ids, self.input_state_ids, self.output_state_ids):
            if tensor.dtype != torch.int64 or tensor.device != self.state_ids.device or not tensor.is_contiguous():
                raise ValueError('Compact ID tensors must be contiguous int64 on one device.')

    def index_select(self, indices):
        """Reindex only small metadata; never index state payloads."""
        if indices is Ellipsis:
            return self
        return type(self)(self.state_ids[indices], self.input_state_ids[indices], self.output_state_ids[indices],
                          partial_bank=self.partial_bank)

    def merge(self, other: 'SSMStateMetadata'):
        if (self.state_ids.shape[1] != other.state_ids.shape[1]
                or self.output_state_ids.shape[1] != other.output_state_ids.shape[1]):
            raise ValueError('Merging compact rows requires equal bank and update widths.')
        return type(self)(torch.cat((self.state_ids, other.state_ids)),
                          torch.cat((self.input_state_ids, other.input_state_ids)),
                          torch.cat((self.output_state_ids, other.output_state_ids)), partial_bank=self.partial_bank
                          or other.partial_bank)

    def to_device(self, device, non_blocking=False):
        return type(self)(*(tensor.to(device, non_blocking=non_blocking)
                            for tensor in (self.state_ids, self.input_state_ids, self.output_state_ids)),
                          partial_bank=self.partial_bank)

    def record_stream(self, stream):
        for tensor in (self.state_ids, self.input_state_ids, self.output_state_ids):
            if tensor.is_cuda:
                tensor.record_stream(stream)


class SSMStateCursor:
    """Worker-owned current-state cursor, independent of active-batch order.

    Indexed by auxiliary request slot, never by payload unit. The host owner must hold every bank/source through
    execution and order slot recycling. Fresh history0 ignores the previous occupant; explicit restore IDs override
    retained state. Binding is observational; consumed-state selection commits in the same kernel that plans the next
    private outputs. No payload storage.
    """

    def __init__(self, request_slots: int, device='cuda'):
        if request_slots < 2:
            raise ValueError('A compact cursor needs an active request slot; stateless models must not allocate it.')
        self.current_ids = torch.full((request_slots, ), -1, dtype=torch.int64, device=device)

    def bind(self, state: SSMStateMetadata, request_ids: torch.Tensor, history_lengths: torch.Tensor,
             source_ids: torch.Tensor | None = None, *,
             prefill: 'SSMPrefillMetadata | None' = None) -> SSMStateMetadata:
        from .kernels.cuda.compact_state import select_compact_state_ids
        inputs, outputs = select_compact_state_ids(state.state_ids, state.input_state_ids,
                                                   state.output_state_ids.shape[1],
                                                   runtime_ids=self.current_ids, request_ids=request_ids,
                                                   history_lengths=history_lengths, source_ids=source_ids,
                                                   prefill=prefill, allow_partial_bank=state.partial_bank)
        return SSMStateMetadata(state.state_ids, inputs, outputs, partial_bank=state.partial_bank)

    def advance(self, state: SSMStateMetadata, request_ids: torch.Tensor, consumed: torch.Tensor,
                width: int | None = None) -> SSMStateMetadata:
        from .kernels.cuda.compact_state import select_compact_state_ids
        width = state.output_state_ids.shape[1] if width is None else width
        inputs, outputs = select_compact_state_ids(state.state_ids, state.input_state_ids, width,
                                                   output_ids=state.output_state_ids,
                                                   consumed=consumed, runtime_ids=self.current_ids,
                                                   request_ids=request_ids, allow_partial_bank=state.partial_bank)
        return SSMStateMetadata(state.state_ids, inputs, outputs, partial_bank=state.partial_bank)

    def rebind(self, state: SSMStateMetadata, bank_ids: torch.Tensor, *,
               decode_ids: torch.Tensor | None = None) -> SSMStateMetadata:
        """Plan against a fully admitted bank without advancing the cursor.

        Keep the selected input and old submitted table intact; only write permissions change.
        """
        from .kernels.cuda.compact_state import select_compact_state_ids
        if bank_ids.shape != state.state_ids.shape:
            raise ValueError('Rebinding cannot change bank capacity or batch rows.')
        inputs, outputs = select_compact_state_ids(bank_ids, state.input_state_ids, state.output_state_ids.shape[1],
                                                   decode_ids=decode_ids)
        return SSMStateMetadata(bank_ids, inputs, outputs)


@dataclass(frozen=True)
class SSMPrefillMetadata:
    """One-forward packed layout/checkpoint export, never persistent decode
    state.

    Build from scheduler-known CPU lengths before dispatch. Backend operators must not reconstruct chunk indices by
    reading device lengths back. IDs are reservations owned and pinned by the caller, not allocations by this record.
    """
    cu_seqlens: torch.Tensor
    seq_idx: torch.Tensor
    chunk_indices: torch.Tensor
    chunk_offsets: torch.Tensor
    is_init_token: torch.Tensor
    checkpoint_ids: torch.Tensor | None = None
    checkpoint_offsets: torch.Tensor | None = None

    def __post_init__(self):
        if self.cu_seqlens.ndim != 1 or not self.cu_seqlens.numel():
            raise ValueError('Expected cumulative lengths with a leading zero.')
        batch = self.cu_seqlens.numel() - 1
        if (self.seq_idx.ndim != 1 or self.is_init_token.shape != self.seq_idx.shape
                or self.chunk_indices.ndim != 2 or self.chunk_indices.shape[1] != 2
                or self.chunk_offsets.shape != (batch + 1,)):
            raise ValueError('Invalid compact prefill layout geometry.')
        if (self.checkpoint_ids is None) != (self.checkpoint_offsets is None):
            raise ValueError('Checkpoint IDs and offsets must be supplied together.')
        for tensor, dtype in ((self.cu_seqlens, torch.int64), (self.seq_idx, torch.int32),
                              (self.chunk_indices, torch.int32), (self.chunk_offsets, torch.int32),
                              (self.is_init_token, torch.bool), (self.checkpoint_ids, torch.int64),
                              (self.checkpoint_offsets, torch.int64)):
            if tensor is not None and (tensor.device != self.cu_seqlens.device or tensor.dtype != dtype
                                       or not tensor.is_contiguous()):
                raise ValueError('Compact prefill metadata requires canonical contiguous types on one device.')
        for tensor in (self.checkpoint_ids, self.checkpoint_offsets):
            if tensor is not None and tensor.shape != (batch,):
                raise ValueError('Checkpoint metadata must match prefill rows.')

    @classmethod
    def from_lengths(cls, lengths: list[int], histories: list[int], *,
                     checkpoint_ids: list[int] | None = None, checkpoint_offsets: list[int] | None = None):
        """Create CPU metadata; unsupported internal boundaries fail pre-
        submit."""
        if len(lengths) != len(histories) or any(n < 0 for n in (*lengths, *histories)):
            raise ValueError('Expected matching nonnegative host sequence/history lengths.')
        if (checkpoint_ids is None) != (checkpoint_offsets is None):
            raise ValueError('Checkpoint IDs and offsets must be supplied together.')
        if checkpoint_ids is not None:
            if len(checkpoint_ids) != len(lengths) or len(checkpoint_offsets) != len(lengths):
                raise ValueError('Checkpoint metadata must match prefill rows.')
            for length, unit, offset in zip(lengths, checkpoint_ids, checkpoint_offsets):
                if unit > 0 and not (0 < offset <= length and (offset == length or offset % 64 == 0)):
                    raise ValueError('Compact checkpoint must be a chunk boundary or the final boundary.')
        cumulative, chunks, offsets = [0], [], [0]
        for row, length in enumerate(lengths):
            cumulative.append(cumulative[-1] + length)
            chunks.extend((row, i) for i in range((length + 63) // 64))
            offsets.append(len(chunks))
        seq_idx = torch.repeat_interleave(torch.arange(len(lengths), dtype=torch.int32),
                                           torch.tensor(lengths, dtype=torch.int64))
        init = torch.zeros(cumulative[-1], dtype=torch.bool)
        for start, length, history in zip(cumulative, lengths, histories):
            if length and history == 0:
                init[start] = True
        return cls(torch.tensor(cumulative, dtype=torch.int64), seq_idx,
                   torch.tensor(chunks, dtype=torch.int32).reshape(-1, 2),
                   torch.tensor(offsets, dtype=torch.int32), init,
                   None if checkpoint_ids is None else torch.tensor(checkpoint_ids, dtype=torch.int64),
                   None if checkpoint_offsets is None else torch.tensor(checkpoint_offsets, dtype=torch.int64))

    def to_device(self, device, non_blocking=False):
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = value.to(device, non_blocking=non_blocking) if value is not None else None
        return type(self)(**values)

    def record_stream(self, stream):
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None and value.is_cuda:
                value.record_stream(stream)
