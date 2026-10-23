# Copyright (c) OpenMMLab. All rights reserved.
"""White-box host-ID fixtures, NOT the serving ownership API or numerical
oracle.

Production uses prepare_prefill_bank and submit_prefill/submit_decode, with current IDs selected on device. These
historical exact-ID fixtures exercise host permission counterexamples; tests of them do not prove serving wiring. Keep
independent logical/numerical oracles separate from both implementations.
"""
from dataclasses import dataclass

from lmdeploy.pytorch.paging.compact_state import StateAccess, StateBank, StateReference


@dataclass(frozen=True)
class StateAccessRow:
    """Host-known prefill/restore access, validated atomically for a batch.

    Outputs include an optional reserved checkpoint destination. The resulting StateAccess holds exactly these IDs, even
    if a later bank/trie binding moves.
    """
    bank: StateBank
    input_id: int
    output_ids: tuple[int, ...]
    checkpoint: StateReference | None = None

    def __post_init__(self):
        object.__setattr__(self, 'output_ids', tuple(self.output_ids))


def prepare_prefill_output(owner, bank: StateBank, input_id: int) -> int:
    """Select a private output, reusing an exclusive input when authorized.

    Zero/shared inputs need one private destination. A private input (also a completed Store GET destination) needs a
    second unit only in the disjoint-input mode. This allocates no checkpoint payload.
    """
    owner._check_bank(bank)
    if bank.phase != 'prefill':
        raise ValueError('Prefill output preparation requires a prefill bank.')
    if bank.loaded_input is not None and input_id != bank.loaded_input[0]:
        raise ValueError('First model access must read the loaded private input.')
    if input_id != 0:
        unit = owner._units.get(input_id)
        if unit is None or (unit.bank is not bank and (
                bank.input_ref is None or not bank.input_ref.active
                or bank.input_ref._unit is not unit or unit.phase not in ('submitted', 'ready'))):
            raise ValueError('Input must be private or a bound immutable reference.')
    if owner.capacity.allow_input_alias and input_id in bank.unit_ids:
        return input_id
    for unit_id in bank.unit_ids:
        if unit_id != input_id:
            return unit_id
    if not owner._grow_bank(bank, 2):
        raise RuntimeError('No capacity for a distinct prefill output.')
    return bank.unit_ids[-1]


def freeze_ready(owner, bank: StateBank, unit_id: int, boundary: int) -> StateReference:
    """Freeze an already completed boundary, without a producer or copy.

    Caller proves the exact boundary is still materialized. Pending private writes are rejected by reserve_checkpoint,
    including later queued writes.
    """
    reference = owner.reserve_checkpoint(bank, boundary, unit_id)
    bank.pending_checkpoint = None
    reference._unit.phase = 'ready'
    reference._unit.producer_bank = None
    reference._replacement = None
    return reference


def submit(owner, rows: tuple[StateAccessRow, ...], stream: int = 0,
           wait_for: tuple[StateAccess, ...] = ()) -> StateAccess:
    """Validate all rows before taking holds; then commit one access.

    This is for host-known prefill/Store plans, not a D2H check of device decode IDs. Decode bank metadata validation is
    a separate integration contract; it must follow the capacity reuse permission and exclude immutable IDs by
    construction.
    """
    if any(dependency.failed for dependency in wait_for):
        raise ValueError('A failed compact dependency cannot be submitted.')
    inputs, writes, banks, producers = set(), [], set(), []
    permitted_aliases = set()
    units = set()
    for row in rows:
        bank = row.bank
        owner._check_bank(bank)
        if bank.stream != stream:
            raise ValueError('A private bank has one ordered execution stream.')
        if bank in banks:
            raise ValueError('Duplicate request bank in one access.')
        banks.add(bank)
        if bank.loaded_input is not None and row.input_id != bank.loaded_input[0]:
            raise ValueError('First model access must read the loaded private input.')
        inputs.add(row.input_id)
        if row.input_id != 0:
            unit = owner._units.get(row.input_id)
            if unit is None:
                raise ValueError('Input unit is not live.')
            if unit.bank is not bank:
                if (bank.input_ref is None or not bank.input_ref.active
                        or bank.input_ref._unit is not unit or unit.phase not in ('submitted', 'ready')):
                    raise ValueError('Input must be private or a bound immutable reference.')
                producer = unit.producer
                if (producer is not None and not producer.finished and producer.stream != stream
                        and producer not in wait_for):
                    raise ValueError('Cross-stream checkpoint read requires its producer dependency.')
            units.add(unit)
        if (owner.capacity.allow_input_alias and row.output_ids
                and row.output_ids[-1] == row.input_id and row.input_id in bank.unit_ids):
            permitted_aliases.add(row.input_id)
        checkpoint = None
        if row.checkpoint is not None:
            checkpoint = owner._check_reference(row.checkpoint)
            if (row.checkpoint.owner != 'reservation' or checkpoint.phase != 'reserved'
                    or checkpoint.producer_bank is not bank):
                raise ValueError('Checkpoint has already been submitted or cancelled.')
            if checkpoint.unit_id not in row.output_ids:
                raise ValueError('Checkpoint reservation needs its producer output.')
            producers.append(checkpoint)
        for unit_id in row.output_ids:
            unit = owner._units.get(unit_id)
            if unit is None or (unit.bank is not bank and unit is not checkpoint):
                raise ValueError('Output must be a private unit or this producer reservation.')
            writes.append(unit)
            units.add(unit)
    if len(set(writes)) != len(writes):
        raise ValueError('Duplicate destination unit in one access.')
    if inputs.intersection(unit.unit_id for unit in writes) - permitted_aliases:
        raise ValueError('Output aliases a batch input unit.')
    access = StateAccess(tuple(units), tuple(writes), tuple(banks), stream, tuple(wait_for))
    for unit in units:
        unit.accesses.add(access)
    for unit in writes:
        unit.writers.add(access)
    for bank in banks:
        bank.accesses.add(access)
    for row in rows:
        # This API already names the physical input explicitly; unlike
        # submit_decode, it needs no separate CPU packet binding commit.
        row.bank.loaded_input = None
        if row.checkpoint is not None:
            row.checkpoint._replacement = None
    for unit in producers:
        unit.phase = 'submitted'
        unit.producer = access
    return owner._register_access(access)
