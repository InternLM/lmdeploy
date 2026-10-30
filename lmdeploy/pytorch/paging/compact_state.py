# Copyright (c) OpenMMLab. All rights reserved.
"""Host ownership of one compact-unit pool; no device or per-token operations.

Private banks belong to requests. Checkpoints remove write permission before publication. References and submitted
accesses independently delay recycling. The caller reports *actual* forward/I/O completion, not dispatch completion.
"""
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from lmdeploy.pytorch.config import SSMStateCapacity

if TYPE_CHECKING:
    from lmdeploy.pytorch.messages import SchedulerSequence

Owner = Literal['reservation', 'trie', 'request', 'store']


@dataclass(eq=False)
class _Unit:
    unit_id: int
    bank: 'StateBank | None' = None
    holders: Counter = field(default_factory=Counter)
    accesses: set = field(default_factory=set)
    writers: set = field(default_factory=set)
    boundary: int | None = None
    phase: str | None = None  # failed may precede physical producer completion
    producer: 'StateAccess | None' = None
    producer_bank: 'StateBank | None' = None


@dataclass(eq=False)
class StateBank:
    """Request-private unit IDs and independently indexed auxiliary slot.

    ``unit_ids`` is a snapshot, replaced rather than modified during promotion.
    Decode uses the device token boundary to select among these IDs; this owner
    never reads per-token acceptance back to the host.
    """
    auxiliary_id: int
    unit_ids: tuple[int, ...]
    stream: int
    input_ref: 'StateReference | None' = None
    # Logical reservation only. Released producers retain their independent
    # access holds, so a later prefetched chunk may reserve another output.
    pending_checkpoint: 'StateReference | None' = None
    pending_load: 'StateAccess | None' = None
    # One-shot private input (unit ID, exact token boundary), not a shared ref.
    loaded_input: tuple[int, int] | None = None
    closing: bool = False
    failed: bool = False
    accesses: set = field(default_factory=set)
    # Prefill owns only its current/output units. Decode admission grows this
    # bank before a fixed-width device ring may be submitted.
    phase: Literal['prefill', 'decode'] = 'decode'


@dataclass(eq=False)
class StateReference:
    """One independently releasable holder of a particular physical unit.

    Capture this handle at dispatch/I/O submission. Do not look up a mutable trie node's current slot when completing an
    older operation.
    """
    _unit: _Unit
    owner: Owner
    active: bool = True
    _store_access: 'StateAccess | None' = None

    @property
    def unit_id(self) -> int:
        return self._unit.unit_id

    @property
    def boundary(self) -> int:
        return self._unit.boundary

    @property
    def phase(self) -> str:
        """Physical producer phase; logical release does not imply
        completion."""
        return self._unit.phase

    @property
    def producing(self) -> bool:
        return self.pending_producer is not None

    @property
    def pending_producer(self) -> 'StateAccess | None':
        """Producer ticket requiring ordering, not permission to
        consume/recycle.

        Store first retains a readable reference; its normal worker event wait orders the transfer. This ticket also
        enforces host completion order.
        """
        producer = self._unit.producer
        return producer if producer is not None and not producer.finished else None

    def shares_storage(self, other: 'StateReference') -> bool:
        """Compare physical allocation identity, even after logical release.

        A recycled numeric unit ID must not make an old reference equal to a newer allocation or redirect completion to
        a replaced trie node.
        """
        return self._unit is other._unit


@dataclass(eq=False)
class StateAccess:
    """Submitted access lifetime and required producer dependencies.

    Cross-stream dependencies are explicit: the executor must enqueue their
    ready-event waits before launching this access. A completion callback alone
    cannot create CUDA stream ordering. Failure invalidates values immediately;
    only finished=True proves that this access no longer uses physical storage.
    """
    _units: tuple[_Unit, ...]
    _writes: tuple[_Unit, ...]
    _banks: tuple[StateBank, ...]
    stream: int
    dependencies: tuple['StateAccess', ...]
    finished: bool = False
    failed: bool = False
    _store_reference: StateReference | None = None
    _load_boundary: int | None = None
    bank_ids: tuple[tuple[int, ...], ...] = ()
    read_only_input_ids: tuple[int, ...] = ()
    # Exact logical references captured at submission, independently of later
    # bank rebinding. Positive decode consumption may retire only these.
    input_references: tuple[StateReference | None, ...] = ()
    # Monotonic within this owner lifetime; transport never serializes handles.
    access_id: int = field(default=0, init=False)


class SSMStateManager:
    """Unified-mode StateManager implementation over a single StateAllocator.

    Mutually exclusive with the legacy slot manager, not an additional pool. No per-owner allocator, snapshot pool or
    checkpoint-format conversion.
    """

    def __init__(self, capacity: SSMStateCapacity):
        # Imported here so state_manager can expose the mutually exclusive
        # construction branch without a module-level cycle.
        from .state_manager import StateAllocator

        self.capacity = capacity
        self.allocator = StateAllocator(max(0, capacity.num_units - 1), offset=1)
        self._units: dict[int, _Unit] = {}
        self._banks: dict[int, StateBank] = {}
        self._store_refs = 0
        self._next_access_id = 1
        self._accesses: dict[int, StateAccess] = {}

    def is_allocated(self, seq: 'SchedulerSequence') -> bool:
        """The sequence's logical state identifies its auxiliary request
        slot."""
        return seq.logical_state >= 0

    def bank_for(self, seq: 'SchedulerSequence') -> StateBank:
        """Resolve live request ownership, never interpret auxiliary ID as unit
        ID."""
        bank = self._banks.get(seq.logical_state)
        if bank is None:
            raise ValueError('Sequence does not own a compact state bank.')
        self._check_bank(bank, allow_failed=True, allow_loading=True)
        return bank

    def allocate(self, seq: 'SchedulerSequence') -> None:
        """Admit prefill storage with guaranteed later decode expansion."""
        if self.is_allocated(seq):
            self._check_bank(self.bank_for(seq))
            return
        bank = self.allocate_bank(prefill=True)
        seq.logical_state = bank.auxiliary_id

    def free(self, seq: 'SchedulerSequence') -> None:
        """Detach a sequence immediately; submitted holds delay physical
        reuse."""
        if not self.is_allocated(seq):
            return
        self.release_bank(self.bank_for(seq))
        seq.logical_state = -1

    def get_num_free_runtime_slots(self) -> int:
        """Auxiliary admission slots, including slots held by closing banks."""
        return self.capacity.request_capacity - len(self._banks)

    def get_num_free_runtime(self) -> int:
        """Admit only requests whose eventual complete ring still fits."""
        if not self.capacity.bank_width:
            return 0
        return min(self.get_num_free_runtime_slots(),
                   self.get_num_free_optional() // self.capacity.bank_width)

    def get_num_free_optional(self) -> int:
        """Physical free units excluding admitted prefills' expansion credits.

        These credits are capacity, not allocations or payload holds. Optional checkpoints/Store leases may borrow
        unadmitted capacity, but cannot prevent an already admitted prefill from becoming runnable decode.
        """
        reserved = sum(self.capacity.bank_width - len(bank.unit_ids)
                       for bank in self._banks.values() if bank.phase == 'prefill' and not bank.closing)
        return max(0, self.get_num_free() - reserved)

    def allocate_bank(self, stream: int = 0, *, prefill: bool = False) -> StateBank:
        """Admit one prefill destination or a complete decode bank atomically.

        Serving starts with one prefill destination and reserves capacity for later promotion. Padding in its ID table
        is not allocated payload; decode submission requires the complete ring.
        """
        width = 1 if prefill else self.capacity.bank_width
        if self.get_num_free_runtime() == 0:
            raise RuntimeError('No capacity for a complete state bank.')
        slot = next(i for i in range(1, self.capacity.request_slots) if i not in self._banks)
        units = [self._allocate_unit() for _ in range(width)]
        bank = StateBank(slot, tuple(unit.unit_id for unit in units), stream,
                         phase='prefill' if prefill else 'decode')
        self._banks[slot] = bank
        for unit in units:
            unit.bank = bank
        return bank


    def promote_to_decode(self, bank: StateBank) -> bool:
        """Atomically acquire missing ring units, retaining every existing ID.

        False leaves the prefill result, auxiliary slot and submitted plans intact. The scheduler must resolve capacity
        before dispatching decode; this method neither evicts checkpoints nor discards a completed prefill. Existing
        accesses keep their original snapshots and completion holds.
        """
        self._check_bank(bank)
        if not self._grow_bank(bank, self.capacity.bank_width):
            return False
        bank.phase = 'decode'
        return True

    def _grow_bank(self, bank: StateBank, width: int) -> bool:
        missing = width - len(bank.unit_ids)
        if missing <= 0:
            return True
        if self.get_num_free() < missing:
            return False
        # This owner is synchronous: capacity cannot change between the check
        # and allocations. Publish one new tuple only after acquiring all IDs.
        units = [self._allocate_unit() for _ in range(missing)]
        for unit in units:
            unit.bank = bank
        bank.unit_ids = bank.unit_ids + tuple(unit.unit_id for unit in units)
        return True

    def release_bank(self, bank: StateBank) -> None:
        """Stop new plans; retain units and auxiliary slot until accesses
        drain."""
        if bank.closing:
            return
        self._check_bank(bank, allow_failed=True, allow_loading=True)
        bank.closing = True
        if bank.input_ref is not None:
            self.release_reference(bank.input_ref)
            bank.input_ref = None
        # Device acceptance may select any private candidate. Decode pins the
        # bank as a whole, instead of creating V host holds every token step.
        if not any(access.bank_ids for access in bank.accesses):
            for unit_id in bank.unit_ids:
                unit = self._units[unit_id]
                unit.bank = None
                self._recycle(unit)
        pending = bank.pending_checkpoint
        if pending is not None and pending.active:
            if pending._unit.phase == 'reserved':
                self.cancel_checkpoint(pending)
            else:
                self.release_reference(pending)
        self._retire_bank(bank)

    def reserve_checkpoint(self, bank: StateBank, boundary: int) -> StateReference:
        """Reserve an immutable output outside the request's writable bank."""
        self._check_bank(bank)
        if bank.pending_checkpoint is not None:
            raise ValueError('One pending checkpoint per request is supported.')
        if boundary < 0:
            raise ValueError('Checkpoint boundary must be non-negative.')
        if self.get_num_free_optional() == 0:
            raise RuntimeError('No free states outside reserved decode expansion capacity.')
        unit = self._allocate_unit()
        unit.boundary = boundary
        unit.phase = 'reserved'
        unit.producer_bank = bank
        reference = self._reference(unit, 'reservation')
        bank.pending_checkpoint = reference
        return reference

    def cancel_checkpoint(self, reference: StateReference) -> None:
        """Roll back an unsubmitted reservation; queued writes need
        completion."""
        if not reference.active:
            return
        unit = self._check_reference(reference)
        if reference.owner != 'reservation' or unit.phase != 'reserved':
            raise ValueError('Only an unsubmitted reservation can be rolled back.')
        unit.producer_bank.pending_checkpoint = None
        unit.phase = 'failed'
        unit.boundary = None
        unit.producer_bank = None
        self.release_reference(reference)

    def reserve_decode_snapshot(self, access_id: int, bank: StateBank, boundary: int) -> StateReference:
        """Add a conditional snapshot destination before dispatching a bound
        forward.

        Unlike direct prefill exports, adjacent prefetched forwards may attempt the same boundary. Each owns a private
        destination, held by its existing completion ticket even when acceptance or cancellation discards the snapshot.
        """
        access = self._accesses[access_id]
        self._check_bank(bank)
        if access.finished or access.failed or bank not in access._banks or not access.bank_ids:
            raise ValueError('Decode snapshots require this bank\'s live model forward.')
        if boundary <= 0 or self.get_num_free_optional() == 0:
            raise ValueError('Decode snapshots require a positive boundary and optional capacity.')
        unit = self._allocate_unit()
        unit.boundary = boundary
        unit.phase = 'submitted'
        unit.producer = access
        unit.producer_bank = bank
        unit.accesses.add(access)
        unit.writers.add(access)
        access._units += (unit,)
        access._writes += (unit,)
        return self._reference(unit, 'reservation')


    def retain(self, reference: StateReference, owner: Owner) -> StateReference:
        """Acquire immutable ownership; submitted producers still need
        ordering."""
        unit = self._check_reference(reference)
        if owner not in ('trie', 'request', 'store'):
            raise ValueError('Only trie, request and Store consumers can retain a checkpoint.')
        if unit.phase not in ('submitted', 'ready'):
            raise ValueError('Checkpoint is not ordered-readable.')
        if owner == 'store' and self._store_refs >= self.capacity.store_budget:
            raise RuntimeError('No Store state save credit.')
        return self._reference(unit, owner)

    def bind_input(self, bank: StateBank, reference: StateReference) -> None:
        """Bind a shared input outside the writable bank; no payload copy."""
        self._check_bank(bank)
        if bank.loaded_input is not None:
            raise ValueError('A loaded private input cannot also bind a shared checkpoint.')
        retained = self.retain(reference, 'request')
        previous, bank.input_ref = bank.input_ref, retained
        if previous is not None:
            self.release_reference(previous)

    def bind_produced_inputs(self, access: StateAccess, banks: tuple[StateBank, ...],
                             references: tuple[StateReference | None, ...]) -> None:
        """Atomically retain final/checkpoint outputs for future request reads.

        The producer's old input is already held by its submitted access. Each new final is a reserved one-shot output
        outside the full writable bank, so publication never exposes a destination in a future write table.
        """
        if access.finished or len(banks) != len(references) or len(set(banks)) != len(banks):
            raise ValueError('Produced input binding requires matching live producer rows.')
        retained = []
        try:
            for bank, reference in zip(banks, references):
                self._check_bank(bank)
                if reference is None:
                    continue
                unit = self._check_reference(reference)
                if (bank not in access._banks or unit.producer is not access or unit.producer_bank is not bank
                        or unit not in access._writes or unit.bank is not None):
                    raise ValueError("Final input must be this bank's submitted immutable producer output.")
                retained.append((bank, self.retain(reference, 'request')))
        except Exception:
            for _, reference in retained:
                self.release_reference(reference)
            raise
        for bank, reference in retained:
            previous, bank.input_ref = bank.input_ref, reference
            if previous is not None:
                self.release_reference(previous)

    def clear_input(self, bank: StateBank) -> None:
        """Clear only after existing output metadata proves private progress.

        a=0 retains this logical input. Prior submitted reads have independent access holds, so clearing the logical
        reference cannot recycle early.
        """
        self._check_bank(bank, allow_failed=True, allow_loading=True)
        if bank.input_ref is not None:
            self.release_reference(bank.input_ref)
            bank.input_ref = None

    def release_reference(self, reference: StateReference) -> None:
        """Idempotent physical-unit release, independent of mutable trie
        nodes."""
        if not reference.active:
            return
        unit = self._check_reference(reference)
        if unit.phase == 'reserved' and reference.owner == 'reservation':
            raise ValueError('Cancel an unsubmitted checkpoint instead of dropping its reservation.')
        reference.active = False
        unit.holders[reference.owner] -= 1
        if (reference.owner == 'reservation' and unit.producer_bank is not None
                and unit.producer_bank.pending_checkpoint is reference):
            unit.producer_bank.pending_checkpoint = None
        if reference.owner == 'store':
            if reference._store_access is None or reference._store_access.finished:
                self._store_refs -= 1
        self._recycle(unit)


    def submit_decode(self, banks: tuple[StateBank, ...], stream: int = 0,
                      wait_for: tuple[StateAccess, ...] = (), *,
                      checkpoints: tuple[StateReference | None, ...] = ()) -> StateAccess:
        for bank in banks:
            self._check_bank(bank)
            if bank.phase != 'decode' or len(bank.unit_ids) != self.capacity.bank_width:
                raise ValueError('Decode requires a fully admitted decode bank.')
        return self._submit_banks(banks, stream, wait_for, checkpoints=checkpoints)

    def submit_prefill(self, banks: tuple[StateBank, ...], stream: int = 0,
                       wait_for: tuple[StateAccess, ...] = (), *,
                       checkpoints: tuple[StateReference | None, ...] = ()) -> StateAccess:
        """Pin only allocated units; padding in the device table owns
        nothing."""
        for bank in banks:
            self._check_bank(bank)
            if not 1 <= len(bank.unit_ids) <= self.capacity.bank_width:
                raise ValueError('Prefill requires an admitted nonempty state bank.')
        return self._submit_banks(banks, stream, wait_for, checkpoints=checkpoints)

    def _submit_banks(self, banks: tuple[StateBank, ...], stream: int,
                      wait_for: tuple[StateAccess, ...], *,
                      checkpoints: tuple[StateReference | None, ...]) -> StateAccess:
        """Pin immutable bank snapshots, without host per-token ID selection.

        The device indexer follows the capacity's explicit reuse permission.
        Bank allocation/freeze guarantees distinct private IDs and exclusion
        of *all* shared units. ``bank_ids`` is the submitted table version;
        ``read_only_input_ids`` describes read holds, NOT a repeated bind command.
        One-time checkpoint binding and device acceptance belong to input metadata.
        Prefill uses the same whole-bank holds and may produce one reserved
        checkpoint per row, outside its private bank. All validation precedes
        mutation; the checkpoint and final outputs share one completion ticket.
        """
        if any(dependency.failed for dependency in wait_for):
            raise ValueError('A failed compact dependency cannot be submitted.')
        if len(set(banks)) != len(banks):
            raise ValueError('Duplicate request bank in one access.')
        if checkpoints and len(checkpoints) != len(banks):
            raise ValueError('Checkpoint producers must match the submitted bank rows.')
        reads, input_ids = set(), []
        for bank in banks:
            self._check_bank(bank)
            if bank.stream != stream:
                raise ValueError('A private bank has one ordered execution stream.')
            if bank.input_ref is None:
                input_ids.append(-1)
                continue
            unit = self._check_reference(bank.input_ref)
            if unit.phase not in ('submitted', 'ready'):
                raise ValueError('Checkpoint is not ordered-readable.')
            producer = unit.producer
            if (producer is not None and not producer.finished and producer.stream != stream
                    and producer not in wait_for):
                raise ValueError('Cross-stream checkpoint read requires its producer dependency.')
            reads.add(unit)
            input_ids.append(unit.unit_id)
        writes = set()
        for bank, reference in zip(banks, checkpoints):
            if reference is None:
                continue
            unit = self._check_reference(reference)
            if (reference is not bank.pending_checkpoint or unit.producer_bank is not bank
                    or reference.owner != 'reservation' or unit.phase != 'reserved'
                    or unit.bank is not None or unit.writers or unit in reads or unit in writes):
                raise ValueError('Checkpoint output must be this bank\'s exclusive unsubmitted reservation.')
            writes.add(unit)
        access = StateAccess(tuple(reads | writes), tuple(writes), tuple(banks), stream, tuple(wait_for),
                             bank_ids=tuple(bank.unit_ids for bank in banks),
                             read_only_input_ids=tuple(input_ids),
                             input_references=tuple(bank.input_ref for bank in banks) if reads else ())
        for unit in reads | writes:
            unit.accesses.add(access)
        for unit in writes:
            unit.writers.add(access)
            unit.phase = 'submitted'
            unit.producer = access
        for bank in banks:
            bank.accesses.add(access)
        return self._register_access(access)

    def submit_load(self, bank: StateBank, boundary: int) -> StateAccess:
        """Lease one private destination until actual all-rank Store
        completion.

        Admission already owns the load destination and auxiliary slot. No staging, freeze/replacement, Store-save
        credit or extra unit is allocated. A pending load excludes all model plans, even on the same stream: Store
        writes are asynchronous and cannot use model stream ordering as proof.
        """
        self._check_bank(bank)
        if type(boundary) is not int or boundary <= 0:
            raise ValueError('A compact load requires a positive exact token boundary.')
        if (bank.accesses or bank.input_ref is not None or bank.pending_checkpoint is not None
                or bank.loaded_input is not None):
            raise ValueError('A compact load requires an idle, unbound request bank.')
        unit = self._units[bank.unit_ids[0]]
        access = StateAccess((unit,), (unit,), (bank,), bank.stream, (), _load_boundary=boundary)
        unit.accesses.add(access)
        unit.writers.add(access)
        bank.accesses.add(access)
        bank.pending_load = access
        return self._register_access(access)

    def consume_loaded_inputs(self, access: StateAccess) -> None:
        """Commit one-shot loaded IDs after the CPU forward packet is bound.

        InputsMaker validates exact histories and inserts positive source IDs before calling this. Preparation failure
        must retain these overrides; after dispatch, ordinary forward holds protect the private input.
        """
        if self._accesses.get(access.access_id) is not access or not access.bank_ids:
            raise ValueError('Loaded input binding requires a live model bank access.')
        for bank in access._banks:
            bank.loaded_input = None

    def submit_save(self, reference: StateReference, stream: int,
                    wait_for: tuple[StateAccess, ...] = ()) -> StateAccess:
        """Protect direct Store reads even if cancellation drops its logical
        ref."""
        if any(dependency.failed for dependency in wait_for):
            raise ValueError('A failed compact dependency cannot be submitted.')
        unit = self._check_reference(reference)
        if (reference.owner != 'store' or unit.phase not in ('submitted', 'ready')
                or reference._store_access is not None):
            raise ValueError('Store reads require a readable Store reference.')
        producer = unit.producer
        if (producer is not None and not producer.finished and producer.stream != stream
                and producer not in wait_for):
            raise ValueError('Cross-stream Store read requires its producer dependency.')
        access = StateAccess((unit,), (), (), stream, tuple(wait_for), _store_reference=reference)
        reference._store_access = access
        unit.accesses.add(access)
        return self._register_access(access)

    def cancel_unsubmitted_save(self, access: StateAccess) -> None:
        """Roll back a CPU save transaction that has never reached a worker.

        This is NOT request cancellation or I/O failure. No read was dispatched, so its producer need not finish; the
        producer's own write holds remain. Submitted saves must instead wait for actual all-worker I/O completion.
        """
        if access.finished:
            return
        if self._accesses.get(access.access_id) is not access or access._store_reference is None:
            raise ValueError("Only this owner's unsubmitted Store read can be cancelled.")
        access.dependencies = ()
        self.complete(access, success=False)
        self.release_reference(access._store_reference)

    def get_num_free_store_saves(self) -> int:
        """Independent Store credits, including logically released in-flight
        reads."""
        return self.capacity.store_budget - self._store_refs

    def invalidate(self, access_id: int) -> tuple[int, ...]:
        """Make failed work and its queued descendants unreadable, without
        freeing.

        Failure notification is not GPU/I/O completion. A later request-bank update depends on the earlier runtime state
        even without an explicit cross-stream edge. Shared-checkpoint readers and their produced outputs transitively
        fail as well; unrelated/earlier submissions stay intact.
        """
        if type(access_id) is not int or not 0 < access_id < self._next_access_id:
            raise ValueError('Unknown compact failure identity.')
        if access_id not in self._accesses:
            return ()
        affected, banks, units = [], set(), set()
        for access in self._accesses.values():
            if access.access_id < access_id:
                continue
            if (access.access_id != access_id and not banks.intersection(access._banks)
                    and not units.intersection(access._units)
                    and not any(dependency.failed for dependency in access.dependencies)):
                continue
            access.failed = True
            affected.append(access.access_id)
            banks.update(access._banks)
            for bank in access._banks:
                bank.failed = True
            for unit in access._writes:
                if unit.producer is access:
                    unit.phase = 'failed'
                    units.add(unit)
        return tuple(affected)

    def needs_input_consumption(self, access_id: int) -> bool:
        """Skip token-mask work when this submission had no shared input.

        Consult the captured access, not the bank's current binding: a later prefetch may already have installed a
        different logical reference.
        """
        access = self._accesses.get(access_id)
        return access is not None and bool(access.input_references)

    def acknowledge(self, access_id: int, *, consumed_inputs: tuple[bool, ...] | None = None) -> None:
        """Release a successful all-rank completion's exact submitted access.

        IDs are never reused within an owner lifetime. Duplicate old reports are harmless even after the physical bank
        IDs have been recycled. Failure/drain paths must prove quiescence before calling complete with success=False; a
        failed RPC is not a completion acknowledgment. Optional consumption comes from the existing AR/ARSpec returned-
        token mask after target-state advance, never from a new unit-ID readback. Missing/zero consumption retains the
        logical input (including a=0).
        """
        if type(access_id) is not int or not 0 < access_id < self._next_access_id:
            raise ValueError('Unknown compact access completion identity.')
        access = self._accesses.get(access_id)
        if access is not None:
            if access._store_reference is not None or access._load_boundary is not None:
                raise ValueError('A forward acknowledgment cannot complete Store I/O.')
            if (access.failed or any(dependency.failed for dependency in access.dependencies)
                    or any(ref is not None and ref.phase == 'failed' for ref in access.input_references)):
                raise RuntimeError('Failed compact work cannot be acknowledged as successful output.')
            if consumed_inputs is not None:
                if (not isinstance(consumed_inputs, tuple) or len(consumed_inputs) != len(access._banks)
                        or (access.input_references and len(access.input_references) != len(access._banks))
                        or any(type(consumed) is not bool for consumed in consumed_inputs)):
                    raise ValueError('Consumption must match the submitted bank rows with host boolean evidence.')
            self.complete(access)
            if consumed_inputs is not None:
                for bank, reference, consumed in zip(access._banks, access.input_references, consumed_inputs):
                    if consumed and reference is not None and not bank.closing and bank.input_ref is reference:
                        self.clear_input(bank)

    def complete(self, access: StateAccess, success: bool = True) -> None:
        """Call at actual GPU/I/O completion; duplicate completion is harmless.

        On producer failure the coordinator must also withdraw trie entries and fail/recompute consumers. Existing
        consumer holds still protect storage; this owner never assumes failure means outstanding GPU work has drained.
        """
        if access.finished:
            return
        if (self._accesses.get(access.access_id) is not access
                or any(self._units.get(unit.unit_id) is not unit or access not in unit.accesses
                       for unit in access._units)
                or any(self._banks.get(bank.auxiliary_id) is not bank or access not in bank.accesses
                       for bank in access._banks)):
            raise ValueError('Access does not belong to this state owner.')
        for dependency in access.dependencies:
            if not dependency.finished:
                raise ValueError('Dependency must complete before its waiting access.')
        access.failed = (access.failed or not success or any(dependency.failed for dependency in access.dependencies)
                         or any(ref is not None and ref.phase == 'failed' for ref in access.input_references))
        access.finished = True
        if access._load_boundary is not None:
            bank = access._banks[0]
            bank.pending_load = None
            if access.failed:
                # A partial object is not a model input. Paging must discard
                # the bank and rematch/recompute after this terminal result.
                bank.failed = True
            elif not bank.closing and not bank.failed:
                bank.loaded_input = (access._writes[0].unit_id, access._load_boundary)
        if access._store_reference is not None and not access._store_reference.active:
            self._store_refs -= 1
        for unit in access._units:
            if unit.producer is access:
                bank = unit.producer_bank
                if bank.pending_checkpoint is not None and bank.pending_checkpoint._unit is unit:
                    bank.pending_checkpoint = None
                unit.phase = 'failed' if access.failed else 'ready'
                unit.producer = None
                unit.producer_bank = None
            unit.accesses.remove(access)
            unit.writers.discard(access)
            self._recycle(unit)
        for bank in access._banks:
            bank.accesses.remove(access)
            self._retire_bank(bank)
        del self._accesses[access.access_id]

    def finish_cancelled_forwards_after_worker_drain(self, *, preserve_store: bool = False) -> None:
        """Retire discarded completions after every worker has stopped using
        caches.

        Caller must first stop scheduling, end all sessions, and successfully await the existing all-worker sleep drain.
        RPC failure/cancellation is NOT this proof. No device synchronization belongs in this host owner. Store leases
        require their own I/O completion and cannot be swept here. With preserve_store=True, retire only model producers
        and leave EVERY Store read/write hold intact for the connector's separate I/O drain.
        """
        if any(not bank.closing for bank in self._banks.values()):
            raise RuntimeError('Compact worker drain requires all request banks to be closed.')
        accesses = tuple(self._accesses.values())
        if preserve_store:
            accesses = tuple(access for access in accesses
                             if access._store_reference is None and access._load_boundary is None)
        elif self._store_refs or any(access._store_reference is not None or access._load_boundary is not None
                                    for access in accesses):
            raise RuntimeError('Compact model drain cannot retire Store I/O ownership.')
        # Validate the whole ordered set before changing any completion state.
        # Dependencies already finished are harmless; unresolved foreign/future
        # work cannot be certified by this owner's worker-drain boundary.
        preceding = set()
        for access in accesses:
            if any(not dependency.finished and dependency not in preceding for dependency in access.dependencies):
                raise RuntimeError('Compact worker drain has an unproven external dependency.')
            preceding.add(access)
        for access in accesses:
            # Outputs may have been discarded or never executed. They are not
            # reusable checkpoints even though their storage is now quiescent.
            self.complete(access, success=False)

    def get_num_free(self) -> int:
        return self.allocator.get_num_free()

    def _register_access(self, access: StateAccess) -> StateAccess:
        access.access_id = self._next_access_id
        self._next_access_id += 1
        self._accesses[access.access_id] = access
        return access

    def _allocate_unit(self) -> _Unit:
        unit = _Unit(int(self.allocator.allocate()))
        self._units[unit.unit_id] = unit
        return unit

    def _reference(self, unit: _Unit, owner: Owner) -> StateReference:
        unit.holders[owner] += 1
        if owner == 'store':
            self._store_refs += 1
        return StateReference(unit, owner)

    def _check_bank(self, bank: StateBank, *, allow_failed: bool = False, allow_loading: bool = False) -> None:
        if self._banks.get(bank.auxiliary_id) is not bank or bank.closing:
            raise ValueError('Request bank is not live.')
        if bank.pending_load is not None and not allow_loading:
            raise ValueError('Compact request bank is awaiting Store I/O completion.')
        if bank.failed and not allow_failed:
            raise ValueError('Failed compact request bank requires cancellation/recompute before reuse.')

    def _check_reference(self, reference: StateReference) -> _Unit:
        unit = reference._unit
        if not reference.active or self._units.get(unit.unit_id) is not unit:
            raise ValueError('State reference is not live.')
        return unit

    def _recycle(self, unit: _Unit) -> None:
        if unit.bank is None and not unit.accesses and not any(unit.holders.values()):
            del self._units[unit.unit_id]
            self.allocator.free(unit.unit_id)

    def _retire_bank(self, bank: StateBank) -> None:
        if bank.closing and not bank.accesses:
            for unit_id in bank.unit_ids:
                unit = self._units.get(unit_id)
                if unit is not None and unit.bank is bank:
                    unit.bank = None
                    self._recycle(unit)
            del self._banks[bank.auxiliary_id]
