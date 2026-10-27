# Copyright (c) OpenMMLab. All rights reserved.
"""Unit ownership transitions; no GPU completion is inferred from dispatch."""
import random
from types import SimpleNamespace

import pytest

from lmdeploy.pytorch.config import SSMStateCapacity
from lmdeploy.pytorch.paging.compact_state import SSMStateManager
from lmdeploy.pytorch.paging.state_manager import build_state_manager
from tests.pytorch.compact_state_reference import StateAccessRow, submit


def manager(requests=2, width=2, checkpoints=2, store=1):
    return SSMStateManager(SSMStateCapacity(requests + 1, width, checkpoints, store))


def assert_drained(owner):
    assert owner.get_num_free() == owner.capacity.num_units - 1
    assert not owner._units
    assert not owner._banks
    assert not owner._accesses
    assert owner._store_refs == 0


def make_ready(owner, bank, boundary=64):
    checkpoint = owner.reserve_checkpoint(bank, boundary)
    access = submit(owner, (StateAccessRow(bank, 0, (checkpoint.unit_id,), checkpoint),), stream=bank.stream)
    owner.complete(access)
    return checkpoint


def test_only_one_allocator_and_atomic_full_bank_admission():
    owner = build_state_manager(SimpleNamespace(compact_state_capacity=SSMStateCapacity(3, 4, 0, 0)))
    assert isinstance(owner, SSMStateManager)
    a, b = owner.allocate_bank(), owner.allocate_bank()
    assert len(a.unit_ids) == len(b.unit_ids) == 4
    assert not set(a.unit_ids).intersection(b.unit_ids)
    assert a.auxiliary_id != b.auxiliary_id
    assert owner.get_num_free() == 0
    with pytest.raises(RuntimeError, match='complete state bank'):
        owner.allocate_bank()
    with pytest.raises(RuntimeError, match='No free states'):
        owner.reserve_checkpoint(a, 64)
    assert len(a.unit_ids) == 4
    owner.release_bank(a)
    owner.release_bank(b)
    assert_drained(owner)


def test_stateless_has_no_units_or_banks():
    owner = SSMStateManager(SSMStateCapacity(0, 0))
    assert owner.get_num_free() == 0
    with pytest.raises(RuntimeError):
        owner.allocate_bank()


def test_reservation_capacity_failure_is_atomic():
    owner = manager(requests=1, checkpoints=0, store=0)
    bank = owner.allocate_bank()
    original = bank.unit_ids
    with pytest.raises(RuntimeError):
        owner.reserve_checkpoint(bank, 64)
    assert bank.unit_ids is original
    owner.release_bank(bank)
    assert_drained(owner)


def test_reservation_rolls_back_before_submission():
    owner = manager()
    bank = owner.allocate_bank()
    initial = bank.unit_ids
    free = owner.get_num_free()
    checkpoint = owner.reserve_checkpoint(bank, 64)
    assert owner.get_num_free() == free - 1
    assert checkpoint.unit_id not in bank.unit_ids
    with pytest.raises(ValueError, match='ordered-readable'):
        owner.retain(checkpoint, 'trie')
    with pytest.raises(ValueError, match='Cancel'):
        owner.release_reference(checkpoint)
    owner.cancel_checkpoint(checkpoint)
    assert bank.unit_ids == initial
    assert owner.get_num_free() == free
    owner.release_bank(bank)
    assert_drained(owner)


def test_internal_checkpoint_single_producer_and_future_writers_excluded():
    owner = manager()
    bank = owner.allocate_bank()
    original = bank.unit_ids
    checkpoint = owner.reserve_checkpoint(bank, 64)
    replacement = bank.unit_ids[1]
    assert replacement != checkpoint.unit_id
    assert bank.unit_ids == original
    # A checkpoint cannot be written without its producer reservation.
    with pytest.raises(ValueError, match='private unit'):
        submit(owner, (StateAccessRow(bank, 0, (checkpoint.unit_id,)),))
    producer = submit(owner, (StateAccessRow(bank, 0, (checkpoint.unit_id,), checkpoint),))
    trie = owner.retain(checkpoint, 'trie')
    owner.bind_input(bank, checkpoint)
    successor = submit(owner, (StateAccessRow(bank, checkpoint.unit_id, (replacement,)),))
    with pytest.raises(ValueError, match='already been submitted'):
        submit(owner, (StateAccessRow(bank, 0, (checkpoint.unit_id,), checkpoint),))
    with pytest.raises(ValueError, match='unsubmitted'):
        owner.cancel_checkpoint(checkpoint)
    owner.release_reference(checkpoint)
    owner.release_reference(trie)
    owner.release_bank(bank)  # cancellation, not completion
    assert checkpoint.unit_id in owner._units
    assert replacement in owner._units
    owner.complete(producer)
    assert checkpoint.unit_id in owner._units  # still read by successor
    owner.complete(successor)
    owner.complete(successor)  # duplicated callback cannot free a new owner
    assert_drained(owner)


def test_checkpoint_reservation_never_changes_submitted_bank_ids():
    owner = manager()
    bank = owner.allocate_bank()
    access = owner.submit_decode((bank,))
    snapshot = bank.unit_ids
    checkpoint = owner.reserve_checkpoint(bank, 64)
    assert bank.unit_ids is snapshot and access.bank_ids == (snapshot,)
    assert checkpoint.unit_id not in snapshot
    owner.cancel_checkpoint(checkpoint)
    owner.complete(access)
    owner.release_bank(bank)
    assert_drained(owner)


def test_auxiliary_slot_and_units_live_until_all_forward_completion():
    owner = manager(requests=1)
    bank = owner.allocate_bank()
    first = submit(owner, (StateAccessRow(bank, 0, bank.unit_ids[:1]),))
    second = submit(owner, (StateAccessRow(bank, bank.unit_ids[0], bank.unit_ids[1:]),))
    owner.release_bank(bank)
    owner.complete(second)  # completion notification order need not be FIFO
    with pytest.raises(RuntimeError, match='complete state bank'):
        owner.allocate_bank()
    owner.complete(first)
    next_bank = owner.allocate_bank()
    assert next_bank.auxiliary_id == bank.auxiliary_id
    owner.complete(first)  # late callback after actual slot reuse
    assert owner._banks[next_bank.auxiliary_id] is next_bank
    owner.release_bank(next_bank)
    assert_drained(owner)


def test_two_consumers_a0_store_and_eviction_keep_physical_identity():
    owner = manager(requests=3)
    producer, a, b = [owner.allocate_bank() for _ in range(3)]
    checkpoint = make_ready(owner, producer)
    trie = owner.retain(checkpoint, 'trie')
    store = owner.retain(checkpoint, 'store')
    with pytest.raises(RuntimeError, match='save credit'):
        owner.retain(checkpoint, 'store')
    owner.bind_input(a, checkpoint)
    owner.bind_input(b, checkpoint)
    gpu_a = submit(owner, (StateAccessRow(a, checkpoint.unit_id, a.unit_ids[:1]),))
    gpu_b = submit(owner, (StateAccessRow(b, checkpoint.unit_id, b.unit_ids[:1]),))
    io = owner.submit_save(store, stream=10)
    owner.release_reference(checkpoint)
    owner.release_reference(trie)  # node pin=0 and node evicted
    owner.release_reference(store)  # cancelling save does NOT end I/O
    owner.clear_input(a)  # positive acceptance; old GPU read still outstanding
    owner.complete(gpu_b)  # a=0 for b, retain its logical input
    assert b.input_ref.active
    assert checkpoint.unit_id in owner._units
    owner.release_bank(a)
    owner.release_bank(b)
    owner.release_bank(producer)
    owner.complete(io)
    assert checkpoint.unit_id in owner._units  # a's delayed consumer
    owner.complete(gpu_a)
    assert_drained(owner)


def test_same_anchor_replacement_and_late_completions_do_not_touch_new_unit():
    owner = manager()
    a, b = owner.allocate_bank(), owner.allocate_bank()
    old = make_ready(owner, a)
    node_ref = owner.retain(old, 'trie')
    owner.bind_input(b, old)
    delayed = submit(owner, (StateAccessRow(b, old.unit_id, b.unit_ids[:1]),))
    new = make_ready(owner, a)
    old_node_ref = node_ref
    node_ref = owner.retain(new, 'trie')  # mutable TrieNode has been rebound
    owner.release_reference(old_node_ref)
    owner.release_reference(old)
    owner.clear_input(b)
    owner.complete(delayed)
    assert old.unit_id not in owner._units
    owner.complete(delayed)
    assert node_ref.active and new.unit_id in owner._units
    for ref in (new, node_ref):
        owner.release_reference(ref)
    owner.release_bank(a)
    owner.release_bank(b)
    assert_drained(owner)


def test_batch_negative_cases_are_atomic_before_any_write_hold():
    owner = manager()
    a, b = owner.allocate_bank(), owner.allocate_bank()
    cases = [
        (StateAccessRow(a, a.unit_ids[0], (a.unit_ids[0], a.unit_ids[1])),),
        (StateAccessRow(a, 0, (a.unit_ids[1], a.unit_ids[1])),),
        (StateAccessRow(a, 0, (b.unit_ids[0],)),),
        (StateAccessRow(a, b.unit_ids[0], a.unit_ids[:1]),),
        (StateAccessRow(a, 0, (0,)),),
        (StateAccessRow(a, 0, (-1,)),),
        (StateAccessRow(a, 0, (owner.capacity.num_units,)),),
        (StateAccessRow(a, 0, a.unit_ids[:1]), StateAccessRow(a, 0, a.unit_ids[1:])),
    ]
    free = owner.get_num_free()
    for rows in cases:
        with pytest.raises(ValueError):
            submit(owner, rows)
        assert owner.get_num_free() == free
        assert not a.accesses and not b.accesses
        assert all(not u.accesses and not u.writers for u in owner._units.values())
    owner.release_bank(a)
    owner.release_bank(b)
    assert_drained(owner)


def test_checkpoint_producer_cannot_switch_requests():
    owner = manager()
    a, b = owner.allocate_bank(), owner.allocate_bank()
    checkpoint = owner.reserve_checkpoint(a, 64)
    with pytest.raises(ValueError):
        submit(owner, (StateAccessRow(b, 0, (checkpoint.unit_id,), checkpoint),))
    owner.cancel_checkpoint(checkpoint)
    owner.release_bank(a)
    owner.release_bank(b)
    assert_drained(owner)


def test_cross_stream_consumers_require_explicit_producer_dependency():
    owner = manager()
    a, b = owner.allocate_bank(stream=1), owner.allocate_bank(stream=2)
    checkpoint = owner.reserve_checkpoint(a, 64)
    producer = submit(owner, (StateAccessRow(a, 0, (checkpoint.unit_id,), checkpoint),), stream=1)
    owner.bind_input(b, checkpoint)
    rows = (StateAccessRow(b, checkpoint.unit_id, b.unit_ids[:1]),)
    with pytest.raises(ValueError, match='producer dependency'):
        submit(owner, rows, stream=2)
    consumer = submit(owner, rows, stream=2, wait_for=(producer,))
    owner.release_bank(a)
    owner.release_bank(b)
    owner.release_reference(checkpoint)
    with pytest.raises(ValueError, match='Dependency must complete'):
        owner.complete(consumer)
    owner.complete(producer)
    owner.complete(consumer)
    assert_drained(owner)


def test_failure_rejects_new_consumers_but_does_not_recycle_pending_ones():
    owner = manager()
    a, b = owner.allocate_bank(), owner.allocate_bank()
    checkpoint = owner.reserve_checkpoint(a, 64)
    producer = submit(owner, (StateAccessRow(a, 0, (checkpoint.unit_id,), checkpoint),))
    owner.bind_input(b, checkpoint)
    consumer = submit(owner, (StateAccessRow(b, checkpoint.unit_id, b.unit_ids[:1]),))
    owner.complete(producer, success=False)
    with pytest.raises(ValueError, match='ordered-readable'):
        owner.retain(checkpoint, 'trie')
    owner.release_reference(checkpoint)
    owner.release_bank(a)
    owner.release_bank(b)
    assert checkpoint.unit_id in owner._units
    owner.complete(consumer, success=False)
    assert_drained(owner)


def test_cancel_independent_reservation_after_request_release():
    owner = manager()
    bank = owner.allocate_bank()
    checkpoint = owner.reserve_checkpoint(bank, 64)
    use = submit(owner, (StateAccessRow(bank, 0, bank.unit_ids[:1]),))
    owner.release_bank(bank)
    owner.cancel_checkpoint(checkpoint)
    owner.complete(use)
    assert_drained(owner)


def test_500_randomized_host_permission_fixture_cycles():
    rng = random.Random(109)
    owner = manager(requests=4, width=4, checkpoints=2, store=1)
    # 500 cycles, >20 transitions each, all terminal callback orders randomized.
    for cycle in range(500):
        banks = [owner.allocate_bank() for _ in range(4)]
        a = banks[0]
        checkpoint = owner.reserve_checkpoint(a, 64 * (cycle + 1))
        producer = submit(owner, (StateAccessRow(a, 0, (checkpoint.unit_id,), checkpoint),))
        trie = owner.retain(checkpoint, 'trie')
        store = owner.retain(checkpoint, 'store')
        accesses = [producer]
        for bank in banks[1:]:
            owner.bind_input(bank, checkpoint)
            accesses.append(submit(owner, (StateAccessRow(bank, checkpoint.unit_id, bank.unit_ids[:1]),)))
        accesses.append(owner.submit_save(store, stream=0))
        for ref in (checkpoint, trie, store):
            owner.release_reference(ref)
        for bank in banks:
            owner.release_bank(bank)
        rng.shuffle(accesses)
        for access in accesses:
            owner.complete(access, success=bool(cycle % 7))
            owner.complete(access)  # duplicate callbacks
        assert_drained(owner)


def test_store_cancel_keeps_save_credit_until_io_terminal():
    owner = manager(requests=1, store=1)
    bank = owner.allocate_bank()
    checkpoint = make_ready(owner, bank)
    store = owner.retain(checkpoint, 'store')
    access = owner.submit_save(store, stream=0)
    with pytest.raises(ValueError):
        owner.submit_save(store, stream=0)
    owner.release_reference(store)
    with pytest.raises(RuntimeError, match='save credit'):
        owner.retain(checkpoint, 'store')
    assert owner._store_refs == 1
    owner.complete(access)
    next_save = owner.retain(checkpoint, 'store')
    owner.release_reference(next_save)
    owner.release_reference(checkpoint)
    owner.release_bank(bank)
    assert_drained(owner)


def test_at_most_one_pending_internal_checkpoint_and_cancel_drains_it():
    owner = manager(requests=1, checkpoints=3)
    bank = owner.allocate_bank()
    checkpoint = owner.reserve_checkpoint(bank, 64)
    with pytest.raises(ValueError, match='One pending checkpoint'):
        owner.reserve_checkpoint(bank, 128)
    owner.release_bank(bank)
    assert not checkpoint.active
    owner.cancel_checkpoint(checkpoint)
    assert_drained(owner)


@pytest.mark.parametrize('submitted', [False, True])
@pytest.mark.parametrize('success', [False, True])
def test_released_reservation_allows_next_producer_without_losing_old_holds(submitted, success):
    owner = manager(requests=1, checkpoints=0, store=2)
    bank = owner.allocate_bank()
    first = owner.reserve_checkpoint(bank, 1024)
    producer = owner.submit_decode((bank,), checkpoints=(first,))
    store = owner.retain(first, 'store')
    io = owner.submit_save(store, stream=0)
    owner.release_reference(first)
    assert bank.pending_checkpoint is None
    assert first.unit_id in owner._units and not producer.finished and not io.finished
    second = owner.reserve_checkpoint(bank, 1088)
    assert second.unit_id != first.unit_id
    next_producer = owner.submit_decode((bank,), checkpoints=(second,)) if submitted else None
    owner.complete(producer, success=success)
    owner.complete(producer)  # A duplicate old acknowledgment cannot clear the new reservation.
    assert bank.pending_checkpoint is second
    assert first.unit_id in owner._units and owner._store_refs == 1
    owner.release_reference(store)
    assert first.unit_id in owner._units and owner._store_refs == 1
    owner.complete(io)
    assert first.unit_id not in owner._units and owner._store_refs == 0
    if next_producer is not None:
        owner.complete(next_producer)
        owner.release_reference(second)
    else:
        owner.cancel_checkpoint(second)
    owner.release_bank(bank)
    assert_drained(owner)


def test_foreign_owner_cannot_complete_access_with_same_numeric_unit_ids():
    owner, foreign = manager(), manager()
    bank, other_bank = owner.allocate_bank(), foreign.allocate_bank()
    assert bank.unit_ids == other_bank.unit_ids
    access = submit(owner, (StateAccessRow(bank, 0, bank.unit_ids[:1]),))
    with pytest.raises(ValueError, match='this state owner'):
        foreign.complete(access)
    owner.complete(access)
    owner.release_bank(bank)
    foreign.release_bank(other_bank)
    assert_drained(owner)
    assert_drained(foreign)


def test_decode_bank_snapshot_survives_cancel_and_later_reservation():
    owner = manager(requests=2, width=6)
    a, b = owner.allocate_bank(), owner.allocate_bank()
    checkpoint = make_ready(owner, a)
    owner.bind_input(b, checkpoint)
    access = owner.submit_decode((a, b))
    assert access.bank_ids == (a.unit_ids, b.unit_ids)
    assert access.read_only_input_ids == (-1, checkpoint.unit_id)
    assert len(access._units) == 1  # not 2*V unit-level holds
    later = owner.reserve_checkpoint(a, 128)
    assert later.unit_id not in access.bank_ids[0]
    owner.cancel_checkpoint(later)
    owner.clear_input(b)
    owner.release_reference(checkpoint)
    owner.release_bank(a)
    owner.release_bank(b)
    assert len(owner._units) == 2 * 6 + 1
    assert owner._banks
    owner.complete(access)
    assert_drained(owner)


def test_unused_reservation_can_roll_back_without_changing_prefetched_bank():
    owner = manager(requests=1)
    bank = owner.allocate_bank()
    original = bank.unit_ids
    checkpoint = owner.reserve_checkpoint(bank, 64)
    access = owner.submit_decode((bank,))
    assert checkpoint.unit_id not in access.bank_ids[0]
    owner.cancel_checkpoint(checkpoint)
    assert bank.unit_ids == original
    owner.release_bank(bank)
    owner.complete(access)
    assert_drained(owner)


def test_decode_overlap_and_mixed_prefill_holds():
    owner = manager(requests=1)
    bank = owner.allocate_bank()
    first = owner.submit_decode((bank,))
    second = owner.submit_decode((bank,))
    third = submit(owner, (StateAccessRow(bank, bank.unit_ids[0], bank.unit_ids[1:]),))
    owner.release_bank(bank)
    owner.complete(second)
    owner.complete(first)
    assert len(owner._units) == owner.capacity.bank_width
    owner.complete(third)
    assert_drained(owner)


def test_access_ids_remain_unique_after_unit_reuse_and_duplicate_acknowledgment():
    owner = manager(requests=1, width=1, checkpoints=0, store=0)
    bank = owner.allocate_bank()
    first = owner.submit_decode((bank,))
    second = owner.submit_decode((bank,))
    owner.release_bank(bank)
    owner.acknowledge(first.access_id)
    assert first.finished and not second.finished and owner.get_num_free() == 0
    owner.acknowledge(second.access_id)
    assert_drained(owner)
    replacement = owner.allocate_bank()
    third = owner.submit_decode((replacement,))
    owner.release_bank(replacement)
    assert first.access_id < second.access_id < third.access_id
    owner.acknowledge(first.access_id)
    owner.acknowledge(second.access_id)
    assert not third.finished and owner.get_num_free() == 0
    for invalid in (0, -1, True, '1', third.access_id + 1):
        with pytest.raises(ValueError, match='identity'):
            owner.acknowledge(invalid)
    owner.acknowledge(third.access_id)
    assert_drained(owner)


def test_foreign_empty_access_cannot_be_completed_by_another_owner():
    first, second = manager(), manager()
    own = submit(first, ())
    foreign = submit(second, ())
    assert own.access_id == foreign.access_id
    with pytest.raises(ValueError, match='does not belong'):
        first.complete(foreign)
    assert not own.finished and not foreign.finished
    first.acknowledge(own.access_id)
    second.acknowledge(foreign.access_id)
    assert_drained(first)
    assert_drained(second)


def test_forward_acknowledgment_cannot_release_store_io_hold():
    owner = manager(requests=1, width=1, checkpoints=1, store=1)
    bank = owner.allocate_bank()
    checkpoint = make_ready(owner, bank)
    store = owner.retain(checkpoint, 'store')
    access = owner.submit_save(store, stream=0)
    owner.release_reference(store)
    owner.release_reference(checkpoint)
    owner.release_bank(bank)
    with pytest.raises(ValueError, match='Store I/O'):
        owner.acknowledge(access.access_id)
    assert not access.finished and owner._store_refs == 1
    owner.complete(access)  # Only actual Store I/O completion may do this.
    assert_drained(owner)


def test_decode_ack_consumption_retires_exact_input_but_not_a0_or_later_read():
    owner = manager(requests=3)
    producer, first, second = [owner.allocate_bank() for _ in range(3)]
    checkpoint = make_ready(owner, producer)
    owner.bind_input(first, checkpoint)
    owner.bind_input(second, checkpoint)
    read_first = first.input_ref
    read_second = second.input_ref
    owner.release_reference(checkpoint)  # No trie holder; requests now own it.
    access = owner.submit_decode((first, second))
    later = owner.submit_decode((first,))
    owner.acknowledge(access.access_id, consumed_inputs=(True, False))
    assert first.input_ref is None and not read_first.active
    assert second.input_ref is read_second and read_second.active
    assert checkpoint.unit_id in owner._units
    owner.clear_input(second)
    assert checkpoint.unit_id in owner._units  # Prefetched GPU read is independent.
    owner.acknowledge(later.access_id, consumed_inputs=(True,))
    assert checkpoint.unit_id not in owner._units
    for bank in (producer, first, second):
        owner.release_bank(bank)
    assert_drained(owner)


@pytest.mark.parametrize('same_unit', [False, True])
def test_decode_ack_never_clears_a_later_logical_binding(same_unit):
    owner = manager(requests=2)
    producer, consumer = [owner.allocate_bank() for _ in range(2)]
    old = make_ready(owner, producer)
    owner.bind_input(consumer, old)
    access = owner.submit_decode((consumer,))
    new = old if same_unit else make_ready(owner, producer, boundary=128)
    owner.bind_input(consumer, new)
    current = consumer.input_ref
    owner.acknowledge(access.access_id, consumed_inputs=(True,))
    assert consumer.input_ref is current and current.active
    owner.acknowledge(access.access_id, consumed_inputs=(True,))  # Duplicate old notification.
    assert consumer.input_ref is current
    owner.release_bank(consumer)
    owner.release_reference(old)
    owner.release_reference(new)
    owner.release_bank(producer)
    assert_drained(owner)


@pytest.mark.parametrize('consumed', [(1,), (), (True, False), [True]])
def test_decode_ack_bad_consumption_does_not_complete_or_release(consumed):
    owner = manager(requests=1)
    bank = owner.allocate_bank()
    checkpoint = make_ready(owner, bank)
    owner.bind_input(bank, checkpoint)
    access = owner.submit_decode((bank,))
    with pytest.raises(ValueError, match='Consumption must match'):
        owner.acknowledge(access.access_id, consumed_inputs=consumed)
    assert not access.finished and bank.input_ref.active
    owner.acknowledge(access.access_id)  # Unknown consumption must preserve reference.
    assert bank.input_ref.active
    owner.release_reference(checkpoint)
    owner.release_bank(bank)
    assert_drained(owner)


def test_cancelled_forward_drain_preserves_monotonic_tickets_and_fails_outputs():
    owner = manager()
    bank = owner.allocate_bank()
    checkpoint = owner.reserve_checkpoint(bank, 64)
    producer = owner.submit_decode((bank,), checkpoints=(checkpoint,))
    retained = owner.retain(checkpoint, 'trie')
    owner.bind_input(bank, retained)
    consumer = owner.submit_decode((bank,), wait_for=(producer,))
    with pytest.raises(RuntimeError, match='closed'):
        owner.finish_cancelled_forwards_after_worker_drain()
    assert not producer.finished and not consumer.finished
    owner.release_bank(bank)
    owner.finish_cancelled_forwards_after_worker_drain()
    assert producer.finished and consumer.finished and retained.phase == 'failed'
    assert not owner._banks and not owner._accesses
    owner.release_reference(retained)
    owner.finish_cancelled_forwards_after_worker_drain()
    assert_drained(owner)
    replacement = owner.allocate_bank()
    fresh = owner.submit_decode((replacement,))
    assert fresh.access_id > consumer.access_id
    owner.acknowledge(producer.access_id)
    assert not fresh.finished
    owner.release_bank(replacement)
    owner.finish_cancelled_forwards_after_worker_drain()
    assert_drained(owner)


def test_model_drain_rejects_store_ownership_without_completing_anything():
    owner = manager()
    bank = owner.allocate_bank()
    checkpoint = owner.reserve_checkpoint(bank, 64)
    producer = owner.submit_decode((bank,), checkpoints=(checkpoint,))
    store = owner.retain(checkpoint, 'store')
    saving = owner.submit_save(store, stream=0)
    owner.release_bank(bank)
    with pytest.raises(RuntimeError, match='Store I/O'):
        owner.finish_cancelled_forwards_after_worker_drain()
    assert not producer.finished and not saving.finished
    owner.complete(producer)
    owner.complete(saving)
    owner.release_reference(store)
    owner.finish_cancelled_forwards_after_worker_drain()
    assert_drained(owner)


def test_worker_drain_rejects_unfinished_foreign_dependency_before_completion():
    first, second = manager(), manager()
    a, b = first.allocate_bank(), second.allocate_bank()
    producer = first.submit_decode((a,))
    consumer = second.submit_decode((b,), wait_for=(producer,))
    second.release_bank(b)
    with pytest.raises(RuntimeError, match='external dependency'):
        second.finish_cancelled_forwards_after_worker_drain()
    assert not consumer.finished and second._units
    first.release_bank(a)
    first.finish_cancelled_forwards_after_worker_drain()
    second.finish_cancelled_forwards_after_worker_drain()
    assert_drained(first)
    assert_drained(second)


def test_invalidation_is_transitive_but_not_completion_or_earlier_bank_failure():
    owner = manager(requests=4, width=1, checkpoints=4)
    a, b, c, unrelated = (owner.allocate_bank() for _ in range(4))
    earlier = owner.submit_decode((a,))
    first = owner.reserve_checkpoint(a, 64)
    producer = owner.submit_decode((a,), checkpoints=(first,))
    independent = owner.submit_decode((unrelated,))
    owner.bind_input(b, first)
    second = owner.reserve_checkpoint(b, 128)
    consumer = owner.submit_decode((b,), checkpoints=(second,))
    owner.bind_input(c, second)
    dependent = owner.submit_decode((c,))
    continuation = owner.submit_decode((a,))
    free, units = owner.get_num_free(), set(owner._units)
    affected = owner.invalidate(producer.access_id)
    assert affected == (producer.access_id, consumer.access_id, dependent.access_id, continuation.access_id)
    assert first.phase == second.phase == 'failed' and first.producing and second.producing
    assert not earlier.failed and not independent.failed and not unrelated.failed
    assert all(bank.failed for bank in (a, b, c))
    assert owner.get_num_free() == free and set(owner._units) == units
    assert not any(access.finished for access in owner._accesses.values())
    with pytest.raises(RuntimeError, match='acknowledged as successful'):
        owner.acknowledge(consumer.access_id)
    with pytest.raises(ValueError, match='Failed compact request bank'):
        owner.submit_decode((b,))
    with pytest.raises(ValueError, match='ordered-readable'):
        owner.bind_input(unrelated, first)
    with pytest.raises(ValueError, match='failed compact dependency'):
        owner.submit_decode((unrelated,), wait_for=(producer,))
    for bank in (a, b, c, unrelated):
        owner.release_bank(bank)
    assert set(owner._units) == units  # All whole-bank/producer/reader accesses still hold storage.
    owner.acknowledge(earlier.access_id)
    owner.acknowledge(independent.access_id)
    for access in (producer, consumer, dependent, continuation):
        owner.complete(access)  # Even a nominally successful completion cannot resurrect failed output.
        assert access.failed
    assert first.phase == second.phase == 'failed'
    owner.invalidate(producer.access_id)  # Retired ticket cannot affect later owners.
    assert_drained(owner)


def test_completed_failed_producer_does_not_allow_queued_reader_success():
    owner = manager(width=1)
    a, b = owner.allocate_bank(), owner.allocate_bank()
    checkpoint = owner.reserve_checkpoint(a, 64)
    producer = owner.submit_decode((a,), checkpoints=(checkpoint,))
    owner.bind_input(b, checkpoint)
    consumer = owner.submit_decode((b,))
    owner.complete(producer, success=False)
    owner.release_reference(checkpoint)
    with pytest.raises(RuntimeError, match='Failed compact work'):
        owner.acknowledge(consumer.access_id)
    assert not consumer.finished
    owner.complete(consumer)
    assert consumer.failed
    owner.release_bank(a)
    owner.release_bank(b)
    assert_drained(owner)


@pytest.mark.parametrize('width', [1, 4, 6])
def test_direct_load_owns_existing_bank_without_store_credit_or_extra_unit(width):
    owner = manager(requests=1, width=width, checkpoints=0, store=0)
    bank = owner.allocate_bank()
    original, free = bank.unit_ids, owner.get_num_free()
    load = owner.submit_load(bank, 64)
    assert load._writes[0].unit_id == original[0]
    assert bank.unit_ids is original and owner.get_num_free() == free == 0
    assert owner._store_refs == 0
    for operation in (lambda: owner.submit_load(bank, 128), lambda: owner.submit_decode((bank,)),
                      lambda: submit(owner, (StateAccessRow(bank, 0, (original[-1],)),)),
                      lambda: owner.reserve_checkpoint(bank, 64)):
        with pytest.raises(ValueError, match='Store I/O'):
            operation()
    with pytest.raises(ValueError, match='forward acknowledgment'):
        owner.acknowledge(load.access_id)
    owner.complete(load)
    assert bank.loaded_input == (original[0], 64)
    assert bank.pending_load is None
    forward = owner.submit_decode((bank,))
    owner.consume_loaded_inputs(forward)
    assert bank.loaded_input is None
    owner.release_bank(bank)
    assert owner.get_num_free_runtime_slots() == 0
    owner.acknowledge(forward.access_id)
    assert_drained(owner)


@pytest.mark.parametrize('failure', [False, True])
def test_load_cancel_never_recycles_before_actual_io_terminal(failure):
    owner = manager(requests=1, checkpoints=0, store=0)
    bank = owner.allocate_bank()
    load = owner.submit_load(bank, 64)
    target = load._writes[0].unit_id
    owner.release_bank(bank)
    assert target in owner._units
    assert owner.get_num_free_runtime_slots() == 0
    with pytest.raises(RuntimeError, match='Store I/O'):
        owner.finish_cancelled_forwards_after_worker_drain()
    owner.complete(load, success=not failure)
    assert bank.loaded_input is None
    assert_drained(owner)
    fresh = owner.allocate_bank()
    owner.complete(load)  # duplicate old completion must not bind a new owner
    assert fresh.loaded_input is None
    owner.release_bank(fresh)
    assert_drained(owner)


@pytest.mark.parametrize('invalidate', [False, True])
def test_partial_or_invalidated_load_cannot_publish_private_input(invalidate):
    owner = manager(requests=1)
    bank = owner.allocate_bank()
    load = owner.submit_load(bank, 64)
    if invalidate:
        owner.invalidate(load.access_id)
    owner.complete(load, success=invalidate)
    assert bank.failed and bank.loaded_input is None
    with pytest.raises(ValueError, match='Failed compact'):
        owner.submit_decode((bank,))
    owner.release_bank(bank)
    assert_drained(owner)


@pytest.mark.parametrize('boundary', [0, -1, True, 1.5])
def test_load_boundary_rejection_is_atomic(boundary):
    owner = manager()
    bank = owner.allocate_bank()
    with pytest.raises(ValueError, match='positive exact'):
        owner.submit_load(bank, boundary)
    assert not bank.accesses and bank.pending_load is None
    owner.release_bank(bank)
    assert_drained(owner)


def test_load_rejects_submitted_model_work_and_ready_unconsumed_input():
    owner = manager()
    bank = owner.allocate_bank()
    forward = owner.submit_decode((bank,))
    with pytest.raises(ValueError, match='idle, unbound'):
        owner.submit_load(bank, 64)
    owner.complete(forward)
    load = owner.submit_load(bank, 64)
    owner.complete(load)
    with pytest.raises(ValueError, match='idle, unbound'):
        owner.submit_load(bank, 128)
    # Load-ready cancellation has no fabricated model or restore event.
    owner.release_bank(bank)
    assert_drained(owner)


def test_loaded_identity_cannot_be_replaced_by_shared_input():
    owner = manager(requests=2)
    producer, bank = owner.allocate_bank(), owner.allocate_bank()
    ref = make_ready(owner, producer)
    owner.complete(owner.submit_load(bank, 128))
    source = bank.loaded_input[0]
    with pytest.raises(ValueError, match='also bind'):
        owner.bind_input(bank, ref)
    with pytest.raises(ValueError, match='must read the loaded'):
        submit(owner, (StateAccessRow(bank, 0, (bank.unit_ids[-1],)),))
    assert bank.loaded_input == (source, 128)
    access = submit(owner, (StateAccessRow(bank, source, (bank.unit_ids[-1],)),))
    assert bank.loaded_input is None
    owner.complete(access)
    owner.release_reference(ref)
    owner.release_bank(producer)
    owner.release_bank(bank)
    assert_drained(owner)


def test_production_owner_excludes_host_id_reference_planner():
    """The exact-ID test fixture must not become a second serving submit
    API."""
    from lmdeploy.pytorch.paging import compact_state

    assert not hasattr(compact_state, 'StateAccessRow')
    for name in ('prepare_prefill_output', 'checkpoint_ready', 'submit'):
        assert not hasattr(SSMStateManager, name)


def test_production_completion_preserves_producer_and_store_holds():
    owner = SSMStateManager(SSMStateCapacity(2, 1, 1, 1))
    bank = owner.allocate_bank(prefill=True)
    reference = owner.reserve_checkpoint(bank, 64)
    producer = owner.submit_prefill((bank,), checkpoints=(reference,))
    retained = owner.retain(reference, 'store')
    assert retained.shares_storage(reference)
    assert retained.pending_producer is producer and reference.producing
    save = owner.submit_save(retained, stream=-1, wait_for=(retained.pending_producer,))
    assert not producer.finished and not save.finished
    owner.release_bank(bank)
    owner.release_reference(reference)
    owner.release_reference(retained)
    # Logical release must not erase producer ordering or physical I/O holds.
    assert retained.pending_producer is producer
    assert retained.shares_storage(reference)
    with pytest.raises(ValueError, match='Dependency must complete'):
        owner.complete(save)
    owner.acknowledge(producer.access_id)
    assert retained.pending_producer is None and not reference.producing
    assert producer.finished and not save.finished
    assert owner.get_num_free_store_saves() == 0
    owner.complete(save)
    assert save.finished
    assert owner.get_num_free_store_saves() == 1
    assert owner.get_num_free() == owner.capacity.num_units - 1


def test_reference_identity_does_not_follow_recycled_numeric_unit_id():
    owner = SSMStateManager(SSMStateCapacity(2, 1, 1))
    bank = owner.allocate_bank(prefill=True)
    previous = owner.reserve_checkpoint(bank, 64)
    owner.cancel_checkpoint(previous)
    current = owner.reserve_checkpoint(bank, 128)
    assert previous.unit_id == current.unit_id
    assert not previous.shares_storage(current)
    owner.cancel_checkpoint(current)
    owner.release_bank(bank)
    assert owner.get_num_free() == owner.capacity.num_units - 1
