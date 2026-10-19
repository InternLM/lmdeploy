# Copyright (c) OpenMMLab. All rights reserved.
"""Phase admission and guaranteed decode expansion contracts."""
import pytest

from lmdeploy.pytorch.config import CompactStateCapacity
from lmdeploy.pytorch.paging.compact_state import CompactStateManager
from tests.pytorch.compact_state_reference import StateAccessRow, freeze_ready, prepare_prefill_output, submit


def manager(requests=2, width=4, checkpoints=1, store=1):
    return CompactStateManager(CompactStateCapacity(requests + 1, width, checkpoints, store))


def drained(owner):
    assert not owner._units and not owner._banks and not owner._accesses
    assert owner.get_num_free() == owner.capacity.num_units - 1


@pytest.mark.parametrize('width', [1, 4, 6])
def test_prefill_allocates_one_then_two_not_decode_ring(width):
    owner = manager(width=width)
    free = owner.get_num_free()
    bank = owner.allocate_bank(prefill=True)
    assert len(bank.unit_ids) == 1 and bank.phase == 'prefill'
    first = prepare_prefill_output(owner, bank, 0)
    assert owner.get_num_free() == free - 1
    a = submit(owner, (StateAccessRow(bank, 0, (first,)),))
    second = prepare_prefill_output(owner, bank, first)
    assert second != first and len(bank.unit_ids) == 2
    b = submit(owner, (StateAccessRow(bank, first, (second,)),))
    # Same-stream chunk reuse never allocates a full decode ring.
    assert prepare_prefill_output(owner, bank, second) == first
    assert owner.get_num_free() == free - 2
    with pytest.raises(ValueError, match='fully admitted'):
        owner.submit_decode((bank,))
    owner.complete(a)
    owner.complete(b)
    owner.release_bank(bank)
    drained(owner)


@pytest.mark.parametrize('width', [1, 4, 6])
def test_promotion_preserves_prefetched_producer_and_physical_ids(width):
    owner = manager(width=width)
    bank = owner.allocate_bank(prefill=True)
    before = bank.unit_ids
    producer = submit(owner, (StateAccessRow(bank, 0, before),))
    producer_ids = tuple(u.unit_id for u in producer._units)
    assert owner.promote_to_decode(bank)
    assert bank.phase == 'decode' and len(bank.unit_ids) == width + 1
    assert bank.unit_ids[:1] == before
    assert tuple(u.unit_id for u in producer._units) == producer_ids
    decode = owner.submit_decode((bank,))
    submitted = decode.bank_ids
    assert submitted == (bank.unit_ids,)
    free = owner.get_num_free()
    assert owner.promote_to_decode(bank)  # idempotent, no new allocation/table
    assert decode.bank_ids is submitted and owner.get_num_free() == free
    owner.release_bank(bank)
    assert all(i in owner._units for i in submitted[0])
    owner.complete(producer)
    assert bank.auxiliary_id in owner._banks
    owner.complete(decode)
    drained(owner)


@pytest.mark.parametrize('width', [1, 4, 6])
def test_optional_checkpoint_cannot_steal_admitted_decode_capacity(width):
    owner = manager(width=width, checkpoints=0, store=0)
    banks = tuple(owner.allocate_bank(prefill=True) for _ in range(2))
    assert owner.get_num_free() == 2 * width
    assert owner.get_num_free_optional() == 0
    assert owner.get_num_free_runtime() == 0
    producer = owner.submit_prefill(banks)
    original = producer.bank_ids
    for bank in banks:
        with pytest.raises(RuntimeError, match='reserved decode'):
            owner.reserve_checkpoint(bank, 64)
        owner.prepare_prefill_bank(bank, private_input=True)
        assert len(bank.unit_ids) == 2
        assert owner.promote_to_decode(bank)
        assert bank.unit_ids[:1] == original[bank.auxiliary_id - 1]
    assert owner.get_num_free() == 0
    assert producer.bank_ids == original and not producer.finished
    decode = owner.submit_decode(banks)
    for bank in banks:
        owner.release_bank(bank)
    owner.complete(producer)
    owner.complete(decode)
    drained(owner)


def test_promotion_capacity_fault_keeps_result_and_rolls_back_nothing(monkeypatch):
    owner = manager(checkpoints=0, store=0)
    bank = owner.allocate_bank(prefill=True)
    producer = owner.submit_prefill((bank,))
    original = bank.unit_ids
    all_units = set(owner._units)
    # Optional clients cannot cause this now. Inject an allocator invariant
    # failure to verify the transition itself never partially publishes IDs.
    with monkeypatch.context() as patch:
        patch.setattr(owner, 'get_num_free', lambda: 0)
        assert not owner.promote_to_decode(bank)
    assert bank.phase == 'prefill' and bank.unit_ids is original
    assert set(owner._units) == all_units and not producer.finished
    assert owner.promote_to_decode(bank)
    owner.complete(producer)
    owner.release_bank(bank)
    drained(owner)


def test_small_prefill_does_not_expand_auxiliary_request_capacity():
    owner = manager(requests=1)
    bank = owner.allocate_bank(prefill=True)
    assert owner.get_num_free() > 0
    with pytest.raises(RuntimeError, match='complete state bank'):
        owner.allocate_bank(prefill=True)
    owner.release_bank(bank)
    drained(owner)


def test_shared_checkpoint_needs_only_one_private_prefill_destination():
    owner = manager()
    donor = owner.allocate_bank(prefill=True)
    first = donor.unit_ids[0]
    access = submit(owner, (StateAccessRow(donor, 0, (first,)),))
    owner.complete(access)
    checkpoint = freeze_ready(owner, donor, first, 64)
    target = owner.allocate_bank(prefill=True)
    owner.bind_input(target, checkpoint)
    free = owner.get_num_free()
    output = prepare_prefill_output(owner, target, checkpoint.unit_id)
    assert len(target.unit_ids) == 1 and output != checkpoint.unit_id
    assert owner.get_num_free() == free
    consumer = submit(owner, (StateAccessRow(target, checkpoint.unit_id, (output,)),))
    owner.release_reference(checkpoint)
    owner.release_bank(donor)
    owner.release_bank(target)
    assert first in owner._units
    owner.complete(consumer)
    drained(owner)


def test_pending_store_get_blocks_growth_and_completed_get_needs_two():
    owner = manager()
    bank = owner.allocate_bank(prefill=True)
    load = owner.submit_load(bank, 1024)
    before = bank.unit_ids
    with pytest.raises(ValueError, match='awaiting Store'):
        prepare_prefill_output(owner, bank, before[0])
    with pytest.raises(ValueError, match='awaiting Store'):
        owner.promote_to_decode(bank)
    assert bank.unit_ids is before
    owner.complete(load)
    assert bank.loaded_input == (before[0], 1024)
    with pytest.raises(ValueError, match='loaded private input'):
        prepare_prefill_output(owner, bank, 0)
    output = prepare_prefill_output(owner, bank, before[0])
    assert len(bank.unit_ids) == 2 and output != before[0]
    assert bank.loaded_input == (before[0], 1024)
    producer = submit(owner, (StateAccessRow(bank, before[0], (output,)),))
    assert bank.loaded_input is None
    assert owner.promote_to_decode(bank)
    owner.release_bank(bank)
    assert before[0] in owner._units
    owner.complete(producer)
    drained(owner)


def test_cancel_pending_load_keeps_destination_until_io_completes():
    owner = manager()
    bank = owner.allocate_bank(prefill=True)
    first = bank.unit_ids[0]
    load = owner.submit_load(bank, 64)
    owner.release_bank(bank)
    assert first in owner._units
    with pytest.raises(ValueError, match='not live'):
        owner.promote_to_decode(bank)
    owner.complete(load, success=False)
    drained(owner)


def test_reject_foreign_input_without_allocating_output():
    owner = manager()
    bank = owner.allocate_bank(prefill=True)
    other = owner.allocate_bank(prefill=True)
    free = owner.get_num_free()
    for invalid in [-1, 99999, other.unit_ids[0]]:
        with pytest.raises(ValueError, match='bound immutable'):
            prepare_prefill_output(owner, bank, invalid)
    assert owner.get_num_free() == free and len(bank.unit_ids) == 1
    owner.release_bank(bank)
    owner.release_bank(other)
    drained(owner)


def test_decode_bank_cannot_use_prefill_preparation():
    owner = manager()
    bank = owner.allocate_bank()
    with pytest.raises(ValueError, match='prefill bank'):
        prepare_prefill_output(owner, bank, 0)
    owner.release_bank(bank)
    drained(owner)


def test_promotion_excludes_frozen_unit_until_actual_store_put_completion():
    owner = manager(requests=1, checkpoints=0, store=1)
    bank = owner.allocate_bank(prefill=True)
    first = bank.unit_ids[0]
    producer = submit(owner, (StateAccessRow(bank, 0, (first,)),))
    owner.complete(producer)
    checkpoint = freeze_ready(owner, bank, first, 64)
    store = owner.retain(checkpoint, 'store')
    io = owner.submit_save(store, stream=7)
    owner.bind_input(bank, checkpoint)
    assert owner.promote_to_decode(bank)
    assert first not in bank.unit_ids
    decode = owner.submit_decode((bank,))
    assert decode.read_only_input_ids == (first,)
    owner.release_reference(checkpoint)
    owner.release_reference(store)
    owner.release_bank(bank)
    owner.complete(decode)
    assert first in owner._units and not io.finished
    owner.complete(io)
    assert owner._store_refs == 0
    drained(owner)
