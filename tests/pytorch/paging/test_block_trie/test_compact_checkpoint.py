# Copyright (c) OpenMMLab. All rights reserved.
"""Actual trie exact identity with direct compact-unit ownership, no GPU
model."""

import pytest

from lmdeploy.pytorch.config import CacheConfig, CompactStateCapacity, SchedulerConfig
from lmdeploy.pytorch.messages import SequenceMeta
from lmdeploy.pytorch.paging import Scheduler
from lmdeploy.pytorch.paging.block_trie import BlockTrie
from lmdeploy.pytorch.paging.block_trie.compact_checkpoint import CompactStateCheckpointLifecycle
from lmdeploy.pytorch.paging.compact_state import CompactStateManager
from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy
from tests.pytorch.compact_state_reference import freeze_ready


def _environment(width=4, checkpoint_budget=8, request_slots=5):
    # Scheduler is a real sequence/KV fixture, not the not-yet-enabled compact
    # APC scheduler. This trie explicitly owns compact checkpoint references.
    scheduler = Scheduler(SchedulerConfig(max_batches=4, max_session_len=4096),
                          CacheConfig(max_batches=4, block_size=64, num_cpu_blocks=0, num_gpu_blocks=512,
                                      enable_prefix_caching=True),
                          seq_meta=SequenceMeta(64, strategy=ARSequenceStrategy()))
    owner = CompactStateManager(CompactStateCapacity(request_slots, width, checkpoint_budget))
    trie = BlockTrie(allocator=scheduler.block_manager.allocator, block_size=64, enabled=True,
                     checkpoint_state_manager=owner)
    assert isinstance(trie.state_checkpoints, CompactStateCheckpointLifecycle)
    return scheduler, owner, trie


def _sequence(scheduler, owner, trie, tokens, *, match=False):
    seq = scheduler.add_session(len(scheduler.sessions)).add_sequence(tokens)
    if match:
        trie.match(seq)
    owner.allocate(seq)
    scheduler.block_manager.allocate(seq)
    trie.allocate(seq)
    return seq


def _submit(owner, seq):
    reference = seq.prefix_cache.pending_save.node.state_checkpoint.state_reference
    access = owner.submit_prefill((owner.bank_for(seq),), checkpoints=(reference,))
    return reference, access


@pytest.mark.parametrize('width', [1, 4, 6])
def test_exact_trie_hit_binds_two_readers_and_internal_boundary_producer(width):
    scheduler, owner, trie = _environment(width)
    lifecycle = trie.state_checkpoints
    tokens = list(range(2120))
    producer = _sequence(scheduler, owner, trie, tokens)
    slot = lifecycle.reserve_save(producer, step=2048)
    source_node = producer.prefix_cache.pending_save.node
    reservation, produced = _submit(owner, producer)
    assert reservation.producing and reservation.phase == 'submitted'
    assert lifecycle.publish_save(producer)
    assert source_node.state_checkpoint.state_reference.owner == 'trie'
    assert not reservation.active
    assert lifecycle.evict(1) == 0  # Publication is not producer completion.
    owner.acknowledge(produced.access_id)
    assert not reservation.producing
    readers = [_sequence(scheduler, owner, trie, tokens, match=True) for _ in range(2)]
    for seq in readers:
        assert seq.num_history_ids == 2048
        assert seq.num_token_ids == 72
        assert lifecycle.bind_restore(seq) == slot
        assert owner.bank_for(seq).input_ref.unit_id == slot
    assert source_node.state_checkpoint.pin_count == 2
    assert lifecycle.evict(1) == 0
    # Export c=2112 while the selected input interval still ends at e=2120.
    cp = lifecycle.reserve_save(readers[0], step=2112)
    destination = readers[0].prefix_cache.pending_save.node
    reference, forward = _submit(owner, readers[0])
    assert forward.read_only_input_ids == (slot,)
    assert reference.boundary == 2112 and cp not in forward.bank_ids[0]
    assert readers[0].num_history_ids == 2048 and readers[0].num_token_ids == 72
    assert lifecycle.publish_save(readers[0])
    assert destination.state_checkpoint.step == 2112
    assert len(destination.state_checkpoint.exact_match_data.token_ids) == 2112
    other = owner.submit_prefill((owner.bank_for(readers[1]),))
    # Completed first reader unpins only its own KV restore, not the other.
    owner.acknowledge(forward.access_id)
    lifecycle.unpin_restore(readers[0])
    assert source_node.state_checkpoint.pin_count == 1
    owner.acknowledge(other.access_id)
    lifecycle.unpin_restore(readers[1])
    lifecycle.release_checkpoint(source_node)
    assert source_node.state_checkpoint is None
    assert slot in owner._units  # Physical request references outlive the node.
    owner.clear_input(owner.bank_for(readers[0]))
    assert slot in owner._units
    owner.clear_input(owner.bank_for(readers[1]))
    assert slot not in owner._units
    lifecycle.release_checkpoint(destination)
    for seq in (producer, *readers):
        owner.free(seq)
    assert not owner._units and not owner._banks and not owner._accesses


def test_node_replacement_never_overwrites_a_reader_held_unit():
    scheduler, owner, trie = _environment()
    lifecycle = trie.state_checkpoints
    tokens = list(range(128))
    first = _sequence(scheduler, owner, trie, tokens)
    old_slot = lifecycle.reserve_save(first, step=80)
    node = first.prefix_cache.pending_save.node
    _, produced = _submit(owner, first)
    owner.acknowledge(produced.access_id)
    lifecycle.publish_save(first)
    reader = _sequence(scheduler, owner, trie, tokens, match=True)
    assert lifecycle.bind_restore(reader) == old_slot
    read = owner.submit_prefill((owner.bank_for(reader),))
    owner.acknowledge(read.access_id)
    lifecycle.unpin_restore(reader)
    old_reference = owner.bank_for(reader).input_ref
    new_slot = lifecycle.reserve_save(first, step=96)
    assert new_slot > 0 and new_slot != old_slot
    assert node.state_checkpoint.slot == new_slot and not node.state_checkpoint.published
    assert old_reference.active and old_reference.boundary == 80
    lifecycle.discard_save(first)
    assert old_slot in owner._units
    owner.free(reader)
    owner.free(first)
    assert not owner._units


def test_partial_kv_allocation_failure_rolls_back_unit_reservation(monkeypatch):
    scheduler, owner, trie = _environment()
    seq = _sequence(scheduler, owner, trie, list(range(128)))
    free = owner.get_num_free()
    monkeypatch.setattr(trie.allocator, 'allocate', lambda *a, **k: (_ for _ in ()).throw(MemoryError('KV full')))
    assert trie.state_checkpoints.reserve_save(seq, step=80) == -1
    assert owner.get_num_free() == free
    assert owner.bank_for(seq).pending_checkpoint is None
    assert not seq.prefix_cache.pending_save.is_pending


@pytest.mark.parametrize('partial', [False, True])
def test_failed_publication_retires_exact_resources_until_producer_completes(monkeypatch, partial):
    scheduler, owner, trie = _environment()
    seq = _sequence(scheduler, owner, trie, list(range(128)))
    lifecycle = trie.state_checkpoints
    before = scheduler.block_manager.get_num_free_gpu_blocks()
    slot = lifecycle.reserve_save(seq, step=80 if partial else 64)
    node = seq.prefix_cache.pending_save.node
    reference, access = _submit(owner, seq)

    def fail(*args):
        raise RuntimeError('invalid exact identity')

    monkeypatch.setattr(lifecycle, '_snapshot_match_data', fail)
    with pytest.raises(RuntimeError, match='invalid exact identity'):
        lifecycle.publish_save(seq)
    assert node.state_checkpoint is None and not seq.prefix_cache.pending_save.is_pending
    assert not reference.active and reference.producing
    assert slot in owner._units
    lifecycle.collect_completed()
    assert scheduler.block_manager.get_num_free_gpu_blocks() == before - int(partial)
    owner.free(seq)
    assert owner.get_num_free_runtime() == 3  # Closing bank is still held.
    owner.complete(access, success=False)
    lifecycle.collect_completed()
    lifecycle.collect_completed()  # No double-free.
    assert scheduler.block_manager.get_num_free_gpu_blocks() == before
    assert not owner._units and not lifecycle._retired_tails


def test_reservation_cannot_publish_before_submission():
    scheduler, owner, trie = _environment()
    seq = _sequence(scheduler, owner, trie, list(range(128)))
    lifecycle = trie.state_checkpoints
    lifecycle.reserve_save(seq, step=64)
    node = seq.prefix_cache.pending_save.node
    with pytest.raises(RuntimeError, match='submitted producer'):
        lifecycle.publish_save(seq)
    assert node.state_checkpoint is None and owner.bank_for(seq).pending_checkpoint is None


@pytest.mark.parametrize('method,args', [('prepare_restore_batch', ([],)),
                                         ('reserve_prefill_save_batch', ([],)),
                                         ('reserve_decode_save_batch', ([], 64))])
def test_compact_lifecycle_never_emits_legacy_state_copies(method, args):
    _, _, trie = _environment()
    with pytest.raises(NotImplementedError):
        getattr(trie.state_checkpoints, method)(*args)


@pytest.mark.parametrize('failure', ['length', 'foreign_bank', 'submitted', 'foreign_owner'])
def test_whole_bank_checkpoint_submission_validates_atomically(failure):
    owner = CompactStateManager(CompactStateCapacity(3, 4, 4))
    banks = (owner.allocate_bank(), owner.allocate_bank())
    refs = tuple(owner.reserve_checkpoint(bank, 64) for bank in banks)
    checkpoints = refs
    if failure == 'length':
        checkpoints = refs[:1]
    elif failure == 'foreign_bank':
        checkpoints = refs[::-1]
    elif failure == 'submitted':
        owner.submit_decode((banks[1],), checkpoints=(refs[1],))
    else:
        foreign = CompactStateManager(CompactStateCapacity(2, 4, 1))
        checkpoints = (refs[0], foreign.reserve_checkpoint(foreign.allocate_bank(), 64))
    accesses = set(owner._accesses)
    with pytest.raises(ValueError):
        owner.submit_decode(banks, checkpoints=checkpoints)
    assert set(owner._accesses) == accesses
    assert refs[0].phase == 'reserved' and not refs[0].producing
    assert not owner._units[refs[0].unit_id].writers


@pytest.mark.parametrize('already_pinned', [False, True])
def test_restore_binding_failure_preserves_prior_pin_and_selection(monkeypatch, already_pinned):
    scheduler, owner, trie = _environment()
    lifecycle = trie.state_checkpoints
    tokens = list(range(128))
    producer = _sequence(scheduler, owner, trie, tokens)
    slot = lifecycle.reserve_save(producer, step=64)
    _, access = _submit(owner, producer)
    owner.acknowledge(access.access_id)
    lifecycle.publish_save(producer)
    reader = _sequence(scheduler, owner, trie, tokens, match=True)
    restore = reader.prefix_cache.restore
    checkpoint = restore.node.state_checkpoint
    if already_pinned:
        assert lifecycle.pin_restore(reader)

    def fail(*args):
        raise ValueError('binding rejected')

    monkeypatch.setattr(owner, 'bind_input', fail)
    with pytest.raises(ValueError, match='binding rejected'):
        lifecycle.bind_restore(reader)
    assert restore.slot == slot and restore.is_selected
    assert restore.pinned == already_pinned
    assert checkpoint.pin_count == int(already_pinned)
    assert owner.bank_for(reader).input_ref is None


def test_failed_producer_cannot_bind_or_leak_restore_pin():
    scheduler, owner, trie = _environment()
    lifecycle = trie.state_checkpoints
    tokens = list(range(128))
    producer = _sequence(scheduler, owner, trie, tokens)
    lifecycle.reserve_save(producer, step=64)
    _, access = _submit(owner, producer)
    lifecycle.publish_save(producer)
    reader = _sequence(scheduler, owner, trie, tokens, match=True)
    node = reader.prefix_cache.restore.node
    owner.complete(access, success=False)
    with pytest.raises(RuntimeError, match='producer failed'):
        lifecycle.bind_restore(reader)
    assert not reader.prefix_cache.restore.pinned and node.state_checkpoint.pin_count == 0
    assert owner.bank_for(reader).input_ref is None
    # Failure-coordinator integration must withdraw failed publications. This
    # owner-level guard is not a substitute for that production integration.
    lifecycle.release_checkpoint(node)
    owner.free(reader)
    owner.free(producer)
    assert not owner._units


def test_replacement_kv_failure_preserves_old_published_identity(monkeypatch):
    scheduler, owner, trie = _environment()
    lifecycle = trie.state_checkpoints
    tokens = list(range(128))
    seq = _sequence(scheduler, owner, trie, tokens)
    slot = lifecycle.reserve_save(seq, step=80)
    node = seq.prefix_cache.pending_save.node
    _, access = _submit(owner, seq)
    owner.acknowledge(access.access_id)
    lifecycle.publish_save(seq)
    checkpoint = node.state_checkpoint
    free = owner.get_num_free()
    monkeypatch.setattr(trie.allocator, 'allocate', lambda *a, **k: (_ for _ in ()).throw(MemoryError('KV full')))
    assert lifecycle.reserve_save(seq, step=96) == -1
    assert node.state_checkpoint is checkpoint and checkpoint.published and checkpoint.slot == slot
    assert checkpoint.state_reference.active and checkpoint.step == 80
    assert owner.get_num_free() == free and owner.bank_for(seq).pending_checkpoint is None
    assert list(lifecycle._index.unique_nodes()) == [node]


def test_index_failure_retired_tail_is_independent_of_reused_node(monkeypatch):
    scheduler, owner, trie = _environment()
    lifecycle = trie.state_checkpoints
    tokens = list(range(128))
    first = _sequence(scheduler, owner, trie, tokens)
    second = _sequence(scheduler, owner, trie, tokens)
    before = scheduler.block_manager.get_num_free_gpu_blocks()
    lifecycle.reserve_save(first, step=80)
    node = first.prefix_cache.pending_save.node
    old_tail = node.state_checkpoint.frozen_block_id
    _, old_access = _submit(owner, first)
    original_add = lifecycle._index.add

    def fail_after_index(node):
        original_add(node)
        raise RuntimeError('index publication failed')

    monkeypatch.setattr(lifecycle._index, 'add', fail_after_index)
    with pytest.raises(RuntimeError, match='index publication failed'):
        lifecycle.publish_save(first)
    assert not list(lifecycle._index.unique_nodes())
    monkeypatch.setattr(lifecycle._index, 'add', original_add)
    lifecycle.reserve_save(second, step=96)
    assert second.prefix_cache.pending_save.node is node
    replacement = node.state_checkpoint
    assert replacement.frozen_block_id != old_tail
    _, new_access = _submit(owner, second)
    lifecycle.publish_save(second)
    owner.complete(old_access, success=False)
    lifecycle.collect_completed()
    assert node.state_checkpoint is replacement and replacement.published
    assert scheduler.block_manager.get_num_free_gpu_blocks() == before - 1
    owner.acknowledge(new_access.access_id)
    lifecycle.release_checkpoint(node)
    lifecycle.collect_completed()
    assert scheduler.block_manager.get_num_free_gpu_blocks() == before
    owner.free(first)
    owner.free(second)
    assert not owner._units


def test_whole_bank_checkpoint_cross_stream_requires_producer_dependency():
    owner = CompactStateManager(CompactStateCapacity(3, 4, 2))
    producer_bank = owner.allocate_bank(stream=11)
    reader_bank = owner.allocate_bank(stream=22)
    reference = owner.reserve_checkpoint(producer_bank, 64)
    produced = owner.submit_decode((producer_bank,), stream=11, checkpoints=(reference,))
    owner.bind_input(reader_bank, reference)
    output = owner.reserve_checkpoint(reader_bank, 128)
    with pytest.raises(ValueError, match='producer dependency'):
        owner.submit_decode((reader_bank,), stream=22, checkpoints=(output,))
    assert output.phase == 'reserved' and not output.producing
    consumed = owner.submit_decode((reader_bank,), stream=22, wait_for=(produced,), checkpoints=(output,))
    owner.release_reference(reference)
    owner.release_reference(output)
    owner.release_bank(producer_bank)
    owner.release_bank(reader_bank)
    owner.complete(produced)
    owner.complete(consumed)
    assert not owner._units and not owner._accesses


def test_actual_cuda_completion_retains_trie_input_and_partial_tail():
    import torch

    if not torch.cuda.is_available():
        pytest.skip('CUDA event lifetime test')
    scheduler, owner, trie = _environment()
    lifecycle = trie.state_checkpoints
    tokens = list(range(128))
    producer = _sequence(scheduler, owner, trie, tokens)
    slot = lifecycle.reserve_save(producer, step=80)
    node = producer.prefix_cache.pending_save.node
    _, produced = _submit(owner, producer)
    lifecycle.publish_save(producer)
    readers = [_sequence(scheduler, owner, trie, tokens, match=True) for _ in range(2)]
    for reader in readers:
        assert lifecycle.bind_restore(reader) == slot
    consumed = [owner.submit_prefill((owner.bank_for(reader),)) for reader in readers]
    pool = torch.zeros(owner.capacity.num_units, 16, device='cuda')
    results = [torch.empty(16, device='cuda') for _ in readers]
    torch.cuda.synchronize()  # Test setup only, not a runtime requirement.
    torch.cuda._sleep(150_000_000)
    pool[slot].fill_(7)
    producer_done = torch.cuda.Event()
    producer_done.record()
    done = []
    for result in results:
        result.copy_(pool[slot])  # Test-only consumer; no model/zero-copy claim.
        event = torch.cuda.Event()
        event.record()
        done.append(event)
    for seq in (producer, *readers):
        owner.free(seq)
    assert not producer_done.query(), 'Delay must cover the cancellation/eviction window'
    assert lifecycle.evict(1) == 0 and node.state_checkpoint.pin_count == 2
    with pytest.raises(RuntimeError, match='pinned'):
        lifecycle.release_checkpoint(node)
    producer_done.synchronize()
    owner.acknowledge(produced.access_id)
    for reader, event, access, result in zip(readers, done, consumed, results):
        event.synchronize()
        torch.testing.assert_close(result, torch.full_like(result, 7), rtol=0, atol=0)
        owner.acknowledge(access.access_id)
        lifecycle.unpin_restore(reader)
    lifecycle.release_checkpoint(node)
    assert not owner._units and not owner._banks and not owner._accesses


def test_logical_eviction_does_not_promise_reader_held_physical_capacity():
    scheduler, owner, trie = _environment(checkpoint_budget=1)
    lifecycle = trie.state_checkpoints
    tokens = list(range(192))
    producer = _sequence(scheduler, owner, trie, tokens)
    slot = lifecycle.reserve_save(producer, step=64)
    node = producer.prefix_cache.pending_save.node
    _, access = _submit(owner, producer)
    owner.acknowledge(access.access_id)
    lifecycle.publish_save(producer)
    reader = _sequence(scheduler, owner, trie, tokens, match=True)
    lifecycle.bind_restore(reader)
    read = owner.submit_prefill((owner.bank_for(reader),))
    owner.acknowledge(read.access_id)
    lifecycle.unpin_restore(reader)
    others = [_sequence(scheduler, owner, trie, tokens) for _ in range(2)]
    assert owner.get_num_free_optional() == 0
    assert lifecycle.reserve_save(others[0], step=128) == -1
    assert node.state_checkpoint is None and slot in owner._units
    assert owner.bank_for(reader).input_ref.active
    assert owner.get_num_free_optional() == 0 and not others[0].prefix_cache.pending_save.is_pending
    owner.clear_input(owner.bank_for(reader))
    assert owner.get_num_free_optional() == 1
    assert lifecycle.reserve_save(others[0], step=128) > 0
    lifecycle.discard_save(others[0])
    for seq in (producer, reader, *others):
        owner.free(seq)
    assert not owner._units


def test_runtime_admission_evicts_enough_units_for_full_bank_not_one_slot():
    scheduler, owner, trie = _environment(width=4, checkpoint_budget=1, request_slots=3)
    lifecycle = trie.state_checkpoints
    seq = _sequence(scheduler, owner, trie, list(range(512)))
    for step in range(64, 385, 64):
        assert lifecycle.reserve_save(seq, step=step) > 0
        _, access = _submit(owner, seq)
        owner.acknowledge(access.access_id)
        lifecycle.publish_save(seq)
    assert owner.get_num_free_optional() == 0 and owner.get_num_free_runtime_slots() == 1
    assert owner.get_num_free_runtime() == 0
    assert len(list(lifecycle._index.unique_nodes())) == 6
    assert lifecycle.make_runtime_state_available()
    assert owner.get_num_free_optional() == owner.capacity.bank_width == 5
    assert len(list(lifecycle._index.unique_nodes())) == 1
    other = owner.allocate_bank()
    # No free request auxiliary slot: dropping the last CP cannot help.
    assert not lifecycle.make_runtime_state_available()
    assert len(list(lifecycle._index.unique_nodes())) == 1
    owner.release_bank(other)
    owner.free(seq)
    assert lifecycle.evict(1) == 1
    assert not owner._units


def _decode_submit(owner, lifecycle, seq, step):
    assert owner.promote_to_decode(owner.bank_for(seq))
    reservations = lifecycle.prepare_decode([seq], [step])
    assert len(reservations) == 1
    reference = reservations[0].reference
    access = owner.submit_decode((owner.bank_for(seq),), checkpoints=(reference,))
    lifecycle.attach_forward(access, [seq], final_checkpoints=(reference,), decode_reservations=reservations)
    return reservations[0], access


@pytest.mark.parametrize('complete_first', [False, True])
def test_decode_replacement_is_reserved_before_old_publication_retires(complete_first):
    scheduler, owner, trie = _environment(width=1)
    lifecycle = trie.state_checkpoints
    seq = _sequence(scheduler, owner, trie, list(range(256)))
    first, access = _decode_submit(owner, lifecycle, seq, 64)
    lifecycle.publish_forward(access.access_id)
    owner.acknowledge(access.access_id)
    lifecycle.complete_forward(access.access_id)
    old_node = seq.prefix_cache.decode_checkpoint_node
    old = old_node.state_checkpoint
    old_ref = owner.bank_for(seq).input_ref
    second, next_access = _decode_submit(owner, lifecycle, seq, 128)
    assert old_node.state_checkpoint is old and old.pin_count == 1
    assert old_ref.unit_id in next_access.read_only_input_ids
    assert second.reference.unit_id not in next_access.bank_ids[0]
    if complete_first:
        owner.acknowledge(next_access.access_id, consumed_inputs=(True,))
    lifecycle.publish_forward(next_access.access_id)
    assert old_node.state_checkpoint is None and old.pin_count == 0
    new_node = seq.prefix_cache.decode_checkpoint_node
    assert new_node.state_checkpoint.step == 128
    assert new_node.state_checkpoint.exact_match_data.token_ids.tolist() == list(range(128))
    if not complete_first:
        assert old_ref.unit_id in owner._units  # The queued read, not a mutable trie node, owns it.
        owner.acknowledge(next_access.access_id, consumed_inputs=(True,))
    lifecycle.complete_forward(next_access.access_id)
    assert old_ref.unit_id not in owner._units
    assert owner.bank_for(seq).input_ref.unit_id == second.reference.unit_id
    lifecycle.release_checkpoint(new_node)
    owner.free(seq)
    assert not owner._units and not owner._accesses and not lifecycle._forwards


@pytest.mark.parametrize('action', ['cancel', 'discard', 'reader', 'no_space'])
def test_decode_replacement_failure_preserves_old_checkpoint(action):
    scheduler, owner, trie = _environment(width=1, request_slots=2, checkpoint_budget=2)
    lifecycle = trie.state_checkpoints
    seq = _sequence(scheduler, owner, trie, list(range(256)))
    first, access = _decode_submit(owner, lifecycle, seq, 64)
    lifecycle.publish_forward(access.access_id)
    owner.acknowledge(access.access_id)
    lifecycle.complete_forward(access.access_id)
    old_node = seq.prefix_cache.decode_checkpoint_node
    old = old_node.state_checkpoint
    if action == 'no_space':
        held = freeze_ready(owner, owner.bank_for(seq), owner.bank_for(seq).unit_ids[-1], 32)
        assert owner.get_num_free() == 0
        assert lifecycle.prepare_decode([seq], [128]) == ()
        assert old_node.state_checkpoint is old and old.pin_count == 0
        owner.release_reference(held)
    elif action == 'discard':
        reservations = lifecycle.prepare_decode([seq], [128])
        assert old.pin_count == 1
        lifecycle.discard_decode(reservations)
        lifecycle.discard_decode(reservations)
    else:
        second, access = _decode_submit(owner, lifecycle, seq, 128)
        if action == 'cancel':
            owner.free(seq)
        else:
            old.pin_count += 1  # Selected after reservation; publication must not evict it.
        lifecycle.publish_forward(access.access_id)
        owner.acknowledge(access.access_id, consumed_inputs=(True,))
        lifecycle.complete_forward(access.access_id)
        if action == 'reader':
            assert old.pin_count == 1
            old.pin_count -= 1
    assert old_node.state_checkpoint is old and old.pin_count == 0
    lifecycle.release_checkpoint(old_node)
    owner.free(seq)
    assert not owner._units and not owner._accesses


def test_decode_attach_preserves_already_queued_private_kv_pages():
    import numpy as np
    scheduler, owner, trie = _environment(width=1)
    canonical = _sequence(scheduler, owner, trie, list(range(128)))
    seq = scheduler.add_session(1).add_sequence(list(range(128)))
    owner.allocate(seq)
    scheduler.block_manager.allocate(seq)
    blocks = seq.logical_blocks.get_real_blocks().copy()
    canonical_blocks = canonical.logical_blocks.get_real_blocks().copy()
    assert set(blocks).isdisjoint(canonical_blocks)
    free_before = scheduler.block_manager.get_num_free_gpu_blocks()
    reservation, access = _decode_submit(owner, trie.state_checkpoints, seq, 128)
    trie.state_checkpoints.publish_forward(access.access_id)
    np.testing.assert_array_equal(seq.logical_blocks.get_real_blocks(), blocks)
    assert scheduler.block_manager.get_num_free_gpu_blocks() == free_before
    np.testing.assert_array_equal(seq.prefix_cache.decode_checkpoint_node.state_checkpoint.exact_match_data.block_ids,
                                  canonical_blocks)
    owner.acknowledge(access.access_id)
    trie.state_checkpoints.complete_forward(access.access_id)


@pytest.mark.parametrize('boundary', [128, 144])
def test_failed_producer_withdraws_descendant_publications_but_keeps_reader_holds(boundary):
    scheduler, owner, trie = _environment(width=1)
    lifecycle = trie.state_checkpoints
    tokens = list(range(256))
    producer = _sequence(scheduler, owner, trie, tokens)
    lifecycle.reserve_save(producer, step=64)
    _, old_access = _submit(owner, producer)
    lifecycle.publish_save(producer)
    owner.acknowledge(old_access.access_id)
    lifecycle.reserve_save(producer, step=boundary)
    failed_node = producer.prefix_cache.pending_save.node
    failed_checkpoint = failed_node.state_checkpoint
    failed_ref, failed = _submit(owner, producer)
    lifecycle.attach_forward(failed, [producer])
    lifecycle.publish_forward(failed.access_id)
    reader = _sequence(scheduler, owner, trie, tokens, match=True)
    assert reader.num_history_ids == boundary
    lifecycle.bind_restore(reader)
    lifecycle.reserve_save(reader, step=192)
    descendant_node = reader.prefix_cache.pending_save.node
    descendant_ref, read = _submit(owner, reader)
    lifecycle.attach_forward(read, [reader])
    lifecycle.publish_forward(read.access_id)
    assert failed_checkpoint.pin_count == 1
    free_before = owner.get_num_free()
    affected = lifecycle.fail_forward(failed.access_id)
    assert affected == (failed.access_id, read.access_id)
    assert not failed_node.state_checkpoint.published and not descendant_node.state_checkpoint.published
    assert failed_checkpoint.pin_count == 1 and failed_ref.producing
    assert owner.get_num_free() == free_before
    fallback = _sequence(scheduler, owner, trie, tokens, match=True)
    assert fallback.num_history_ids == 64
    with pytest.raises(RuntimeError, match='Failed compact work'):
        owner.acknowledge(read.access_id)
    owner.complete(failed, success=False)
    lifecycle.complete_forward(failed.access_id)
    assert failed_node.state_checkpoint is failed_checkpoint and failed_checkpoint.pin_count == 1
    owner.complete(read, success=False)
    lifecycle.complete_forward(read.access_id)
    assert failed_node.state_checkpoint is None and descendant_node.state_checkpoint is None
    assert not lifecycle._failed_nodes and not lifecycle._forwards and not lifecycle._retired_tails
    for seq in (producer, reader, fallback):
        owner.free(seq)
    lifecycle.evict(100)
    assert not owner._units and not owner._accesses


def test_lookup_rejects_completed_failed_checkpoint_even_without_prior_notification():
    scheduler, owner, trie = _environment(width=1)
    lifecycle = trie.state_checkpoints
    tokens = list(range(192))
    producer = _sequence(scheduler, owner, trie, tokens)
    lifecycle.reserve_save(producer, step=128)
    node = producer.prefix_cache.pending_save.node
    _, access = _submit(owner, producer)
    lifecycle.publish_save(producer)
    owner.complete(access, success=False)
    reader = _sequence(scheduler, owner, trie, tokens, match=True)
    assert reader.num_history_ids == 0 and not reader.prefix_cache.restore.is_selected
    assert node.state_checkpoint is None and not lifecycle._failed_nodes
    owner.free(producer)
    owner.free(reader)


def test_failed_decode_publication_uses_exact_physical_destination():
    scheduler, owner, trie = _environment(width=1)
    lifecycle = trie.state_checkpoints
    seq = _sequence(scheduler, owner, trie, list(range(256)))
    reservation, access = _decode_submit(owner, lifecycle, seq, 128)
    lifecycle.publish_forward(access.access_id)
    node = seq.prefix_cache.decode_checkpoint_node
    assert node.state_checkpoint.published
    lifecycle.fail_forward(access.access_id)
    assert not node.state_checkpoint.published and reservation.reference.producing
    lifecycle.fail_forward(access.access_id)
    owner.free(seq)
    owner.finish_cancelled_forwards_after_worker_drain()
    lifecycle.finish_cancelled_forwards_after_worker_drain()
    assert node.state_checkpoint is None and not lifecycle._failed_nodes
    assert not owner._units and not owner._accesses


def test_failed_unpublished_decode_replacement_unpins_old_only_at_actual_completion():
    scheduler, owner, trie = _environment(width=1)
    lifecycle = trie.state_checkpoints
    seq = _sequence(scheduler, owner, trie, list(range(256)))
    _, old_access = _decode_submit(owner, lifecycle, seq, 64)
    lifecycle.publish_forward(old_access.access_id)
    owner.acknowledge(old_access.access_id)
    lifecycle.complete_forward(old_access.access_id)
    old = seq.prefix_cache.decode_checkpoint_node.state_checkpoint
    reservation, access = _decode_submit(owner, lifecycle, seq, 128)
    assert old.pin_count == 1
    lifecycle.fail_forward(access.access_id)
    assert old.pin_count == 1 and old.published and reservation.reference.active
    assert reservation.reference.phase == 'failed' and reservation.reference.producing
    with pytest.raises(RuntimeError, match='actual owner completion'):
        lifecycle.complete_forward(access.access_id)
    owner.complete(access, success=False)
    lifecycle.complete_forward(access.access_id)
    assert old.pin_count == 0 and old.published and old.state_reference.phase == 'ready'
    assert not reservation.reference.active
    assert not lifecycle._forwards and not lifecycle._failed_nodes
    owner.free(seq)
    lifecycle.evict(100)
    assert not owner._units
