# Copyright (c) OpenMMLab. All rights reserved.
"""Direct-load ownership through the real paging/Store all-rank lifecycle.

The real Scheduler composition root binds its single owner to the connector; lookup/completion are controlled here, not
real network I/O.
"""
from unittest.mock import Mock

import pytest
import torch

from lmdeploy.messages import KVTransferConfig
from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig, SSMStateCapacity
from lmdeploy.pytorch.kv_connector.base import KVConnectorOutput, KVConnectorOutputAggregator, KVConnectorRole
from lmdeploy.pytorch.kv_connector.mooncake.store.connector import MooncakeStoreConnector
from lmdeploy.pytorch.messages import MessageStatus, SequenceMeta
from lmdeploy.pytorch.paging.compact_state import SSMStateManager
from lmdeploy.pytorch.paging.scheduler import Scheduler
from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy


@pytest.fixture
def setup_load():
    schedulers = []

    def create(*, apc=False, width=1, requests=1, checkpoint_budget=None, auxiliary=False, role='kv_consumer'):
        store_budget = 2 if role == 'kv_both' else 0
        capacity = SSMStateCapacity(requests + 1, width,
                                        (2 if apc else 0) if checkpoint_budget is None else checkpoint_budget,
                                        store_budget)
        config = CacheConfig(
            max_batches=requests, block_size=16, num_cpu_blocks=0, num_gpu_blocks=64,
            states_shapes=[((2,), torch.float32)],
            num_state_caches=requests + 1 + capacity.checkpoint_budget + store_budget,
            mooncake_state_save_slots=2,
            prefix_cache_state_budget=capacity.checkpoint_budget, enable_prefix_caching=apc,
            prefill_save_alignment=64, compact_state_capacity=capacity)
        config.kv_transfer_config = KVTransferConfig(kv_connector='MooncakeStoreConnector', kv_role=role)
        config.enable_replayssm = auxiliary
        config.compact_state_capacity = SSMStateCapacity.from_cache_config(
            config, width, runtime_is_auxiliary=auxiliary)
        connector = MooncakeStoreConnector(KVConnectorRole.SCHEDULER, config)
        paging = Scheduler(SchedulerConfig(max_batches=requests, max_session_len=4096), config,
                           SequenceMeta(16, strategy=ARSequenceStrategy()), kv_connector=connector)
        store = connector.connector_scheduler
        store.client.lookup = Mock(return_value=64)
        schedulers.append(paging)
        return paging, store

    yield create
    for paging in schedulers:
        for session_id in tuple(paging.sessions):
            paging.end_session(session_id)
        # Controlled tests have no actual worker or I/O. This is the explicit
        # quiescence proof for their intentionally undelivered metadata.
        paging.reset_cache_after_worker_drain()
        paging.shutdown()


def _start(paging):
    seq = paging.add_session(0).add_sequence(torch.arange(73))
    assert paging.schedule(is_prefill=True).running == []
    assert seq.status == MessageStatus.WAITING_FOR_REMOTE_KVS
    return seq


@pytest.mark.parametrize('apc', [False, True])
@pytest.mark.parametrize('width', [1, 4, 6])
def test_all_rank_ready_binds_existing_private_unit_not_runtime_slot(setup_load, apc, width):
    paging, store = setup_load(apc=apc, width=width, requests=2, checkpoint_budget=2)
    owner = paging.state_manager
    # Make auxiliary and physical IDs differ; slot=2 is not the loaded unit.
    other = owner.allocate_bank()
    frozen = owner.reserve_checkpoint(other, 64)
    seq = _start(paging)
    bank = owner.bank_for(seq)
    metadata = paging.build_connector_meta([])
    load = metadata.load_requests[0]
    target = bank.unit_ids[0]
    assert load.state_unit_id == target != seq.logical_state
    assert load.state_slot is None and metadata.get_state_save_copies() == ()
    assert len(owner._units) == owner.capacity.bank_width + 2
    assert store._state_slots is None and owner._store_refs == 0
    aggregator = KVConnectorOutputAggregator(4)
    for rank in range(4):
        outputs = [None] * 4
        outputs[rank] = KVConnectorOutput(finished_receiving={seq.seq_id})
        paging.update_connector_output(aggregator.aggregate(outputs))
        if rank != 3:
            assert bank.loaded_input is None and bank.pending_load is not None
            assert seq.status == MessageStatus.WAITING_FOR_REMOTE_KVS
            with pytest.raises(ValueError, match='Store I/O'):
                owner.submit_decode((bank,))
    assert seq.num_history_ids == 64
    assert bank.loaded_input == (target, 64) and bank.pending_load is None
    assert not seq.prefix_cache.restore.is_selected
    assert paging.schedule(is_prefill=True).running == [seq]
    assert seq.logical_state == bank.auxiliary_id
    assert not store._compact_loads and not owner._accesses
    # Duplicate completion cannot change/bind the recycled bank.
    paging.update_connector_output(KVConnectorOutput(finished_receiving={seq.seq_id}))
    owner.cancel_checkpoint(frozen)
    owner.release_bank(other)


@pytest.mark.parametrize('action', ['failure', 'stop', 'end'])
@pytest.mark.parametrize('apc', [False, True])
@pytest.mark.parametrize('auxiliary', [False, True])
@pytest.mark.parametrize('role', ['kv_consumer', 'kv_both'])
def test_failure_or_cancellation_waits_for_all_rank_terminal(setup_load, action, apc, auxiliary, role):
    paging, store = setup_load(apc=apc, auxiliary=auxiliary, role=role)
    seq = _start(paging)
    load = paging.build_connector_meta([]).load_requests[0]
    owner, aggregator = paging.state_manager, KVConnectorOutputAggregator(2)
    bank = owner.bank_for(seq)
    if action == 'stop':
        paging.stop_session(0)
    elif action == 'end':
        paging.end_session(0)
    first = KVConnectorOutput(finished_receiving={seq.seq_id},
                              failed_receiving={seq.seq_id} if action == 'failure' else set())
    paging.update_connector_output(aggregator.aggregate([first, None]))
    assert bank.pending_load is not None and bank.loaded_input is None
    assert load.state_unit_id in owner._units
    assert owner.get_num_free_runtime_slots() == 0
    assert seq.logical_state > 0 and seq.num_blocks > 0
    paging.update_connector_output(aggregator.aggregate([None, KVConnectorOutput(finished_receiving={seq.seq_id})]))
    assert not owner._units and not owner._banks and not owner._accesses
    assert not store._compact_loads
    assert seq.logical_state == -1 and seq.num_history_ids == 0
    if action == 'end':
        assert not paging.sessions
    elif action == 'stop':
        assert seq.status == MessageStatus.STOPPED
    else:
        assert paging.schedule(is_prefill=True).running == [seq]
        assert seq.num_history_ids == 0  # no retry into a partially loaded input
        assert store.client.lookup.call_count == 1


@pytest.mark.parametrize('dispatched', [False, True])
def test_compact_shutdown_requires_drain_then_retires_deferred_session(setup_load, dispatched):
    paging, store = setup_load(apc=True)
    seq = _start(paging)
    if dispatched:
        paging.build_connector_meta([])
    bank = paging.state_manager.bank_for(seq)
    paging.end_session(0)
    with pytest.raises(RuntimeError, match='actual I/O completion'):
        paging.shutdown()
    assert paging.kv_connector is not None
    assert seq.seq_id in paging.kv_load_coordinator._loads
    assert bank.pending_load is not None
    paging.reset_cache_after_worker_drain()
    assert not paging.sessions
    assert not paging.state_manager._units and not paging.state_manager._accesses
    assert not store._compact_loads and not store._pending_loads and not store._inflight_loads
    paging.reset_cache_after_worker_drain()


@pytest.mark.parametrize('dispatched', [False, True])
def test_premature_request_finish_cannot_drop_inflight_destination(setup_load, dispatched):
    paging, store = setup_load()
    seq = _start(paging)
    bank = paging.state_manager.bank_for(seq)
    if dispatched:
        paging.build_connector_meta([])
    store.request_finished(seq)  # intentionally violate coordinator deferral
    paging.state_manager.free(seq)
    if dispatched:
        assert bank.pending_load is not None
        assert seq.seq_id in store._inflight_loads
        result = store.update_connector_output(KVConnectorOutput(finished_receiving={seq.seq_id}))
        assert not result.load_results[0].success
    assert not store._compact_loads and not paging.state_manager._units


def test_compact_owner_binding_is_single_and_capacity_checked(setup_load):
    paging, store = setup_load()
    with pytest.raises(ValueError, match='exactly once'):
        store.bind_compact_state_manager(paging.state_manager)
    other = SSMStateManager(SSMStateCapacity(3, 4))
    with pytest.raises(ValueError, match='capacity'):
        store.bind_compact_state_manager(other)


@pytest.mark.parametrize('role', ['kv_producer', 'kv_both'])
def test_compact_producer_never_constructs_legacy_snapshot_allocator(role):
    from lmdeploy.pytorch.kv_connector.base import KVConnectorStepInput

    capacity = SSMStateCapacity(3, 1, 0, 2)
    config = CacheConfig(max_batches=2, block_size=16, num_cpu_blocks=0, num_gpu_blocks=16,
                         states_shapes=[((2,), torch.float32)], num_state_caches=5,
                         mooncake_state_save_slots=2, prefill_save_alignment=64,
                         compact_state_capacity=capacity,
                         kv_transfer_config=KVTransferConfig(kv_connector='MooncakeStoreConnector', kv_role=role))
    connector = MooncakeStoreConnector(KVConnectorRole.SCHEDULER, config)
    owner = SSMStateManager(capacity)
    connector.bind_compact_state_manager(owner)
    store = connector.connector_scheduler
    assert store._state_slots is None
    with pytest.raises(ValueError, match='producer reference'):
        connector.build_connector_meta(KVConnectorStepInput(running=[], connector_token_lens=(64,)))
    assert not store._inflight_saves and not owner._units
    connector.shutdown()


@pytest.mark.parametrize('width', [1, 6])
@pytest.mark.parametrize('apc', [False, True])
def test_auxiliary_replay_store_load_uses_shared_unit_and_all_rank_completion(setup_load, width, apc):
    paging, store = setup_load(apc=apc, width=width, auxiliary=True)
    owner = paging.state_manager
    assert owner.capacity.bank_width == 0 and owner.capacity.load_budget == 1
    seq = _start(paging)
    bank = owner.bank_for(seq)
    load = paging.build_connector_meta([]).load_requests[0]
    assert load.state_unit_id > 0 and load.state_slot is None and bank.unit_ids == ()
    aggregator = KVConnectorOutputAggregator(2)
    paging.update_connector_output(aggregator.aggregate([
        KVConnectorOutput(finished_receiving={seq.seq_id}), None]))
    assert bank.loaded_input is None and bank.pending_load is not None
    paging.update_connector_output(aggregator.aggregate([
        None, KVConnectorOutput(finished_receiving={seq.seq_id})]))
    assert bank.loaded_input == (load.state_unit_id, 64)
    assert bank.input_ref.unit_id == load.state_unit_id
    assert paging.schedule(is_prefill=True).running == [seq]
    access = owner.submit_prefill((bank,))
    owner.consume_loaded_inputs(access)
    owner.acknowledge(access.access_id, consumed_inputs=(True,))
    assert bank.input_ref is None and load.state_unit_id not in owner._units


def _fill_prefix_checkpoints(paging):
    """Publish through the real trie until Replay can borrow no more units."""
    owner = paging.state_manager
    trie = paging.block_trie
    lifecycle = trie.state_checkpoints
    produced = []
    for i in range(owner.capacity.num_units - 1):
        seq = paging.add_session(100 + i).add_sequence(torch.arange(73) + 100 * (i + 1))
        assert lifecycle.make_runtime_state_available()
        owner.allocate(seq)
        paging.block_manager.allocate(seq)
        trie.allocate(seq)
        assert lifecycle.reserve_save(seq, step=64) > 0
        node = seq.prefix_cache.pending_save.node
        reference = node.state_checkpoint.state_reference
        access = owner.submit_prefill((owner.bank_for(seq),), checkpoints=(reference,))
        assert lifecycle.publish_save(seq)
        owner.acknowledge(access.access_id)
        paging.end_session(100 + i)
        produced.append((seq, node))
    if owner.capacity.runtime_is_auxiliary:
        assert owner.get_num_free_optional() == 0 and not owner._accesses
    return [(seq, node) for seq, node in produced if node.state_checkpoint is not None]


@pytest.mark.parametrize('auxiliary', [False, True])
@pytest.mark.parametrize('role', ['kv_consumer', 'kv_both'])
@pytest.mark.parametrize('width', [1, 6])
def test_remote_load_reclaims_checkpoint_pressure(setup_load, auxiliary, role, width):
    paging, store = setup_load(apc=True, checkpoint_budget=1, auxiliary=auxiliary, role=role, width=width)
    nodes = _fill_prefix_checkpoints(paging)
    seq = _start(paging)
    owner = paging.state_manager
    bank = owner.bank_for(seq)
    load = paging.build_connector_meta([]).load_requests[0]
    assert bank.pending_load is not None and load.state_unit_id in owner._units
    if auxiliary:
        assert sum(node.state_checkpoint is not None for _, node in nodes) == len(nodes) - 1
    paging.update_connector_output(KVConnectorOutput(finished_receiving={seq.seq_id}))
    assert seq.num_history_ids == 64 and bank.loaded_input == (load.state_unit_id, 64)
    assert paging.schedule(is_prefill=True).running == [seq]
    assert store.client.lookup.call_count == 1


@pytest.mark.parametrize('hold', ['restore_pin', 'reader_reference'])
@pytest.mark.parametrize('role', ['kv_consumer', 'kv_both'])
def test_load_waits_without_leaks_and_retries_after_holds_release(setup_load, hold, role):
    paging, store = setup_load(apc=True, checkpoint_budget=1, auxiliary=True, role=role)
    owner, lifecycle = paging.state_manager, paging.block_trie.state_checkpoints
    nodes = _fill_prefix_checkpoints(paging)
    readers = []
    for seq, node in nodes:
        if hold == 'restore_pin':
            # Detached sequence handles exercise real restore pin/unpin, without
            # adding unrelated readers to this scheduling queue.
            seq.prefix_cache.restore.select(node.state_checkpoint.slot, node)
            assert lifecycle.pin_restore(seq)
        else:
            readers.append(owner.retain(node.state_checkpoint.state_reference, 'request'))
    seq = paging.add_session(0).add_sequence(torch.arange(73))
    seq.kv_token_limit = 48
    blocks = paging.block_manager.get_num_free_gpu_blocks()
    try:
        for _ in range(2):
            assert paging.schedule(is_prefill=True).running == []
            assert seq.status == MessageStatus.WAITING and seq.logical_state == -1
            assert seq.num_blocks == 0 and seq.num_history_ids == 0
            assert seq.kv_token_limit == 48
            assert paging.block_manager.get_num_free_gpu_blocks() == blocks
            assert not owner._banks and not owner._accesses
            assert seq.seq_id in store._lookup_plans
            assert not store._pending_loads and not store._compact_loads
            assert not paging.kv_load_coordinator._loads and not paging.kv_load_coordinator._prefill_targets
            assert paging.kv_load_coordinator.prefix_cache_stats.num_hit_tokens == 0
        if hold == 'restore_pin':
            assert all(node.state_checkpoint is not None for _, node in nodes)
        else:
            # Logical eviction is insufficient while another reader holds bytes.
            assert all(node.state_checkpoint is None for _, node in nodes)
        # A cold local request needs no shared load destination: don't block it.
        store.client.lookup.return_value = 0
        local = paging.add_session(1).add_sequence(torch.arange(73) + 5000)
        assert paging.schedule(is_prefill=True).running == [local]
        paging.end_session(1)
    finally:
        if hold == 'restore_pin':
            for reader, _ in nodes:
                lifecycle.unpin_restore(reader)
        else:
            for reference in readers:
                owner.release_reference(reference)
    assert paging.schedule(is_prefill=True).running == []
    assert seq.status == MessageStatus.WAITING_FOR_REMOTE_KVS
    assert store.client.lookup.call_count == 2  # One original lookup plus local miss, no remote retry.


@pytest.mark.parametrize('role', ['kv_consumer', 'kv_both'])
def test_load_binding_error_rolls_back_destinations_without_hiding_error(setup_load, monkeypatch, role):
    paging, store = setup_load(apc=True, checkpoint_budget=1, auxiliary=True, role=role)
    _fill_prefix_checkpoints(paging)
    owner = paging.state_manager
    seq = paging.add_session(0).add_sequence(torch.arange(73))
    blocks = paging.block_manager.get_num_free_gpu_blocks()
    original = paging.kv_connector.update_state_after_alloc

    def invalid_binding(request, block_ids, num_external_tokens):
        # Actual connector validation fails before it leases shared storage.
        return original(request, (), num_external_tokens)

    with monkeypatch.context() as patch:
        patch.setattr(paging.kv_connector, 'update_state_after_alloc', invalid_binding)
        with pytest.raises(ValueError, match='allocated load blocks'):
            paging.schedule(is_prefill=True)
    assert seq.logical_state == -1 and seq.num_blocks == 0 and seq.num_history_ids == 0
    assert paging.block_manager.get_num_free_gpu_blocks() == blocks
    assert not owner._banks and not owner._accesses
    assert not store._pending_loads and not store._compact_loads
    assert seq.seq_id in store._lookup_plans
    assert paging.schedule(is_prefill=True).running == []
    assert seq.status == MessageStatus.WAITING_FOR_REMOTE_KVS


@pytest.mark.parametrize('role', ['kv_consumer', 'kv_both'])
@pytest.mark.parametrize('action', ['failure', 'stop', 'end'])
def test_pressure_load_failure_or_cancel_keeps_lease_until_all_ranks(setup_load, role, action):
    paging, store = setup_load(apc=True, width=6, checkpoint_budget=1, auxiliary=True, role=role)
    _fill_prefix_checkpoints(paging)
    owner = paging.state_manager
    seq = _start(paging)
    bank = owner.bank_for(seq)
    load = paging.build_connector_meta([]).load_requests[0]
    if action == 'stop':
        paging.stop_session(0)
    elif action == 'end':
        paging.end_session(0)
    aggregator = KVConnectorOutputAggregator(2)
    first = KVConnectorOutput(finished_receiving={seq.seq_id},
                              failed_receiving={seq.seq_id} if action == 'failure' else set())
    paging.update_connector_output(aggregator.aggregate([first, None]))
    assert bank.pending_load is not None and load.state_unit_id in owner._units
    assert owner.get_num_free_runtime_slots() == 0 and owner.get_num_free() == 0
    assert seq.logical_state > 0 and seq.num_blocks > 0
    paging.update_connector_output(aggregator.aggregate([None, KVConnectorOutput(finished_receiving={seq.seq_id})]))
    assert load.state_unit_id not in owner._units and not owner._banks and not owner._accesses
    assert owner.get_num_free() == 1 and not store._compact_loads
    assert seq.logical_state == -1 and seq.num_history_ids == 0
    if action == 'end':
        assert not paging.sessions
    elif action == 'stop':
        assert seq.status == MessageStatus.STOPPED
    else:
        assert paging.schedule(is_prefill=True).running == [seq]
        assert store.client.lookup.call_count == 1
