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

    def create(*, apc=False, width=1, requests=1, checkpoint_budget=None):
        capacity = SSMStateCapacity(requests + 1, width,
                                        (2 if apc else 0) if checkpoint_budget is None else checkpoint_budget)
        config = CacheConfig(
            max_batches=requests, block_size=16, num_cpu_blocks=0, num_gpu_blocks=64,
            states_shapes=[((2,), torch.float32)], num_state_caches=requests + 1 + capacity.checkpoint_budget,
            prefix_cache_state_budget=capacity.checkpoint_budget, enable_prefix_caching=apc,
            prefill_save_alignment=64, compact_state_capacity=capacity)
        config.kv_transfer_config = KVTransferConfig(kv_connector='MooncakeStoreConnector', kv_role='kv_consumer')
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
def test_failure_or_cancellation_waits_for_all_rank_terminal(setup_load, action, apc):
    paging, store = setup_load(apc=apc)
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
