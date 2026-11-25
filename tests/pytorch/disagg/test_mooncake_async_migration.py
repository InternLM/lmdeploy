# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import json
import sys
from threading import Event
from time import time
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call

import pytest

from lmdeploy.pytorch.disagg.backend import mooncake
from lmdeploy.pytorch.disagg.backend.mooncake import MooncakeBackend
from lmdeploy.pytorch.disagg.config import MooncakeEngineConfig
from lmdeploy.pytorch.disagg.conn.protocol import (
    DistServeInitRequest,
    DistServeKVTransferEndpointInfo,
    MigrationProtocol,
)
from lmdeploy.pytorch.disagg.messages import AssignmentInstruct, DistServeRegisterMRMessage, MigrationAssignment
from lmdeploy.pytorch.disagg.runtime import aggregate_transfer_results


@pytest.fixture
def transfer_engine(monkeypatch, request):
    monkeypatch.delenv('LMDEPLOY_PD_TRANSFER_THREADS', raising=False)
    engine = SimpleNamespace(initialize=Mock(return_value=0), get_rpc_port=Mock(return_value=1234),
                             register_memory=Mock(return_value=0), unregister_memory=Mock(return_value=0),
                             batch_transfer_sync_read=Mock(return_value=0))
    if getattr(request, 'param', 'batch') == 'scatter':
        engine.scatter_transfer_sync_read_with_ticket = Mock()
    factory = Mock(return_value=engine)
    engine_module = ModuleType('mooncake.engine')
    engine_module.TransferEngine = factory
    mooncake_module = ModuleType('mooncake')
    mooncake_module.engine = engine_module
    monkeypatch.setitem(sys.modules, 'mooncake', mooncake_module)
    monkeypatch.setitem(sys.modules, 'mooncake.engine', engine_module)
    monkeypatch.setattr(mooncake, 'get_local_ip_by_remote', Mock(return_value='127.0.0.1'))
    monkeypatch.setattr(mooncake, 'get_rdma_nics', Mock(return_value=['rdma0', 'rdma1']))
    return factory


@pytest.fixture
def init_request():
    config = MooncakeEngineConfig(tp_size=2, ep_size=1, dp_size=1, pp_size=1, dp_rank=0,
                                  block_size=16, num_cpu_blocks=0, num_gpu_blocks=100)
    return DistServeInitRequest(local_engine_id='local', local_engine_config=config,
                               remote_engine_id='peer', remote_engine_config=config,
                               protocol=MigrationProtocol.RDMA, rank=1)


@pytest.fixture
def backend(transfer_engine, init_request, monkeypatch, request):
    if getattr(request, 'param', None) is not None:
        monkeypatch.setenv('LMDEPLOY_PD_TRANSFER_THREADS', request.param)
    backend = MooncakeBackend()
    for peer, addr in [('peer', 2000), ('other', 3000)]:
        backend.p2p_initialize(init_request.model_copy(update=dict(remote_engine_id=peer)))
        backend.register_memory_region(DistServeRegisterMRMessage(
            protocol=MigrationProtocol.RDMA, remote_engine_id=peer, mr_key=0, addr=1000, length=100, offset=0))
        backend.p2p_connect(peer, DistServeKVTransferEndpointInfo(
            protocol=MigrationProtocol.RDMA, endpoint_info=json.dumps(
                dict(session_id=f'{peer}:1234', mr_info={'0': dict(addr=addr, length=100, offset=0)}))))
    with backend.resources.executor:
        yield backend


@pytest.fixture
def assignment():
    return MigrationAssignment(
        protocol=MigrationProtocol.RDMA, remote_engine_id='peer', transfer_id='op',
        batch=[AssignmentInstruct(mr_key=0, source_offset=10, target_offset=20, length=30)])


def test_peers_share_one_engine_registration_and_local_endpoint(backend, transfer_engine):
    transfer_engine.assert_called_once_with()
    engine = transfer_engine.return_value
    engine.initialize.assert_called_once_with('127.0.0.1', 'P2PHANDSHAKE', 'rdma', 'rdma1')
    engine.register_memory.assert_called_once_with(1000, 100)
    mooncake.get_local_ip_by_remote.assert_called_once_with()
    mooncake.get_rdma_nics.assert_called_once_with()
    assert backend.links['peer'] is not backend.links['other']
    assert backend.links['peer'].resources is backend.links['other'].resources is backend.resources
    endpoint = dict(session_id='127.0.0.1:1234', mr_info={'0': dict(addr=1000, length=100, offset=0)})
    for peer in backend.links:
        assert backend.endpoint_info(peer, MigrationProtocol.RDMA) == endpoint


@pytest.mark.parametrize('change', [dict(addr=4000), dict(length=200), dict(offset=10)])
def test_conflicting_registration_cannot_replace_local_allocation(backend, change):
    fields = dict(protocol=MigrationProtocol.RDMA, remote_engine_id='other', mr_key=0,
                  addr=1000, length=100, offset=0)
    with pytest.raises(ValueError, match='Cannot replace'):
        backend.register_memory_region(DistServeRegisterMRMessage(**(fields | change)))
    assert backend.resources.local_kv_table == {'0': dict(addr=1000, length=100, offset=0)}
    backend.resources.engine.register_memory.assert_called_once_with(1000, 100)


def test_failed_initialization_does_not_publish_resources_or_links(transfer_engine, init_request):
    engine = transfer_engine.return_value
    engine.initialize.side_effect = [-1, 0]
    backend = MooncakeBackend()
    with pytest.raises(RuntimeError, match='Failed to initialize'):
        backend.p2p_initialize(init_request)
    assert backend.resources is None and not backend.links
    backend.p2p_initialize(init_request)
    try:
        assert backend.links['peer'].resources is backend.resources
        assert transfer_engine.call_count == 2
    finally:
        backend.close()


@pytest.mark.parametrize('value', ['0', '-1', 'invalid', ''])
def test_invalid_thread_count_does_not_initialize_engine(transfer_engine, init_request, monkeypatch, value):
    monkeypatch.setenv('LMDEPLOY_PD_TRANSFER_THREADS', value)
    backend = MooncakeBackend()
    with pytest.raises(ValueError, match='LMDEPLOY_PD_TRANSFER_THREADS must be a positive integer'):
        backend.p2p_initialize(init_request)
    transfer_engine.assert_not_called()
    assert backend.resources is None and not backend.links


def test_close_unregisters_shared_allocation_once(backend):
    resources = backend.resources
    backend.close()
    backend.close()
    resources.engine.unregister_memory.assert_called_once_with(1000)
    assert not resources.local_kv_table
    assert backend.resources is None and not backend.links


def test_failed_unregistration_preserves_resources_and_links(backend):
    resources = backend.resources
    resources.engine.unregister_memory.side_effect = [-1, 0]
    with pytest.raises(RuntimeError, match='Failed to unregister'):
        backend.close()
    assert backend.resources is resources and len(backend.links) == 2
    assert resources.local_kv_table == {'0': dict(addr=1000, length=100, offset=0)}
    backend.close()
    assert backend.resources is None and not backend.links


def test_peers_read_from_their_own_remote_endpoints(backend, assignment):
    async def run():
        for peer in backend.links:
            result = await backend.p2p_migrate(assignment.model_copy(update=dict(remote_engine_id=peer)))
            assert result.status == 'completed'

    asyncio.run(run())
    assert backend.resources.engine.batch_transfer_sync_read.call_args_list == [
        call('peer:1234', [1010], [2020], [30]), call('other:1234', [1010], [3020], [30])]


@pytest.mark.parametrize(('backend', 'num_threads'), [(None, 4), ('1', 1), ('8', 8)], indirect=['backend'])
def test_submission_slots_are_shared_across_peers(backend, assignment, num_threads):
    started = {1010 + i: Event() for i in range(num_threads)}
    finish = Event()
    resources = backend.resources

    def read(_peer, local_addresses, _remote_addresses, _lengths):
        started[local_addresses[0]].set()
        assert finish.wait(5)
        return 0

    resources.engine.batch_transfer_sync_read.side_effect = read

    async def run():
        assignments = [assignment.model_copy(update=dict(batch=[
            AssignmentInstruct(mr_key=0, source_offset=10 + i, target_offset=20 + i, length=1)
        ])) for i in range(num_threads)]
        jobs = [asyncio.create_task(backend.p2p_migrate(item)) for item in assignments]
        try:
            for event in started.values():
                assert await asyncio.to_thread(event.wait, 3)
            assert resources.slots.locked()
            waiting = assignment.model_copy(update=dict(remote_engine_id='other', deadline=time() + 0.05))
            result = await asyncio.wait_for(backend.p2p_migrate(waiting), timeout=1)
            assert result.status == 'failed_drained'
            assert resources.engine.batch_transfer_sync_read.call_count == num_threads
        finally:
            finish.set()
            results = await asyncio.gather(*jobs)
        assert all(result.status == 'completed' for result in results)
        assert not resources.pending_jobs and not resources.slots.locked()

    asyncio.run(run())


def test_cancel_keeps_native_job_and_slot_until_completion(backend, assignment):
    started, finish = Event(), Event()
    resources = backend.resources
    resources.slots = asyncio.Semaphore(1)

    def read(*_):
        started.set()
        assert finish.wait(5)
        return 0

    resources.engine.batch_transfer_sync_read.side_effect = read

    async def run():
        task = asyncio.create_task(backend.p2p_migrate(assignment))
        try:
            assert await asyncio.to_thread(started.wait, 3)
            # Worker shutdown cancels all tasks, including any child Tasks.
            for pending in asyncio.all_tasks() - {asyncio.current_task()}:
                pending.cancel()
            await asyncio.sleep(0)
            assert not task.done() and resources.slots.locked()
            with pytest.raises(RuntimeError, match='unknown'):
                backend.close()
            resources.engine.unregister_memory.assert_not_called()
        finally:
            finish.set()
            result = await task
        assert result.status == 'completed'
        assert not resources.pending_jobs and not resources.slots.locked()
        backend.close()
        resources.engine.unregister_memory.assert_called_once_with(1000)

    asyncio.run(run())


def test_batch_failure_and_rank_exception_do_not_claim_drain(backend, assignment):
    engine = backend.resources.engine
    engine.batch_transfer_sync_read.return_value = -1
    result, ticket = backend.links['peer']._migrate(assignment)
    assert ticket is None
    assert result.status == 'completion_unknown'
    engine.batch_transfer_sync_read.assert_called_once_with('peer:1234', [1010], [2020], [30])
    merged = aggregate_transfer_results([[result], RuntimeError('rank lost')], ['op'])
    assert merged[0].status == 'completion_unknown'


def test_expired_submission_never_waits_for_a_slot_or_calls_native(backend, assignment):
    assignment = assignment.model_copy(update=dict(deadline=1))
    assert backend.links['peer']._migrate(assignment)[0].status == 'failed_drained'
    resources = backend.resources

    async def run():
        resources.slots = asyncio.Semaphore(0)
        result = await asyncio.wait_for(backend.p2p_migrate(assignment), timeout=1)
        assert result.status == 'failed_drained'
        assert resources.slots.locked()

    asyncio.run(run())
    resources.engine.batch_transfer_sync_read.assert_not_called()


@pytest.mark.parametrize('failure', [-1, RuntimeError('native failure')])
def test_unknown_transfer_does_not_disable_other_requests(backend, assignment, failure):
    resources = backend.resources
    resources.engine.batch_transfer_sync_read.side_effect = [failure, 0]

    async def run():
        result = await backend.p2p_migrate(assignment)
        assert result.status == 'completion_unknown'
        healthy = assignment.model_copy(update=dict(remote_engine_id='other', transfer_id='healthy'))
        assert (await backend.p2p_migrate(healthy)).status == 'completed'
        assert resources.has_unknown_completion
        with pytest.raises(RuntimeError, match='unknown'):
            backend.close()
        resources.engine.unregister_memory.assert_not_called()
        assert backend.resources is resources and len(backend.links) == 2

    asyncio.run(run())


@pytest.mark.parametrize('transfer_engine', ['scatter'], indirect=True)
@pytest.mark.parametrize('terminal', ['COMPLETED', 'FAILED_DRAINED'])
def test_scatter_ticket_drains_after_cancellation_without_blocking_other_transfers(backend, assignment, terminal):
    async def run():
        ready = Event()

        class Ticket:
            drained = False
            status = SimpleNamespace(name='COMPLETION_UNKNOWN')

        ticket = Ticket()
        completed = Ticket()
        completed.drained = True
        completed.status = SimpleNamespace(name='COMPLETED')

        def drain(timeout_ms):
            assert timeout_ms == 0
            if ready.is_set():
                ticket.drained = True
                ticket.status.name = terminal

        ticket.drain = drain
        resources = backend.resources
        resources.slots = asyncio.Semaphore(1)
        resources.engine.scatter_transfer_sync_read_with_ticket.side_effect = [ticket, completed]
        job = asyncio.create_task(backend.p2p_migrate(assignment))

        async def wait_for_ticket():
            while resources.slots.locked() or not resources.pending_tickets:
                await asyncio.sleep(0.01)

        try:
            await asyncio.wait_for(wait_for_ticket(), timeout=3)
            job.cancel()
            healthy = assignment.model_copy(update=dict(remote_engine_id='other', transfer_id='healthy'))
            assert (await asyncio.wait_for(backend.p2p_migrate(healthy), timeout=1)).status == 'completed'
            with pytest.raises(RuntimeError, match='unknown'):
                backend.close()
            resources.engine.unregister_memory.assert_not_called()
            assert not job.done()
        finally:
            ready.set()
            result = await asyncio.wait_for(job, timeout=3)
        assert result.status == terminal.lower()
        assert not resources.pending_tickets and not resources.has_unknown_completion
        assert resources.engine.scatter_transfer_sync_read_with_ticket.call_args_list == [
            call('peer:1234', [1000], [100], [2000], [100], [[10]], [[20]], [[30]]),
            call('other:1234', [1000], [100], [3000], [100], [[10]], [[20]], [[30]])]
        backend.close()
        resources.engine.unregister_memory.assert_called_once_with(1000)

    asyncio.run(run())
