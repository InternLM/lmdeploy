# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lmdeploy.pytorch.disagg.backend.mooncake import MooncakeBackend, MooncakeMigrationManagement
from lmdeploy.pytorch.disagg.conn.protocol import MigrationProtocol
from lmdeploy.pytorch.disagg.messages import AssignmentInstruct, MigrationAssignment, PDTransferResult
from lmdeploy.pytorch.disagg.runtime import aggregate_transfer_results


def test_cancel_keeps_native_job_and_slot_until_completion():
    started, finish = Event(), Event()
    link = object.__new__(MooncakeMigrationManagement)
    link.pending_jobs = set()
    link.pending_tickets = set()

    def migrate(_):
        started.set()
        assert finish.wait(3)
        return 'completed', None

    async def run():
        link.slots = asyncio.Semaphore(1)
        with ThreadPoolExecutor(max_workers=1) as link.executor:
            link._migrate = migrate
            assignment = MigrationAssignment(protocol=MigrationProtocol.RDMA, remote_engine_id='peer', batch=[])
            task = asyncio.create_task(link.p2p_migrate(assignment))
            await asyncio.to_thread(started.wait)
            # Worker shutdown cancels all tasks, including any child Tasks.
            for pending in asyncio.all_tasks() - {asyncio.current_task()}:
                pending.cancel()
            await asyncio.sleep(0)
            assert not task.done() and link.slots.locked()
            backend = MooncakeBackend()
            backend.shared = link
            with pytest.raises(RuntimeError, match='unknown'):
                backend.close()
            finish.set()
            assert await task == 'completed'
            assert not link.pending_jobs
            assert not link.slots.locked()

    asyncio.run(run())


def test_batch_failure_and_rank_exception_do_not_claim_drain():
    link = object.__new__(MooncakeMigrationManagement)
    link.remote_url = 'peer:1234'
    link.local_kv_table = {'0': dict(addr=1000, length=100)}
    link.remote_kv_table = {'0': dict(addr=2000, length=100)}
    link.has_drain_ticket = False
    link.engine = SimpleNamespace(batch_transfer_sync_read=Mock(return_value=-1))
    assignment = MigrationAssignment(
        protocol=MigrationProtocol.RDMA, remote_engine_id='peer', transfer_id='op',
        batch=[AssignmentInstruct(mr_key=0, source_offset=10, target_offset=20, length=30)])
    result, ticket = link._migrate(assignment)
    assert ticket is None
    assert result.status == 'completion_unknown'
    link.engine.batch_transfer_sync_read.assert_called_once_with('peer:1234', [1010], [2020], [30])
    merged = aggregate_transfer_results([[result], RuntimeError('rank lost')], ['op'])
    assert merged[0].status == 'completion_unknown'


def test_expired_submission_never_waits_for_a_slot_or_calls_native():
    link = object.__new__(MooncakeMigrationManagement)
    assignment = MigrationAssignment(protocol=MigrationProtocol.RDMA, remote_engine_id='peer',
                                     transfer_id='expired', batch=[], deadline=1)
    assert link._migrate(assignment)[0].status == 'failed_drained'

    async def run():
        link.slots = asyncio.Semaphore(0)
        result = await asyncio.wait_for(link.p2p_migrate(assignment), timeout=1)
        assert result.status == 'failed_drained'
        assert link.slots.locked()

    asyncio.run(run())


def test_unknown_transfer_does_not_disable_other_requests():
    async def run():
        backend = MooncakeBackend()

        async def migrate(assignment, **kwargs):
            return PDTransferResult(transfer_id=assignment.transfer_id,
                                    status='completion_unknown' if assignment.transfer_id == 'failed' else 'completed')

        backend.links['peer'] = SimpleNamespace(p2p_migrate=migrate)
        for transfer_id, status in [('failed', 'completion_unknown'), ('healthy', 'completed')]:
            result = await backend.p2p_migrate(MigrationAssignment(
                protocol=MigrationProtocol.RDMA, remote_engine_id='peer', transfer_id=transfer_id, batch=[]))
            assert result.status == status

    asyncio.run(run())


@pytest.mark.parametrize('terminal', ['COMPLETED', 'FAILED_DRAINED'])
def test_scatter_ticket_drains_after_cancellation_without_blocking_other_transfers(terminal):
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
        link = object.__new__(MooncakeMigrationManagement)
        link.remote_url = 'peer:1234'
        link.local_kv_table = {'0': dict(addr=1000, length=100)}
        link.remote_kv_table = {'0': dict(addr=2000, length=100)}
        link.has_drain_ticket = True
        link.pending_jobs = set()
        link.pending_tickets = set()
        link.slots = asyncio.Semaphore(1)
        link.engine = SimpleNamespace(scatter_transfer_sync_read_with_ticket=Mock(side_effect=[ticket, completed]),
                                      unregister_memory=Mock(return_value=0))
        backend = MooncakeBackend()
        backend.links['peer'] = backend.shared = link
        assignment = MigrationAssignment(
            protocol=MigrationProtocol.RDMA, remote_engine_id='peer', transfer_id='late',
            batch=[AssignmentInstruct(mr_key=0, source_offset=10, target_offset=20, length=30)])
        with ThreadPoolExecutor(max_workers=1) as link.executor:
            job = asyncio.create_task(backend.p2p_migrate(assignment))
            while link.slots.locked() or not link.pending_tickets:
                await asyncio.sleep(0.01)
            job.cancel()
            healthy = assignment.model_copy(update=dict(transfer_id='healthy'))
            assert (await asyncio.wait_for(backend.p2p_migrate(healthy), timeout=1)).status == 'completed'
            with pytest.raises(RuntimeError, match='unknown'):
                backend.close()
            assert not job.done()
            ready.set()
            result = await asyncio.wait_for(job, timeout=1)
            assert result.status == terminal.lower()
            assert not link.pending_tickets and not backend.has_unknown_completion
            assert link.engine.scatter_transfer_sync_read_with_ticket.call_count == 2
            link.engine.scatter_transfer_sync_read_with_ticket.assert_called_with(
                'peer:1234', [1000], [100], [2000], [100], [[10]], [[20]], [[30]])
            backend.close()
            link.engine.unregister_memory.assert_called_once_with(1000)

    asyncio.run(run())
