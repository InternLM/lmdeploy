# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

from lmdeploy.pytorch.disagg.backend.mooncake import MooncakeBackend, MooncakeMigrationManagement
from lmdeploy.pytorch.disagg.conn.protocol import MigrationProtocol
from lmdeploy.pytorch.disagg.messages import AssignmentInstruct, MigrationAssignment, PDTransferResult
from lmdeploy.pytorch.disagg.runtime import aggregate_transfer_results


def test_cancel_keeps_native_job_and_slot_until_completion():
    started, finish = Event(), Event()
    link = object.__new__(MooncakeMigrationManagement)

    def migrate(_):
        started.set()
        assert finish.wait(3)
        return 'completed'

    async def run():
        link.slots = asyncio.Semaphore(1)
        with ThreadPoolExecutor(max_workers=1) as link.executor:
            link._migrate = migrate
            assignment = MigrationAssignment(protocol=MigrationProtocol.RDMA, remote_engine_id='peer', batch=[])
            task = asyncio.create_task(link.p2p_migrate(assignment))
            await asyncio.to_thread(started.wait)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done() and link.slots.locked()
            finish.set()
            assert await task == 'completed'
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
    result = link._migrate(assignment)
    assert result.status == 'completion_unknown'
    link.engine.batch_transfer_sync_read.assert_called_once_with('peer:1234', [1010], [2020], [30])
    merged = aggregate_transfer_results([[result], RuntimeError('rank lost')], ['op'])
    assert merged[0].status == 'completion_unknown'


def test_expired_submission_never_waits_for_a_slot_or_calls_native():
    link = object.__new__(MooncakeMigrationManagement)
    assignment = MigrationAssignment(protocol=MigrationProtocol.RDMA, remote_engine_id='peer',
                                     transfer_id='expired', batch=[], deadline=1)
    assert link._migrate(assignment).status == 'failed_drained'

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
