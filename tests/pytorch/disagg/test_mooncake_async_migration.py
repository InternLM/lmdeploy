# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

from lmdeploy.pytorch.disagg.backend.mooncake import MooncakeMigrationManagement
from lmdeploy.pytorch.disagg.conn.protocol import MigrationProtocol
from lmdeploy.pytorch.disagg.messages import AssignmentInstruct, MigrationAssignment
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
            task = asyncio.create_task(link.p2p_migrate(None))
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
