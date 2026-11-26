# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from lmdeploy.messages import ResponseType
from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.pytorch.disagg.conn.protocol import MigrationProtocol, MigrationRequest, PDControlRequest
from lmdeploy.pytorch.disagg.messages import PDTransferResult
from lmdeploy.pytorch.engine.engine_loop import EngineLoop, EngineLoopConfig
from lmdeploy.pytorch.messages import SequenceMeta, UpdateTokenMode
from lmdeploy.pytorch.paging.scheduler import Scheduler
from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy


def make_transfer(migrate, logprobs=None):
    def scheduler():
        return Scheduler(SchedulerConfig(max_batches=2, max_session_len=32),
                         CacheConfig(max_batches=2, block_size=4, num_gpu_blocks=8, num_cpu_blocks=0),
                         SequenceMeta(4, strategy=ARSequenceStrategy()))

    prefill, decode = scheduler(), scheduler()
    source = prefill.add_session(1).add_sequence(np.arange(8), preserve_cache=True)
    source.pd_prompt_tokens = np.arange(8)
    prefill.block_manager.allocate(source)
    source.sampling_param.random_seed = 42
    source.update_token_ids(np.array([8]), mode=UpdateTokenMode.PREFILL)
    handoff = prefill.pd_coordinator.publish(source, 'schema', [], first_token_logprobs=logprobs)
    handoff = type(handoff).model_validate_json(handoff.model_dump_json())
    migration = MigrationRequest(protocol=MigrationProtocol.RDMA, remote_engine_id='prefill',
                                 remote_session_id=1, handoff=handoff)
    target = decode.add_session(2).add_sequence(np.arange(8), migration_request=migration)
    target.resp = SimpleNamespace(type=ResponseType.SUCCESS)
    assert decode.schedule_migration() == [target]

    async def control(peer, request):
        sequence = prefill.pd_coordinator.control(request)
        if sequence is not None:
            prefill.end_session(sequence.session_id)

    responses = []
    loop = EngineLoop(
        req_manager=SimpleNamespace(response=lambda resp: responses.append(resp.type)), scheduler=decode,
        state_checkpoints=decode.state_checkpoints, executor=SimpleNamespace(migrate=migrate),
        seq_strategy=ARSequenceStrategy(),
        inputs_maker=SimpleNamespace(has_pending_long_context_chunk=lambda: False),
        config=EngineLoopConfig(role=EngineRole.Decode),
        engine_conn=SimpleNamespace(local_engine_id='decode', control=control))
    return loop, prefill, target, responses


@pytest.mark.parametrize('max_new_tokens', [1, 2])
def test_received_first_token_logprobs_use_normal_response_assembly(max_new_tokens):
    async def run():
        async def migrate(batch):
            return [PDTransferResult(transfer_id=batch.receives[0].handoff.transfer_id, status='completed')]

        loop, _, seq, _ = make_transfer(migrate, [[[-0.2, -0.2, -1.7], [8, 8, 9]]])
        seq.sampling_param.max_new_tokens = max_new_tokens
        seq.resp.data = None
        await loop._receive_handoff(seq)
        await asyncio.gather(*loop._pd_control_tasks)
        output = loop.resp_queue.get_nowait()[seq.session_id]
        assert output.token_ids.tolist() == [8]
        assert output.finish == (max_new_tokens == 1)
        loop._update_logprobs([output])
        assert seq.resp.data['logprobs'] == [{8: -0.2, 9: -1.7}]

    asyncio.run(run())


@pytest.mark.parametrize('force', [False, True])
def test_cancelled_receive_late_completion_cannot_release_reused_blocks(force):
    async def run():
        started = asyncio.Event()
        completion = asyncio.get_running_loop().create_future()

        async def migrate(batch):
            started.set()
            await completion
            return [PDTransferResult(transfer_id=batch.receives[0].handoff.transfer_id, status='completed')]

        loop, prefill, seq, responses = make_transfer(migrate)
        owner = loop.scheduler.pd_coordinator
        task = asyncio.create_task(loop._receive_handoff(seq))
        await asyncio.wait_for(started.wait(), 2)
        transfer_id = seq.migration_request.handoff.transfer_id
        owner.control(PDControlRequest(action='abort', transfer_id=transfer_id,
                                       engine_epoch=prefill.pd_coordinator.epoch, receiver_epoch=owner.epoch,
                                       reason='Prefill cancelled'))
        if force:
            owner.release_timeout = 0
        else:
            completion.set_result(None)
        await asyncio.wait_for(task, 3)
        await asyncio.gather(*loop._pd_control_tasks)
        assert responses == [ResponseType.INTERNAL_ENGINE_ERROR]
        assert owner.snapshot()['forced_releases'] == int(force)
        assert not owner.has_pending() and not prefill.pd_coordinator.has_pending()
        assert loop.scheduler.block_manager.get_num_free_gpu_blocks() == 8
        assert prefill.block_manager.get_num_free_gpu_blocks() == 8

        replacement = loop.scheduler.add_session(3).add_sequence(np.arange(9))
        loop.scheduler.block_manager.allocate(replacement)
        free = loop.scheduler.block_manager.get_num_free_gpu_blocks()
        if force:
            assert loop._pd_native_jobs
            completion.set_result(None)
            await asyncio.gather(*loop._pd_native_jobs)
        assert loop.scheduler.get_session(3) is not None
        assert loop.scheduler.block_manager.get_num_free_gpu_blocks() == free
        assert owner.finish_receive(seq, PDTransferResult(transfer_id=transfer_id, status='completed')) == 'stale'

    asyncio.run(run())


def test_unknown_native_result_expires_without_stopping_decode():
    async def run():
        async def migrate(batch):
            return [PDTransferResult(transfer_id=batch.receives[0].handoff.transfer_id,
                                     status='completion_unknown', error='native failed')]

        loop, prefill, seq, responses = make_transfer(migrate)
        loop.scheduler.pd_coordinator.release_timeout = 0
        await asyncio.wait_for(loop._receive_handoff(seq), 2)
        await asyncio.gather(*loop._pd_control_tasks)
        assert responses == [ResponseType.INTERNAL_ENGINE_ERROR]
        assert not loop.stop_event.is_set()
        assert loop.scheduler.pd_coordinator.snapshot()['quarantined'] == 0
        assert not prefill.pd_coordinator.has_pending()
        replacement = loop.scheduler.add_session(3).add_sequence(np.arange(9))
        assert loop.scheduler.schedule(is_prefill=True).running == [replacement]

    asyncio.run(run())


def test_prefill_abort_before_submission_returns_reason_without_native_io():
    async def run():
        submissions = []

        async def migrate(batch):
            submissions.append(batch)

        loop, prefill, seq, responses = make_transfer(migrate)
        owner = loop.scheduler.pd_coordinator
        handoff = seq.migration_request.handoff
        owner.control(PDControlRequest(action='abort', transfer_id=handoff.transfer_id,
                                       engine_epoch=handoff.engine_epoch, receiver_epoch=owner.epoch,
                                       reason='Prefill admission failed'))
        await loop._receive_handoff(seq)
        await asyncio.gather(*loop._pd_control_tasks)
        assert not submissions
        assert responses == [ResponseType.INTERNAL_ENGINE_ERROR]
        assert seq.resp.err_msg == 'Prefill admission failed'
        assert not owner.has_pending() and not prefill.pd_coordinator.has_pending()

    asyncio.run(run())


def test_receive_completion_wakes_idle_migration_loop_without_spinning():
    async def run():
        loop = object.__new__(EngineLoop)
        loop.stop_event = asyncio.Event()
        loop.migration_event = asyncio.Event()
        loop._sleep_requested = False
        polls = []

        def schedule():
            polls.append(None)
            # Bound the regression itself: a busy loop cannot service asyncio timeouts.
            if len(polls) == 4:
                loop.stop_event.set()
            return []

        loop.scheduler = SimpleNamespace(schedule_migration=schedule, has_migration_waiting=lambda: False)
        loop.migration_event.set()
        task = asyncio.create_task(loop.migration_loop())
        await asyncio.sleep(0)
        assert len(polls) == 1 and not task.done()
        loop.migration_event.set()
        await asyncio.sleep(0)
        assert len(polls) == 2 and not task.done()
        loop.stop_event.set()
        loop.migration_event.set()
        await task

    asyncio.run(run())


def test_preempted_metadata_only_decode_fails_without_replaying_images():
    async def run():
        async def migrate(batch):
            return [PDTransferResult(transfer_id=batch.receives[0].handoff.transfer_id, status='completed')]

        loop, prefill, seq, responses = make_transfer(migrate)
        seq.pd_metadata_only = True
        await loop._receive_handoff(seq)
        await asyncio.gather(*loop._pd_control_tasks)
        loop.scheduler.schedule_pd_decode()
        loop.scheduler.preempt_seqs([seq])
        assert not loop.scheduler.schedule(is_prefill=True).running
        loop._reject_pd_media_recompute()
        assert responses == [ResponseType.INTERNAL_ENGINE_ERROR]
        assert 'retry the request through Prefill' in seq.resp.err_msg
        assert not loop.scheduler.waiting
        assert not prefill.pd_coordinator.has_pending()
        assert loop.scheduler.block_manager.get_num_free_gpu_blocks() == 8
        replacement = loop.scheduler.add_session(3).add_sequence(np.arange(9))
        assert loop.scheduler.schedule(is_prefill=True).running == [replacement]

    asyncio.run(run())
