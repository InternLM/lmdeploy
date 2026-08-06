import asyncio
from contextlib import suppress

import pytest

from lmdeploy.messages import EngineOutput, ResponseType
from lmdeploy.pytorch.engine.mp_engine.base_worker import (
    EngineOutputGather,
    MPStreamError,
    StreamError,
    StreamMailbox,
    StreamPollResult,
)
from lmdeploy.pytorch.engine.mp_engine.ray_engine import RayEngineWorker, RayMPEngine


class _FakeRayMPEngine(RayMPEngine):

    def __init__(self):
        self.create_started = asyncio.Event()
        self.allow_create = asyncio.Event()
        self.get_started = asyncio.Event()
        self.allow_get = asyncio.Event()
        self.drop_called = asyncio.Event()
        self.drop_stream_ids = []
        self.stream_id = 260606
        self.wait_for_create = True
        self.poll_result = StreamPollResult(
            output=EngineOutput(ResponseType.FINISH, [1]),
            has_output=True,
            done=True,
        )

    async def _collective_rpc_async(self, func, *args, **kwargs):
        if func == 'create_stream_task':
            self.create_started.set()
            if self.wait_for_create:
                await self.allow_create.wait()
            return self.stream_id
        if func == 'get_stream_task_result':
            self.get_started.set()
            await self.allow_get.wait()
            return self.poll_result
        if func == 'drop_stream_task':
            self.drop_stream_ids.append(args[0])
            self.drop_called.set()
            return None
        raise AssertionError(f'Unexpected fake Ray RPC: {func}')


def test_stream_mailbox_preserves_none_output():
    mailbox = StreamMailbox()
    mailbox.publish(None)
    mailbox.finish()

    poll_result = mailbox.drain()

    assert poll_result.output is None
    assert poll_result.has_output is True
    assert poll_result.done is True
    assert poll_result.error is None
    assert mailbox.event.is_set() is False


async def _async_test_ray_stream_startup_cancel_drops_remote_stream():
    engine = _FakeRayMPEngine()
    init_done = asyncio.Event()
    stream = engine._collective_rpc_streaming_async('instance_async_stream_infer', init_done)
    stream_task = asyncio.create_task(stream.__anext__())

    await asyncio.wait_for(engine.create_started.wait(), timeout=1)
    stream_task.cancel()
    with suppress(asyncio.CancelledError):
        await stream_task

    assert not init_done.is_set()
    engine.allow_create.set()
    await asyncio.wait_for(init_done.wait(), timeout=1)
    await asyncio.wait_for(engine.drop_called.wait(), timeout=1)
    assert engine.drop_stream_ids == [engine.stream_id]


def test_ray_stream_startup_cancel_drops_remote_stream():
    asyncio.run(_async_test_ray_stream_startup_cancel_drops_remote_stream())


async def _async_test_ray_stream_cancel_after_start_drops_remote_stream():
    engine = _FakeRayMPEngine()
    engine.wait_for_create = False
    init_done = asyncio.Event()
    stream = engine._collective_rpc_streaming_async('instance_async_stream_infer', init_done)
    stream_task = asyncio.create_task(stream.__anext__())

    await asyncio.wait_for(init_done.wait(), timeout=1)
    await asyncio.wait_for(engine.get_started.wait(), timeout=1)
    stream_task.cancel()
    with suppress(asyncio.CancelledError):
        await stream_task

    await asyncio.wait_for(engine.drop_called.wait(), timeout=1)
    assert engine.drop_stream_ids == [engine.stream_id]


def test_ray_stream_cancel_after_start_drops_remote_stream():
    asyncio.run(_async_test_ray_stream_cancel_after_start_drops_remote_stream())


async def _async_test_ray_stream_close_after_output_drops_remote_stream():
    """Model an HTTP abort/disconnect after a response chunk was consumed."""
    engine = _FakeRayMPEngine()
    engine.wait_for_create = False
    engine.allow_get.set()
    engine.poll_result = StreamPollResult(
        output=EngineOutput(ResponseType.SUCCESS, [1, 2]),
        has_output=True,
        done=False,
    )
    init_done = asyncio.Event()
    stream = engine._collective_rpc_streaming_async('instance_async_stream_infer', init_done)

    output = await stream.__anext__()
    assert output.token_ids == [1, 2]
    await stream.aclose()

    await asyncio.wait_for(engine.drop_called.wait(), timeout=1)
    assert engine.drop_stream_ids == [engine.stream_id]


def test_ray_stream_close_after_output_drops_remote_stream():
    asyncio.run(_async_test_ray_stream_close_after_output_drops_remote_stream())


async def _async_test_ray_worker_abort_discards_pending_output():
    """Dropping an active producer must discard, not replay, pending data."""
    worker = RayEngineWorker.__new__(RayEngineWorker)
    worker._stream_id = 0
    worker._stream_aiter = {}
    worker._stream_task = {}
    worker._engine_output_gather = EngineOutputGather()
    producer_closed = asyncio.Event()

    async def fake_stream(notify_add_msg_func=None):
        notify_add_msg_func()
        try:
            yield EngineOutput(ResponseType.SUCCESS, [3, 4])
            await asyncio.Event().wait()
        finally:
            producer_closed.set()

    worker.instance_async_stream_infer = fake_stream
    stream_id = await worker.create_stream_task('instance_async_stream_infer')
    await asyncio.sleep(0)
    assert stream_id in worker._engine_output_gather._output

    await worker.drop_stream_task(stream_id)

    assert producer_closed.is_set()
    assert stream_id not in worker._stream_aiter
    assert stream_id not in worker._stream_task
    assert stream_id not in worker._engine_output_gather._output
    assert await worker.get_stream_task_result(stream_id) == StreamPollResult(done=True)


def test_ray_worker_abort_discards_pending_output():
    asyncio.run(_async_test_ray_worker_abort_discards_pending_output())


async def _async_test_ray_get_stream_task_result_after_drop_is_idempotent():
    worker = RayEngineWorker.__new__(RayEngineWorker)
    worker._stream_aiter = {}
    worker._stream_task = {}
    worker._engine_output_gather = EngineOutputGather()

    poll_result = await worker.get_stream_task_result(404)
    assert poll_result == StreamPollResult(done=True)

    stream_id = 123
    stream_out = StreamMailbox()
    worker._stream_aiter[stream_id] = stream_out

    get_task = asyncio.create_task(worker.get_stream_task_result(stream_id))
    await asyncio.sleep(0)
    worker._stream_aiter.pop(stream_id)

    stream_out.publish(EngineOutput(ResponseType.FINISH, [7]))
    stream_out.finish()
    poll_result = await asyncio.wait_for(get_task, timeout=1)

    assert poll_result.done is True
    assert poll_result.has_output is True
    assert poll_result.output.status == ResponseType.FINISH
    assert poll_result.output.token_ids == [7]


def test_ray_get_stream_task_result_after_drop_is_idempotent():
    asyncio.run(_async_test_ray_get_stream_task_result_after_drop_is_idempotent())


async def _async_test_ray_stream_does_not_repeat_consumed_final_output():
    worker = RayEngineWorker.__new__(RayEngineWorker)
    worker._stream_id = 0
    worker._stream_aiter = {}
    worker._stream_task = {}
    worker._engine_output_gather = EngineOutputGather()
    allow_final = asyncio.Event()
    allow_close = asyncio.Event()

    async def fake_stream(notify_add_msg_func=None):
        notify_add_msg_func()
        yield EngineOutput(ResponseType.SUCCESS, [1, 2])
        await allow_final.wait()
        yield EngineOutput(ResponseType.FINISH, [3, 4, 5])
        # Keep the generator alive after its final yield, matching async
        # session cleanup in EngineInstance.async_stream_infer.
        await allow_close.wait()

    worker.instance_async_stream_infer = fake_stream
    stream_id = await worker.create_stream_task('instance_async_stream_infer')

    poll_result = await worker.get_stream_task_result(stream_id)
    assert poll_result.output.token_ids == [1, 2]
    assert poll_result.has_output is True
    assert poll_result.done is False

    allow_final.set()
    poll_result = await worker.get_stream_task_result(stream_id)
    assert poll_result.output.token_ids == [3, 4, 5]
    assert poll_result.has_output is True
    assert poll_result.done is False

    allow_close.set()
    poll_result = await worker.get_stream_task_result(stream_id)
    assert poll_result.has_output is False
    assert poll_result.done is True
    assert poll_result.error is None


def test_ray_stream_does_not_repeat_consumed_final_output():
    asyncio.run(_async_test_ray_stream_does_not_repeat_consumed_final_output())


async def _async_test_ray_stream_returns_unconsumed_final_output_once():
    worker = RayEngineWorker.__new__(RayEngineWorker)
    worker._stream_id = 0
    worker._stream_aiter = {}
    worker._stream_task = {}
    worker._engine_output_gather = EngineOutputGather()

    async def fake_stream(notify_add_msg_func=None):
        notify_add_msg_func()
        yield EngineOutput(ResponseType.FINISH, [6, 7, 8])

    worker.instance_async_stream_infer = fake_stream
    stream_id = await worker.create_stream_task('instance_async_stream_infer')
    await asyncio.sleep(0)

    poll_result = await worker.get_stream_task_result(stream_id)
    assert poll_result.output.token_ids == [6, 7, 8]
    assert poll_result.has_output is True
    assert poll_result.done is True
    assert poll_result.error is None


def test_ray_stream_returns_unconsumed_final_output_once():
    asyncio.run(_async_test_ray_stream_returns_unconsumed_final_output_once())


async def _async_test_ray_stream_failure_before_output_returns_error_envelope():
    worker = RayEngineWorker.__new__(RayEngineWorker)
    worker._stream_id = 0
    worker._stream_aiter = {}
    worker._stream_task = {}
    worker._engine_output_gather = EngineOutputGather()

    async def fake_stream(notify_add_msg_func=None):
        notify_add_msg_func()
        raise RuntimeError('failed before first output')
        yield

    worker.instance_async_stream_infer = fake_stream
    stream_id = await worker.create_stream_task('instance_async_stream_infer')

    poll_result = await worker.get_stream_task_result(stream_id)
    assert poll_result.has_output is False
    assert poll_result.done is True
    assert poll_result.error.type_name == 'builtins.RuntimeError'
    assert poll_result.error.message == 'failed before first output'


def test_ray_stream_failure_before_output_returns_error_envelope():
    asyncio.run(_async_test_ray_stream_failure_before_output_returns_error_envelope())


async def _async_test_ray_stream_failure_after_pending_output_preserves_both():
    worker = RayEngineWorker.__new__(RayEngineWorker)
    worker._stream_id = 0
    worker._stream_aiter = {}
    worker._stream_task = {}
    worker._engine_output_gather = EngineOutputGather()

    async def fake_stream(notify_add_msg_func=None):
        notify_add_msg_func()
        yield EngineOutput(ResponseType.SUCCESS, [8, 9])
        raise RuntimeError('failed after output')

    worker.instance_async_stream_infer = fake_stream
    stream_id = await worker.create_stream_task('instance_async_stream_infer')
    await asyncio.sleep(0)

    poll_result = await worker.get_stream_task_result(stream_id)
    assert poll_result.output.status == ResponseType.SUCCESS
    assert poll_result.output.token_ids == [8, 9]
    assert poll_result.has_output is True
    assert poll_result.done is True
    assert poll_result.error.type_name == 'builtins.RuntimeError'
    assert poll_result.error.message == 'failed after output'


def test_ray_stream_failure_after_pending_output_preserves_both():
    asyncio.run(_async_test_ray_stream_failure_after_pending_output_preserves_both())


async def _async_test_ray_stream_client_delivers_output_then_engine_error():
    engine = _FakeRayMPEngine()
    engine.wait_for_create = False
    engine.allow_get.set()
    engine.poll_result = StreamPollResult(
        output=EngineOutput(ResponseType.SUCCESS, [8, 9]),
        has_output=True,
        done=True,
        error=StreamError(type_name='builtins.RuntimeError', message='failed after output'),
    )
    init_done = asyncio.Event()

    outputs = [
        output async for output in engine._collective_rpc_streaming_async('instance_async_stream_infer', init_done)
    ]

    assert [output.status for output in outputs] == [ResponseType.SUCCESS, ResponseType.INTERNAL_ENGINE_ERROR]
    assert [output.token_ids for output in outputs] == [[8, 9], []]


def test_ray_stream_client_delivers_output_then_engine_error():
    asyncio.run(_async_test_ray_stream_client_delivers_output_then_engine_error())


async def _async_test_ray_stream_client_maps_early_failure_to_engine_error():
    engine = _FakeRayMPEngine()
    engine.wait_for_create = False
    engine.allow_get.set()
    engine.poll_result = StreamPollResult(
        done=True,
        error=StreamError(type_name='builtins.RuntimeError', message='failed before output'),
    )
    init_done = asyncio.Event()

    outputs = [
        output async for output in engine._collective_rpc_streaming_async('instance_async_stream_infer', init_done)
    ]

    assert len(outputs) == 1
    assert outputs[0].status == ResponseType.INTERNAL_ENGINE_ERROR
    assert outputs[0].token_ids == []


def test_ray_stream_client_maps_early_failure_to_engine_error():
    asyncio.run(_async_test_ray_stream_client_maps_early_failure_to_engine_error())


async def _async_test_ray_generic_stream_delivers_output_then_raises():
    engine = _FakeRayMPEngine()
    engine.wait_for_create = False
    engine.allow_get.set()
    engine.poll_result = StreamPollResult(
        output='pending output',
        has_output=True,
        done=True,
        error=StreamError(type_name='builtins.RuntimeError', message='generic failure'),
    )
    stream = engine._collective_rpc_streaming_async('generic_stream', asyncio.Event())

    assert await stream.__anext__() == 'pending output'
    with pytest.raises(MPStreamError, match='generic failure'):
        await stream.__anext__()


def test_ray_generic_stream_delivers_output_then_raises():
    asyncio.run(_async_test_ray_generic_stream_delivers_output_then_raises())
