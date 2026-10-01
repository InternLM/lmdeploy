import asyncio
from contextlib import suppress

from lmdeploy.messages import EngineOutput, ResponseType
from lmdeploy.pytorch.engine.mp_engine.base_worker import (
    EngineOutputGather,
    StreamMailbox,
    StreamPollResult,
)
from lmdeploy.pytorch.engine.mp_engine.ray_engine import RayEngineWorker, RayMPEngine


def _make_worker(stream=None):
    worker = RayEngineWorker.__new__(RayEngineWorker)
    worker._stream_id = 0
    worker._stream_aiter = {}
    worker._stream_task = {}
    worker._engine_output_gather = EngineOutputGather()
    if stream is not None:
        worker.instance_async_stream_infer = stream
    return worker


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


class _LocalRayMPEngine(RayMPEngine):

    def __init__(self, worker):
        self.worker = worker

    async def _collective_rpc_async(self, func, *args, **kwargs):
        return await getattr(self.worker, func)(*args, **kwargs)


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


async def _async_test_ray_get_stream_task_result_after_drop_is_idempotent():
    worker = _make_worker()

    poll_result = await worker.get_stream_task_result(404)
    assert poll_result == StreamPollResult(done=True)

    stream_id = 123
    stream_out = StreamMailbox()
    worker._stream_aiter[stream_id] = stream_out

    get_task = asyncio.create_task(worker.get_stream_task_result(stream_id))
    await asyncio.sleep(0)
    worker._stream_aiter.pop(stream_id)

    stream_out.publish(None)
    stream_out.finish()
    poll_result = await asyncio.wait_for(get_task, timeout=1)

    assert poll_result == StreamPollResult(output=None, has_output=True, done=True)


def test_ray_get_stream_task_result_after_drop_is_idempotent():
    asyncio.run(_async_test_ray_get_stream_task_result_after_drop_is_idempotent())


async def _async_test_ray_stream_does_not_repeat_final_output():
    allow_close = asyncio.Event()

    async def fake_stream(notify_add_msg_func=None):
        notify_add_msg_func()
        yield EngineOutput(ResponseType.FINISH, [3, 4, 5])
        # Keep the generator alive after its final yield, matching async
        # session cleanup in EngineInstance.async_stream_infer.
        await allow_close.wait()

    engine = _LocalRayMPEngine(_make_worker(fake_stream))
    stream = engine._collective_rpc_streaming_async('instance_async_stream_infer', asyncio.Event())
    outputs_task = asyncio.create_task(anext(stream))
    output = await asyncio.wait_for(outputs_task, timeout=1)
    assert output.token_ids == [3, 4, 5]

    allow_close.set()
    with suppress(StopAsyncIteration):
        await asyncio.wait_for(anext(stream), timeout=1)
        raise AssertionError('final output was repeated')


def test_ray_stream_does_not_repeat_final_output():
    asyncio.run(_async_test_ray_stream_does_not_repeat_final_output())


async def _async_test_ray_stream_error_before_first_output():

    async def failing_stream(notify_add_msg_func=None):
        notify_add_msg_func()
        raise RuntimeError('injected stream failure')
        yield

    engine = _LocalRayMPEngine(_make_worker(failing_stream))
    outputs = [
        output async for output in engine._collective_rpc_streaming_async(
            'instance_async_stream_infer', asyncio.Event())
    ]
    assert [output.status for output in outputs] == [ResponseType.INTERNAL_ENGINE_ERROR]


def test_ray_stream_error_before_first_output():
    asyncio.run(_async_test_ray_stream_error_before_first_output())


def test_engine_output_gather_preserves_logprob_carrier_states():
    for carrier in [None, [], [{2: -0.5}]]:
        gather = EngineOutputGather()
        final = EngineOutput(ResponseType.FINISH, [], logprobs=carrier)
        gather.add(1, final)
        assert gather.pop(1, final).logprobs == carrier
        assert 1 not in gather._output
