import asyncio
from contextlib import suppress

from lmdeploy.messages import EngineOutput, ResponseType
from lmdeploy.pytorch.engine.mp_engine.base_worker import EngineOutputGather
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

    async def _collective_rpc_async(self, func, *args, **kwargs):
        if func == 'create_stream_task':
            self.create_started.set()
            if self.wait_for_create:
                await self.allow_create.wait()
            return self.stream_id
        if func == 'get_stream_task_result':
            self.get_started.set()
            await self.allow_get.wait()
            return EngineOutput(ResponseType.FINISH, [1]), True
        if func == 'drop_stream_task':
            self.drop_stream_ids.append(args[0])
            self.drop_called.set()
            return None
        raise AssertionError(f'Unexpected fake Ray RPC: {func}')


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
    worker = RayEngineWorker.__new__(RayEngineWorker)
    worker._stream_aiter = {}
    worker._stream_task = {}
    worker._engine_output_gather = EngineOutputGather()

    assert await worker.get_stream_task_result(404) == (None, True)

    stream_id = 123
    event = asyncio.Event()
    stream_out = [event, None]
    worker._stream_aiter[stream_id] = stream_out

    get_task = asyncio.create_task(worker.get_stream_task_result(stream_id))
    await asyncio.sleep(0)
    worker._stream_aiter.pop(stream_id)

    stream_out[1] = (EngineOutput(ResponseType.FINISH, [7]), True)
    event.set()
    result, stopped = await asyncio.wait_for(get_task, timeout=1)

    assert stopped is True
    assert result.status == ResponseType.FINISH
    assert result.token_ids == [7]


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

    first, stopped = await worker.get_stream_task_result(stream_id)
    assert first.token_ids == [1, 2]
    assert stopped is False

    allow_final.set()
    final, stopped = await worker.get_stream_task_result(stream_id)
    assert final.token_ids == [3, 4, 5]
    assert stopped is False

    allow_close.set()
    end_marker, stopped = await worker.get_stream_task_result(stream_id)
    assert end_marker is None
    assert stopped is True


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

    final, stopped = await worker.get_stream_task_result(stream_id)
    assert final.token_ids == [6, 7, 8]
    assert stopped is True


def test_ray_stream_returns_unconsumed_final_output_once():
    asyncio.run(_async_test_ray_stream_returns_unconsumed_final_output_once())
