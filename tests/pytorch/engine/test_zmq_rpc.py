import asyncio
import inspect
import multiprocessing as mp
import os
import pickle
from contextlib import suppress
from multiprocessing import resource_sharer
from types import SimpleNamespace

import pytest
import torch
import zmq

from lmdeploy.messages import EngineOutput, ResponseType
from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.pytorch.engine.mp_engine.base import MPEngine
from lmdeploy.pytorch.engine.mp_engine.base_worker import StreamMailbox, StreamPollResult


def test_engine_worker_health_status_is_synchronous():
    from lmdeploy.pytorch.engine.mp_engine.base_worker import EngineWorkerBase

    status = dict(alive=True, message='healthy', schedule_metrics=None)

    class _FakeEngine:

        def get_local_health_status(self):
            return status

    worker = EngineWorkerBase.__new__(EngineWorkerBase)
    worker.engine = _FakeEngine()

    assert not inspect.iscoroutinefunction(worker.get_health_status)
    assert worker.get_health_status() is status

    from lmdeploy.pytorch.engine.engine import Engine
    from lmdeploy.pytorch.paging.pd_coordinator import PDCoordinator
    worker.engine = Engine.__new__(Engine)
    worker.engine.engine_config = SimpleNamespace(role=EngineRole.Decode)
    owner = PDCoordinator(None, None)
    worker.engine.scheduler = SimpleNamespace(pd_coordinator=owner)
    worker.engine.req_manager = SimpleNamespace(is_loop_alive=lambda: True)
    worker.engine._loop_main = None
    worker.engine._engine_loop = None
    worker.engine.get_schedule_metrics = lambda: None
    status = worker.get_health_status()
    assert status['alive'] and status['pd']['quarantined'] == 0

    worker.engine._engine_loop = SimpleNamespace(forward_started_at=0, tasks=[])
    assert not worker.get_health_status()['alive']
    worker.engine._engine_loop.forward_started_at = None
    assert worker.get_health_status()['alive']

    from lmdeploy.pytorch.disagg.conn.protocol import PDControlRequest
    worker.engine.engine_conn = SimpleNamespace(apply_control=owner.control)
    request = PDControlRequest(action='release', transfer_id='expired', engine_epoch='old')
    assert worker.pd_control(request)['status'] == 'CONFLICT'
    request.engine_epoch = owner.epoch
    assert worker.pd_control(request) == {'status': 'SUCCESS'}


@pytest.mark.parametrize('role', [EngineRole.Hybrid, EngineRole.Prefill, EngineRole.Decode])
@pytest.mark.parametrize('elapsed', [None, 179, 180, 181])
def test_engine_health_forward_timeout_only_applies_to_pd(monkeypatch, role, elapsed):
    from lmdeploy.pytorch.engine.engine import Engine
    from lmdeploy.pytorch.paging.pd_coordinator import PDCoordinator

    monkeypatch.setattr('lmdeploy.pytorch.engine.engine.time.monotonic', lambda: 1000)
    engine = Engine.__new__(Engine)
    engine.engine_config = SimpleNamespace(role=role)
    owner = PDCoordinator(None, None)
    owner.transfer_timeout = 180
    metrics = object()
    engine.scheduler = SimpleNamespace(pd_coordinator=owner, schedule_metrics=metrics)
    engine.req_manager = SimpleNamespace(is_loop_alive=lambda: True)
    engine._loop_main = None
    engine._engine_loop = SimpleNamespace(forward_started_at=None if elapsed is None else 1000 - elapsed, tasks=[])

    status = engine.get_local_health_status()
    expected_alive = role == EngineRole.Hybrid or elapsed is None or elapsed < 180
    assert status['alive'] is expected_alive
    assert status['schedule_metrics'] is (metrics if expected_alive else None)

    # Skipping the PD deadline must not hide a stopped inference loop.
    engine._engine_loop.forward_started_at = None
    engine._engine_loop.tasks = [SimpleNamespace(done=lambda: True, get_name=lambda: 'MainLoopMain')]
    status = engine.get_local_health_status()
    assert not status['alive']
    assert status['message'] == "PyTorch engine loop task has stopped: ['MainLoopMain']."


def _exit_when_released(ready, release, startup_delay):
    import time
    time.sleep(startup_delay)
    ready.set()
    if not release.wait(30):
        raise TimeoutError('parent did not release sentinel test child')


class TestZMQRPC:

    def sub_proc(self, shared_dict=None, condition=None):
        from lmdeploy.pytorch.engine.mp_engine.zmq_rpc import AsyncRPCServer
        server = AsyncRPCServer()
        with condition:
            shared_dict['rpc_server_port'] = server.port
            condition.notify()

        async def streaming_method(name):
            for i in range(3):
                yield f'{name}: streaming method {i}'

        async def instance_async_stream_infer(name, notify_add_msg_func=None, session_id=None):
            await asyncio.sleep(0.2)
            if notify_add_msg_func is not None:
                notify_add_msg_func()
            if name == 'error-before-output':
                raise RuntimeError('inference failed before output')
            yield f'{name}: instance async stream infer'

        async def streaming_error(name):
            yield f'{name}: pending output'
            raise RuntimeError('generic stream failed after output')

        def method(name):
            return f'{name}: method'

        async def async_method(name):
            return f'{name}: async method'

        def close():
            print('close server...')
            server.stop()

        def stream_state():
            return (len(server.stream_output), len(server.stream_tasks), len(server._engine_output_gather._output))

        def shared_tensor(tensor):
            assert tensor.is_shared()
            tensor.add_(1)
            return tensor.sum().item()

        server.register_method('method', method)
        server.register_method('async_method', async_method)
        server.register_method('streaming_method', streaming_method)
        server.register_method('streaming_error', streaming_error)
        server.register_method('instance_async_stream_infer', instance_async_stream_infer)
        server.register_method('stream_state', stream_state)
        server.register_method('close', close)
        server.register_method('shared_tensor', shared_tensor)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        asyncio.run(server.run())

    async def async_main(self, port):
        from lmdeploy.pytorch.engine.mp_engine.base_worker import MPStreamError
        from lmdeploy.pytorch.engine.mp_engine.zmq_rpc import AsyncRPCClient
        client = AsyncRPCClient(port=port)

        loop = asyncio.get_event_loop()
        _ = loop.create_task(client.listen())

        # Example usage
        result = client.call('async_method', 'test2')
        assert result == 'test2: async method'
        result = await client.async_call('method', 'test1')
        assert result == 'test1: method'

        pixels = torch.zeros(1024, 1024).share_memory_()
        assert await client.async_call('shared_tensor', pixels) == pixels.numel()
        # The receiver sees the same storage, not a deserialized tensor copy.
        assert torch.all(pixels == 1)

        async for result in client.async_stream_call('streaming_method', asyncio.Event(), 'test3'):
            pass
        assert result == 'test3: streaming method 2'

        error_stream = client.async_stream_call('streaming_error', asyncio.Event(), 'test-error')
        assert await error_stream.__anext__() == 'test-error: pending output'
        with pytest.raises(MPStreamError, match='generic stream failed after output'):
            await error_stream.__anext__()

        inference_error_stream = client.async_stream_call(
            'instance_async_stream_infer',
            asyncio.Event(),
            'error-before-output',
            session_id=98,
            streaming_startup_notify_kwarg='notify_add_msg_func',
        )
        inference_error_outputs = [output async for output in inference_error_stream]
        assert len(inference_error_outputs) == 1
        assert inference_error_outputs[0].status == ResponseType.INTERNAL_ENGINE_ERROR
        assert inference_error_outputs[0].token_ids == []

        init_done = asyncio.Event()
        stream = client.async_stream_call('instance_async_stream_infer',
                                          init_done,
                                          'test4',
                                          session_id=99,
                                          streaming_startup_notify_kwarg='notify_add_msg_func')
        stream_task = asyncio.create_task(stream.__anext__())
        await asyncio.sleep(0.05)
        assert not init_done.is_set()
        stream_task.cancel()
        with suppress(asyncio.CancelledError):
            await stream_task
        await asyncio.wait_for(init_done.wait(), timeout=1)
        await stream.aclose()
        for _ in range(10):
            state = await client.async_call('stream_state')
            if state == (0, 0, 0) and client.pending == {}:
                break
            await asyncio.sleep(0.1)
        assert state == (0, 0, 0)
        assert client.pending == {}

        await client.async_call('close')
        client.stop()

    def test_zmq_rpc(self):
        with mp.Manager() as manager:
            shared_dict = manager.dict()
            condition = manager.Condition()
            ctx = mp.get_context('spawn')
            proc = ctx.Process(target=self.sub_proc, args=(shared_dict, condition), daemon=True)
            proc.start()

            with condition:
                if 'rpc_server_port' not in shared_dict:
                    condition.wait()
            port = shared_dict['rpc_server_port']

        asyncio.run(self.async_main(port))

        proc.join()


class _FakeMPEngine(MPEngine):

    def __init__(self):
        self.started = asyncio.Event()
        self.allow_init = asyncio.Event()
        self.calls = []
        super().__init__()

    def _collective_rpc(self, func, *args, **kwargs):
        assert func == 'get_engine_config'
        return None

    async def _collective_rpc_async(self, func, *args, **kwargs):
        self.calls.append((func, args))
        return ResponseType.SUCCESS

    async def _collective_rpc_streaming_async(self, func, init_done, *args, **kwargs):

        async def _startup():
            self.started.set()
            await self.allow_init.wait()
            init_done.set()

        startup_task = asyncio.create_task(_startup())
        await asyncio.shield(startup_task)
        yield EngineOutput(ResponseType.CANCEL, [])


@pytest.mark.parametrize('mode', ['cancel', 'cancel_end', 'stream_cancel'])
def test_mp_cancel_during_stream_startup(mode):

    async def _test():
        engine = _FakeMPEngine()
        instance = engine.create_instance()
        stream = instance.async_stream_infer(7)
        stream_task = asyncio.create_task(stream.__anext__())
        await asyncio.wait_for(engine.started.wait(), timeout=1)

        if mode == 'stream_cancel':
            stream_task.cancel()
            with suppress(asyncio.CancelledError):
                await stream_task

        assert await instance.async_cancel(7) == ResponseType.SUCCESS
        state = engine.session_states[7]
        end_task = asyncio.create_task(instance.async_end(7)) if mode == 'cancel_end' else None
        await asyncio.sleep(0)
        assert engine.calls == []
        if end_task is not None:
            assert not end_task.done()

        engine.allow_init.set()
        if mode != 'stream_cancel':
            output = await asyncio.wait_for(stream_task, timeout=1)
            assert output.status == ResponseType.CANCEL
        if end_task is not None:
            assert await asyncio.wait_for(end_task, timeout=1) == ResponseType.SUCCESS
        elif state.cleanup_task is not None:
            await asyncio.wait_for(state.cleanup_task, timeout=1)
        else:
            for _ in range(10):
                if engine.calls:
                    break
                await asyncio.sleep(0)

        expected = [('instance_async_cancel', (7, ))]
        if mode == 'cancel_end':
            expected.append(('instance_async_end', (7, )))
        assert engine.calls == expected
        await stream.aclose()
        assert 7 not in engine.session_states
        assert 7 not in engine.pending_cancel_sessions

    asyncio.run(_test())


def test_zmq_backend_dead_callback_wakes_stream_init_waiters():
    from lmdeploy.pytorch.engine.mp_engine.base import SessionState
    from lmdeploy.pytorch.engine.mp_engine.zmq_engine import ZMQMPEngine

    engine = ZMQMPEngine.__new__(ZMQMPEngine)
    engine.session_states = {11: SessionState()}
    assert not engine.session_states[11].init_done.is_set()

    engine._mark_backend_dead()
    assert engine.session_states[11].init_done.is_set()


async def _async_test_zmq_pending_rpc_wakes_on_backend_death():
    from lmdeploy.pytorch.engine.mp_engine.zmq_rpc import AsyncRPCClient, RPCServerDeadError

    client = AsyncRPCClient(port=1)
    try:
        future = asyncio.get_running_loop().create_future()
        client.pending['request-id'] = future

        client._mark_server_dead('test backend dead', log=False)
        try:
            await asyncio.wait_for(future, timeout=1)
        except RPCServerDeadError:
            pass
        else:
            raise AssertionError('pending RPC did not fail when backend died')
        assert client.pending == {}
    finally:
        client.stop()


def test_zmq_pending_rpc_wakes_on_backend_death():
    asyncio.run(_async_test_zmq_pending_rpc_wakes_on_backend_death())


@pytest.mark.parametrize('failure', ['pickle', 'send', 'server_death'])
def test_zmq_failed_shared_tensor_requests_release_fds(failure):
    from lmdeploy.pytorch.engine.mp_engine.zmq_rpc import AsyncRPCClient, _RequestPickler, _RPCServerLiveness

    async def run():
        pixels = torch.ones(16).share_memory_()
        # Start the process-wide FD sharer before measuring request resources.
        pickle.loads(_RequestPickler.dumps(pixels, []))
        await asyncio.sleep(0.01)
        initial_fds = len(os.listdir('/proc/self/fd'))
        initial_handles = set(resource_sharer._resource_sharer._cache)
        for _ in range(20):
            client = AsyncRPCClient.__new__(AsyncRPCClient)
            client.pending = {}
            client._server_liveness = _RPCServerLiveness(client.pending)
            client._try_start_listen = lambda: None

            def send(data):
                sent = asyncio.get_running_loop().create_future()
                if failure == 'send':
                    sent.set_exception(zmq.ZMQError(zmq.ENOTSOCK))
                else:
                    sent.set_result(None)
                    client._mark_server_dead('test backend dead', log=False)
                return sent

            client.async_socket = SimpleNamespace(send=send)
            args = (pixels, lambda: None) if failure == 'pickle' else (pixels,)
            with pytest.raises((AttributeError, zmq.ZMQError, RuntimeError)):
                await client.async_call('shared_tensor', *args)
            await asyncio.sleep(0)
            assert client.pending == {}
            assert set(resource_sharer._resource_sharer._cache) == initial_handles
        # Background cleanup may close other FDs while requests are failing.
        assert len(os.listdir('/proc/self/fd')) <= initial_fds

    asyncio.run(run())


@pytest.mark.parametrize('cancel_during', ['send', 'reply'])
def test_zmq_cancel_preserves_unreceived_shared_tensor_handles(cancel_during):
    from lmdeploy.pytorch.engine.mp_engine.zmq_rpc import AsyncRPCClient, _RPCServerLiveness

    async def run():
        client = AsyncRPCClient.__new__(AsyncRPCClient)
        client.pending = {}
        client._server_liveness = _RPCServerLiveness(client.pending)
        client._try_start_listen = lambda: None
        sent = asyncio.get_running_loop().create_future()
        queued = asyncio.Event()
        payloads = []

        def send(data):
            payloads.append(data)
            queued.set()
            return sent

        client.async_socket = SimpleNamespace(send=send)
        pixels = torch.zeros(16).share_memory_()
        initial_handles = set(resource_sharer._resource_sharer._cache)
        task = asyncio.create_task(client.async_call('shared_tensor', pixels))
        await queued.wait()
        if cancel_during == 'reply':
            sent.set_result(None)
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(client.pending) == 1
        assert len(resource_sharer._resource_sharer._cache) == len(initial_handles) + 1

        request = pickle.loads(payloads.pop())  # The receiver attaches after cancellation.
        request['args'][0].add_(1)
        assert torch.all(pixels == 1)
        if cancel_during == 'send':
            sent.set_result(None)
        client._set_reply(dict(request_id=request['request_id'], success=True, result=None))
        await asyncio.sleep(0)
        assert client.pending == {}
        assert set(resource_sharer._resource_sharer._cache) == initial_handles

    asyncio.run(run())


async def _async_test_zmq_get_stream_output_after_drop_is_idempotent():
    from lmdeploy.pytorch.engine.mp_engine.zmq_rpc import AsyncRPCServer

    server = AsyncRPCServer()
    try:
        stream_id = 123
        stream_out = StreamMailbox()
        server.stream_output[stream_id] = stream_out

        get_task = asyncio.create_task(server.get_stream_output(stream_id))
        await asyncio.sleep(0)
        server.drop_stream_output(stream_id)

        stream_out.finish()
        poll_result = await asyncio.wait_for(get_task, timeout=1)
        assert poll_result == StreamPollResult(done=True)
        assert await server.get_stream_output(stream_id) == StreamPollResult(done=True)
    finally:
        server.stop()
        server.socket.close(0)
        server.context.term()


def test_zmq_get_stream_output_after_drop_is_idempotent():
    asyncio.run(_async_test_zmq_get_stream_output_after_drop_is_idempotent())


async def _async_test_zmq_process_sentinel_marks_backend_dead(startup_delay):
    from lmdeploy.pytorch.engine.mp_engine.zmq_rpc import AsyncRPCClient

    ctx = mp.get_context('spawn')
    ready, release = ctx.Event(), ctx.Event()
    proc = ctx.Process(target=_exit_when_released, args=(ready, release, startup_delay))
    proc.start()
    assert proc.is_alive()

    marked = asyncio.Event()
    client = AsyncRPCClient(port=1,
                            server_alive_callback=proc.is_alive,
                            server_sentinel=proc.sentinel,
                            server_dead_callback=marked.set)
    try:
        # Spawn imports (including torch under coverage) are not sentinel latency.
        assert await asyncio.to_thread(ready.wait, 60), 'child startup timed out'
        client._register_server_sentinel()
        release.set()
        await asyncio.wait_for(marked.wait(), timeout=5)
    finally:
        release.set()
        client.stop()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
        proc.close()


@pytest.mark.parametrize('startup_delay', [0, 6])
def test_zmq_process_sentinel_marks_backend_dead(startup_delay):
    asyncio.run(_async_test_zmq_process_sentinel_marks_backend_dead(startup_delay))
