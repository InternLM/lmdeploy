import asyncio
import multiprocessing as mp
from contextlib import suppress

import pytest

from lmdeploy.messages import EngineOutput, ResponseType
from lmdeploy.pytorch.engine.mp_engine.base import MPEngine
from lmdeploy.pytorch.engine.mp_engine.base_worker import StreamMailbox, StreamPollResult


def _sleep_then_exit():
    import time
    time.sleep(0.2)


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
            if name == 'wait-after-output':
                await asyncio.Event().wait()

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

        server.register_method('method', method)
        server.register_method('async_method', async_method)
        server.register_method('streaming_method', streaming_method)
        server.register_method('streaming_error', streaming_error)
        server.register_method('instance_async_stream_infer', instance_async_stream_infer)
        server.register_method('stream_state', stream_state)
        server.register_method('close', close)

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

        # Model /abort_request or an HTTP disconnect after the first response
        # chunk.  Closing the client generator must drop the remote producer
        # and all pending/coalesced state without replaying that chunk.
        after_output_init = asyncio.Event()
        after_output_stream = client.async_stream_call(
            'instance_async_stream_infer',
            after_output_init,
            'wait-after-output',
            session_id=100,
            streaming_startup_notify_kwarg='notify_add_msg_func',
        )
        assert await after_output_stream.__anext__() == 'wait-after-output: instance async stream infer'
        assert after_output_init.is_set()
        await after_output_stream.aclose()
        for _ in range(10):
            after_output_state = await client.async_call('stream_state')
            if after_output_state == (0, 0, 0) and client.pending == {}:
                break
            await asyncio.sleep(0.1)
        assert after_output_state == (0, 0, 0)
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


async def _async_test_mp_async_end_waits_for_stream_init_after_cancel():
    engine = _FakeMPEngine()
    instance = engine.create_instance()
    stream = instance.async_stream_infer(7)
    stream_task = asyncio.create_task(stream.__anext__())
    await asyncio.wait_for(engine.started.wait(), timeout=1)

    stream_task.cancel()
    with suppress(asyncio.CancelledError):
        await stream_task

    assert await instance.async_cancel(7) == ResponseType.SUCCESS
    end_task = asyncio.create_task(instance.async_end(7))
    await asyncio.sleep(0)
    assert not end_task.done()

    engine.allow_init.set()
    assert await asyncio.wait_for(end_task, timeout=1) == ResponseType.SUCCESS
    assert engine.calls == [('instance_async_cancel', (7, )), ('instance_async_end', (7, ))]
    assert 7 not in engine.session_states


def test_mp_async_end_waits_for_stream_init_after_cancel():
    asyncio.run(_async_test_mp_async_end_waits_for_stream_init_after_cancel())


async def _async_test_mp_cancel_during_stream_startup_is_forwarded():
    engine = _FakeMPEngine()
    instance = engine.create_instance()
    stream = instance.async_stream_infer(8)
    stream_task = asyncio.create_task(stream.__anext__())
    await asyncio.wait_for(engine.started.wait(), timeout=1)

    assert await instance.async_cancel(8) == ResponseType.SUCCESS
    assert engine.calls == []

    engine.allow_init.set()
    output = await asyncio.wait_for(stream_task, timeout=1)
    assert output.status == ResponseType.CANCEL
    for _ in range(10):
        if engine.calls:
            break
        await asyncio.sleep(0)
    assert engine.calls == [('instance_async_cancel', (8, ))]
    await stream.aclose()


def test_mp_cancel_during_stream_startup_is_forwarded():
    asyncio.run(_async_test_mp_cancel_during_stream_startup_is_forwarded())


async def _async_test_mp_cancel_before_stream_entry_stays_local():
    engine = _FakeMPEngine()
    instance = engine.create_instance()
    # Materialize the state as a request dispatcher would, but cancel before
    # async_stream_infer gets its first iteration and starts remote RPC.
    engine.session_states[9]

    assert await instance.async_cancel(9) == ResponseType.SUCCESS
    stream = instance.async_stream_infer(9)
    output = await asyncio.wait_for(stream.__anext__(), timeout=1)
    assert output.status == ResponseType.CANCEL
    await stream.aclose()
    await asyncio.sleep(0)

    assert engine.calls == []
    assert 9 not in engine.session_states


def test_mp_cancel_before_stream_entry_stays_local():
    asyncio.run(_async_test_mp_cancel_before_stream_entry_stays_local())


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


async def _async_test_zmq_pending_output_and_error_are_returned_together():
    from lmdeploy.pytorch.engine.mp_engine.zmq_rpc import AsyncRPCServer

    server = AsyncRPCServer()
    try:
        stream_id = 124
        output = EngineOutput(ResponseType.SUCCESS, [8, 9])
        stream_out = StreamMailbox()
        stream_out.publish(output)
        stream_out.fail(RuntimeError('failed after output'))
        server.stream_output[stream_id] = stream_out
        server._engine_output_gather.add(stream_id, output)

        poll_result = await server.get_stream_output(stream_id)
        assert poll_result.output.status == ResponseType.SUCCESS
        assert poll_result.output.token_ids == [8, 9]
        assert poll_result.has_output is True
        assert poll_result.done is True
        assert poll_result.error.type_name == 'builtins.RuntimeError'
        assert poll_result.error.message == 'failed after output'
        assert stream_id not in server.stream_output
        assert stream_id not in server._engine_output_gather._output
    finally:
        server.stop()
        server.socket.close(0)
        server.context.term()


def test_zmq_pending_output_and_error_are_returned_together():
    asyncio.run(_async_test_zmq_pending_output_and_error_are_returned_together())


async def _async_test_zmq_process_sentinel_marks_backend_dead():
    from lmdeploy.pytorch.engine.mp_engine.zmq_rpc import AsyncRPCClient

    ctx = mp.get_context('spawn')
    proc = ctx.Process(target=_sleep_then_exit)
    proc.start()
    assert proc.is_alive()

    marked = asyncio.Event()
    client = AsyncRPCClient(port=1,
                            server_alive_callback=proc.is_alive,
                            server_sentinel=proc.sentinel,
                            server_dead_callback=marked.set)
    try:
        client._register_server_sentinel()
        await asyncio.wait_for(marked.wait(), timeout=5)
    finally:
        client.stop()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
        proc.close()


def test_zmq_process_sentinel_marks_backend_dead():
    asyncio.run(_async_test_zmq_process_sentinel_marks_backend_dead())
