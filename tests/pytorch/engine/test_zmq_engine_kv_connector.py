# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from lmdeploy.messages import KVTransferConfig, PytorchEngineConfig
from lmdeploy.pytorch.engine.mp_engine.zmq_engine import ZMQMPEngine


def test_zmq_engine_extends_shutdown_timeout_for_kv_connector():
    disabled_config = PytorchEngineConfig()
    enabled_config = PytorchEngineConfig(
        kv_transfer_config=KVTransferConfig(
            kv_connector='MooncakeStoreConnector',
            kv_role='kv_both',
        ))

    assert ZMQMPEngine._get_process_shutdown_timeout(None) == 10
    assert ZMQMPEngine._get_process_shutdown_timeout(disabled_config) == 10
    assert ZMQMPEngine._get_process_shutdown_timeout(enabled_config) == 60


def test_zmq_engine_close_uses_configured_shutdown_timeout():

    class _RPCClient:

        def __init__(self):
            self.stopped = False

        def stop(self):
            self.stopped = True

    class _Process:

        def __init__(self):
            self.join_timeout = None
            self.terminated = False
            self.closed = False

        def terminate(self):
            self.terminated = True

        def join(self, timeout=None):
            self.join_timeout = timeout

        def is_alive(self):
            return False

        def close(self):
            self.closed = True

    engine = ZMQMPEngine.__new__(ZMQMPEngine)
    engine.rpc_client = _RPCClient()
    engine.proc = _Process()
    engine._process_shutdown_timeout = 60
    process = engine.proc

    engine.close()

    assert engine.rpc_client.stopped
    assert process.terminated
    assert process.join_timeout == 60
    assert process.closed
    assert engine.proc is None


def test_zmq_engine_close_reaps_force_killed_process():

    class _RPCClient:

        def stop(self):
            pass

    class _Process:

        def __init__(self):
            self.alive = True
            self.join_timeouts = []
            self.killed = False
            self.closed = False

        def terminate(self):
            pass

        def join(self, timeout=None):
            self.join_timeouts.append(timeout)
            if self.killed:
                self.alive = False

        def is_alive(self):
            return self.alive

        def kill(self):
            self.killed = True

        def close(self):
            self.closed = True

    engine = ZMQMPEngine.__new__(ZMQMPEngine)
    engine.rpc_client = _RPCClient()
    engine.proc = _Process()
    engine._process_shutdown_timeout = 60
    process = engine.proc

    engine.close()

    assert process.killed
    assert process.join_timeouts == [60, None]
    assert process.closed
    assert engine.proc is None


@pytest.mark.parametrize('shutdown_reason', ['signal', 'return', 'error'])
@pytest.mark.parametrize('drain_fails', [False, True])
def test_shutdown_signal_does_not_cancel_engine_drain(monkeypatch, shutdown_reason, drain_fails):
    import asyncio
    import signal
    from types import SimpleNamespace
    from unittest.mock import Mock

    from lmdeploy.pytorch.engine.mp_engine import zmq_engine

    log_error = Mock(wraps=zmq_engine.logger.exception)
    monkeypatch.setattr(zmq_engine.logger, 'exception', log_error)

    async def run():
        handlers = {}
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(loop, 'add_signal_handler', lambda sig, callback: handlers.update({sig: callback}))
        draining = asyncio.Event()
        finish = asyncio.Event()
        events = []

        async def serve():
            if shutdown_reason == 'signal':
                await asyncio.Event().wait()
            elif shutdown_reason == 'error':
                raise RuntimeError('RPC failed')

        async def wait():
            draining.set()
            await finish.wait()
            if drain_fails:
                raise RuntimeError('drain failed')
            events.append('drained')

        server = SimpleNamespace(register_method=Mock(), run=serve, stop=Mock())
        engine = SimpleNamespace(start_loop=Mock(), close=Mock(), wait_tasks=wait,
                                 engine_config=SimpleNamespace(max_batch_size=1))
        task = asyncio.create_task(ZMQMPEngine._mp_proc_async(server, engine))
        await asyncio.sleep(0)
        if shutdown_reason == 'signal':
            handlers[signal.SIGTERM]()
            handlers[signal.SIGTERM]()
        await asyncio.wait_for(draining.wait(), timeout=2)
        handlers[signal.SIGTERM]()
        handlers[signal.SIGINT]()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not task.done()
        finish.set()
        await task
        assert events == ([] if drain_fails else ['drained'])
        assert log_error.call_count == int(shutdown_reason == 'error') + int(drain_fails)
        if drain_fails:
            log_error.assert_any_call('Engine wait_tasks failed during shutdown.')
        engine.close.assert_called_once()

    asyncio.run(run())
