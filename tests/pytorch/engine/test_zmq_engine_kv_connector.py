# Copyright (c) OpenMMLab. All rights reserved.
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
