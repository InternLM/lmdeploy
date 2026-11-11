# Copyright (c) OpenMMLab. All rights reserved.
import signal
from types import SimpleNamespace
from unittest.mock import Mock

from lmdeploy.serve.openai.launch_server import terminate_processes


def test_group_signal_waits_for_frontend_shutdown(monkeypatch):
    killpg = Mock()
    monkeypatch.setattr('os.killpg', killpg)
    process = Mock(pid=123)
    process.is_alive.side_effect = [True, False]
    process.join.side_effect = lambda **kwargs: killpg.assert_not_called()
    terminate_processes([process])
    process.terminate.assert_called_once_with()
    process.join.assert_called_once_with(timeout=15)
    killpg.assert_called_once_with(123, signal.SIGTERM)


def test_stalled_frontend_still_kills_process_group(monkeypatch):
    killpg = Mock()
    monkeypatch.setattr('os.killpg', killpg)
    process = Mock(pid=123)
    process.is_alive.return_value = True
    terminate_processes([process])
    killpg.assert_called_once_with(123, signal.SIGKILL)


def test_dead_frontend_still_terminates_orphaned_group(monkeypatch):
    killpg = Mock()
    monkeypatch.setattr('os.killpg', killpg)
    process = Mock(pid=123)
    process.is_alive.return_value = False
    terminate_processes([process])
    process.terminate.assert_not_called()
    killpg.assert_called_once_with(123, signal.SIGTERM)


def test_dp_launch_respects_visible_gpu_subset(monkeypatch):
    import importlib
    module = importlib.import_module('lmdeploy.serve.openai.launch_server')
    process = Mock(return_value=Mock(exitcode=0))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '4,5,6,7')
    monkeypatch.setattr(module.mp, 'Process', process)
    monkeypatch.setattr(module.mp, 'set_start_method', Mock())
    monkeypatch.setattr(module.signal, 'signal', Mock())
    monkeypatch.setattr(module, 'get_host_ip', lambda: '127.0.0.1')
    monkeypatch.setattr(module, 'is_port_available', lambda _: True)
    module.launch_server(1, 0, 'model', SimpleNamespace(dp=4, tp=1, ep=4), server_port=19500)
    assert [call.kwargs['args'][0] for call in process.call_args_list] == [['4'], ['5'], ['6'], ['7']]
