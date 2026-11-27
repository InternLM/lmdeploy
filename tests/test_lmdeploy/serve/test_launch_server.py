# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import signal
import socket
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from lmdeploy.messages import PytorchEngineConfig
from lmdeploy.serve.openai.launch_server import is_port_available, terminate_processes


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
    monkeypatch.setattr(module, 'is_port_available', lambda *_: True)
    module.launch_server(1, 0, 'model', SimpleNamespace(dp=4, tp=1, ep=4), server_port=19500)
    assert [call.kwargs['args'][0] for call in process.call_args_list] == [['4'], ['5'], ['6'], ['7']]


@pytest.fixture
def launcher(monkeypatch):
    module = importlib.import_module('lmdeploy.serve.openai.launch_server')
    # Port planning uses mocked workers, independent of the runner's GPU allocation.
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.setattr(module.mp, 'set_start_method', Mock())
    monkeypatch.setattr(module.signal, 'signal', Mock())
    monkeypatch.setattr(module, 'get_host_ip', lambda: '127.0.0.2')
    monkeypatch.setattr(module, 'is_port_available', Mock(return_value=True))
    process = Mock(return_value=Mock(exitcode=0))
    monkeypatch.setattr(module.mp, 'Process', process)
    return module


@pytest.mark.parametrize('monitoring_port', [None, 9300])
def test_monitoring_ports_follow_local_dp_ranks(launcher, monitoring_port):
    launcher.launch_server(2, 1, '/unused', PytorchEngineConfig(dp=8, ep=8), server_port=9200,
                           monitoring_port=monitoring_port)
    calls = launcher.mp.Process.call_args_list
    assert len(calls) == 4
    for rank, process_call in enumerate(calls):
        kwargs = process_call.kwargs['kwargs']
        assert kwargs['server_port'] == 9200 + rank
        assert kwargs['backend_config'].dp_rank == 4 + rank
        if monitoring_port is None:
            assert 'monitoring_port' not in kwargs
        else:
            assert kwargs['monitoring_port'] == 9300 + rank


@pytest.mark.parametrize('port', [0, -1, 65535, 9200, 9207, 9193])
def test_invalid_monitoring_port_range_fails_before_spawning(launcher, port):
    with pytest.raises(ValueError, match='[Mm]onitoring'):
        launcher.launch_server(1, 0, '/unused', PytorchEngineConfig(dp=8, ep=8),
                               server_port=9200, monitoring_port=port)
    launcher.mp.Process.assert_not_called()


def test_occupied_monitoring_port_fails_before_spawning(launcher):
    launcher.is_port_available.side_effect = lambda port, host: port != 9303
    with pytest.raises(ValueError, match='Monitoring port 9303 is not available'):
        launcher.launch_server(1, 0, '/unused', PytorchEngineConfig(dp=8, ep=8),
                               server_port=9200, monitoring_port=9300)
    launcher.mp.Process.assert_not_called()
    launcher.is_port_available.assert_any_call(9303, '127.0.0.2')


def test_port_availability_uses_the_child_bind_address():
    for host in ('127.0.0.1', '127.0.0.2'):
        with socket.create_server((host, 0)) as occupied:
            port = occupied.getsockname()[1]
            assert not is_port_available(port, host)
            other_host = '127.0.0.2' if host == '127.0.0.1' else '127.0.0.1'
            assert is_port_available(port, other_host)


def test_automatic_api_ports_skip_monitoring_ports(launcher, monkeypatch):
    monkeypatch.setattr(launcher.random, 'randint', lambda *_: 100)
    launcher.launch_server(1, 0, '/unused', PytorchEngineConfig(dp=2, ep=2),
                           proxy_url='http://router', monitoring_port=3100)
    calls = launcher.mp.Process.call_args_list
    assert [call.kwargs['kwargs']['server_port'] for call in calls] == [3200, 3300]
    assert [call.kwargs['kwargs']['monitoring_port'] for call in calls] == [3100, 3101]
    launcher.is_port_available.assert_has_calls([call(3200, '127.0.0.2'), call(3300, '127.0.0.2')])
