# Copyright (c) OpenMMLab. All rights reserved.
import signal
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
