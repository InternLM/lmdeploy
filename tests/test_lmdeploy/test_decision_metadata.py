# Copyright (c) OpenMMLab. All rights reserved.
"""Read checkpoint temperature metadata without executing checkpoint code."""
from unittest.mock import Mock

import pytest

from lmdeploy.intern_decision_encoding import DEFAULT_TEMPERATURE, resolve_temperature


@pytest.mark.parametrize('assignment, expected', [
    ('DEFAULT_TEMPERATURE = 2.747760550703', 2.747760550703),
    ('DEFAULT_TEMPERATURE: float = 2.100509348278', 2.100509348278),
    ('DEFAULT_TEMPERATURE = 1.99241824', 1.99241824),
])
def test_checkpoint_temperature_without_executing_source(tmp_path, assignment, expected):
    (tmp_path / 'inference.py').write_text(
        'import package_that_must_not_be_loaded\n'
        'raise RuntimeError("checkpoint code must not execute")\n'
        '# DEFAULT_TEMPERATURE = 999\n'
        'def example():\n    DEFAULT_TEMPERATURE = 999\n' + assignment + '\n')
    assert resolve_temperature(tmp_path) == expected


@pytest.mark.parametrize('source', [
    None, b'\xff', 'def broken(', '', 'def example():\n    DEFAULT_TEMPERATURE = 3',
    'DEFAULT_TEMPERATURE = calculate_temperature()', 'DEFAULT_TEMPERATURE = True',
    'DEFAULT_TEMPERATURE = "2"', 'DEFAULT_TEMPERATURE = 0', 'DEFAULT_TEMPERATURE = -1',
    'DEFAULT_TEMPERATURE = 1e999', 'DEFAULT_TEMPERATURE: float',
    'DEFAULT_TEMPERATURE = 2\nDEFAULT_TEMPERATURE = 3',
])
def test_unreadable_temperature_warns_and_falls_back(tmp_path, monkeypatch, source):
    from lmdeploy import intern_decision_encoding as encoding

    if source is not None:
        (tmp_path / 'inference.py').write_bytes(source if isinstance(source, bytes) else source.encode())
    warning = Mock()
    monkeypatch.setattr(encoding.logger, 'warning', warning)
    assert resolve_temperature(tmp_path) == DEFAULT_TEMPERATURE
    warning.assert_called_once()
    message = warning.call_args.args[0]
    assert str(tmp_path / 'inference.py') in message
    assert str(DEFAULT_TEMPERATURE) in message


def test_remote_id_does_not_download_temperature(tmp_path, monkeypatch):
    from lmdeploy import intern_decision_encoding as encoding
    from lmdeploy import utils

    monkeypatch.chdir(tmp_path)
    download = Mock(side_effect=AssertionError('Temperature reading must not download models'))
    warning = Mock()
    monkeypatch.setattr(utils, 'get_model', download)
    monkeypatch.setattr(encoding.logger, 'warning', warning)
    assert resolve_temperature('organization/decision-model') == DEFAULT_TEMPERATURE
    download.assert_not_called()
    warning.assert_called_once()
