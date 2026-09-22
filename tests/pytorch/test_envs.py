import importlib.util
import os
from pathlib import Path

import pytest


def _load_envs_module():
    module_path = Path(__file__).parents[2] / 'lmdeploy' / 'pytorch' / 'envs.py'
    spec = importlib.util.spec_from_file_location('lmdeploy.pytorch.envs', module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_set_envs_restores_getenv_after_exception():
    set_envs = _load_envs_module().set_envs
    original_getenv = os.getenv

    with pytest.raises(RuntimeError):
        with set_envs():
            assert os.getenv is not original_getenv
            raise RuntimeError

    assert os.getenv is original_getenv
