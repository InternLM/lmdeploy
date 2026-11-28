# Copyright (c) OpenMMLab. All rights reserved.
import os
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from packaging.requirements import Requirement


@pytest.mark.parametrize('cuda_major', [12, 13])
def test_turbomind_wheel_dependencies(monkeypatch, cuda_major):
    monkeypatch.setenv('LMDEPLOY_TARGET_DEVICE', 'cuda')
    monkeypatch.delenv('DISABLE_TURBOMIND', raising=False)
    monkeypatch.setitem(sys.modules, 'cmake_build_extension', Mock())
    monkeypatch.setenv('CUDACXX', 'test-nvcc')
    check_output = subprocess.check_output

    def compiler_output(args, **kwargs):
        if args == ['test-nvcc', '--version']:
            return f'release {cuda_major}.0'.encode()
        return check_output(args, **kwargs)

    monkeypatch.setattr(subprocess, 'check_output', compiler_output)
    config = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'setup.py'))
    runtime = [Requirement(req) for req in config['extra_deps']]
    suffix = '-cu12' if cuda_major == 12 else ''
    assert {req.name for req in runtime} == {
        f'nvidia-nccl-cu{cuda_major}',
        f'nvidia-cuda-runtime{suffix}',
        f'nvidia-cublas{suffix}',
        f'nvidia-curand{suffix}',
    }
    for req in runtime:
        assert req.marker is not None
        assert req.marker.evaluate({'sys_platform': 'linux'})
        assert not req.marker.evaluate({'sys_platform': 'win32'})

    build = [Requirement(req).name for req in config['extra_setup_deps']]
    assert build == ([] if os.name == 'nt' else [f'nvidia-nccl-cu{cuda_major}'])
