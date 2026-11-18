# Copyright (c) Shanghai AI Laboratory. All rights reserved.
"""Import-guard tests for the TurboMind GEMM channel.

The channel modules import the native ``_turbomind`` extension at module
scope. A PyTorch-only installation (native extension not built) must keep
importing the model tree and fall back to the AWQ Triton kernels instead
of failing at import time — including with ``LMDEPLOY_TM_KERNELS=off`` or
a non-CUDA backend. These tests must run without the native build, so the
module deliberately does NOT skip when ``turbomind.is_available()`` is
False.
"""

import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')

_BLOCK_NATIVE = """\
import sys


class _BlockNative:
    \"\"\"Hide the native extension to simulate a PyTorch-only install.\"\"\"

    def find_spec(self, name, path=None, target=None):
        if name == 'lmdeploy.turbomind._turbomind':
            raise ImportError('blocked to simulate a PyTorch-only install')


sys.meta_path.insert(0, _BlockNative())
"""


def test_model_tree_imports_without_native_modules():
    """Importing an ordinary PyTorch model survives a missing native build."""
    script = _BLOCK_NATIVE + """
import lmdeploy.pytorch.backends.cuda.tm_gemm_modules
import lmdeploy.pytorch.nn.linear
import lmdeploy.pytorch.models.llama
print('IMPORTS OK')
"""
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, f'stderr:\n{result.stderr}'
    assert 'IMPORTS OK' in result.stdout


def test_channel_unavailable_update_weights_raises(monkeypatch):
    """Packing raises a clear error when the channel cannot be imported.

    The probe cannot catch a channel import failure (it queries the same native build), so the pack site must carry its
    own guard.
    """
    from lmdeploy.pytorch.backends.cuda import tm_gemm_modules

    error = RuntimeError('simulated channel import failure')
    monkeypatch.setattr(tm_gemm_modules, '_load_turbomind_channel', lambda: (None, error))
    impl = tm_gemm_modules.TurbomindGemmW4A16Impl(512, 256, 128)
    with pytest.raises(RuntimeError, match='TurboMind linear channel is unavailable'):
        impl.update_weights(torch.zeros(512, 32, dtype=torch.int32), torch.zeros(4, 256), torch.zeros(4, 32))


def test_builder_auto_falls_back_when_channel_import_fails(monkeypatch):
    """Auto routing falls back when the probe passes but the channel fails to
    import."""
    from lmdeploy.pytorch.backends.cuda import tm_gemm_modules
    from lmdeploy.pytorch.backends.cuda.awq_modules import AwqLinearW4A16Impl
    from lmdeploy.pytorch.backends.tm_gemm_modules import LinearTmGemmBuildSpec

    error = RuntimeError('simulated channel import failure')
    monkeypatch.setattr(tm_gemm_modules, '_load_turbomind_channel', lambda: (None, error))
    spec = LinearTmGemmBuildSpec(
        in_features=512, out_features=256, w_bit=4, group_size=128, bias=False, output_dtype=torch.float16
    )
    impl = tm_gemm_modules._build_linear_tm_gemm(spec)
    assert isinstance(impl, AwqLinearW4A16Impl)


def test_builder_forced_raises_when_channel_import_fails(monkeypatch):
    """Forced turbomind raises when the probe passes but the channel fails to
    import."""
    import lmdeploy.pytorch.envs as envs
    from lmdeploy.pytorch.backends.cuda import tm_gemm_modules
    from lmdeploy.pytorch.backends.tm_gemm_modules import LinearTmGemmBuildSpec

    error = RuntimeError('simulated channel import failure')
    monkeypatch.setattr(tm_gemm_modules, '_load_turbomind_channel', lambda: (None, error))
    monkeypatch.setattr(envs, 'w4a16_gemm_backend', 'turbomind')
    monkeypatch.setenv('LMDEPLOY_TM_KERNELS', '')
    spec = LinearTmGemmBuildSpec(
        in_features=512, out_features=256, w_bit=4, group_size=128, bias=False, output_dtype=torch.float16
    )
    with pytest.raises(RuntimeError, match='unavailable or incompatible'):
        tm_gemm_modules._build_linear_tm_gemm(spec)
