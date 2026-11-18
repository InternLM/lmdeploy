# Copyright (c) Shanghai AI Laboratory. All rights reserved.
"""Routing tests for the W4A16 GEMM backend selection.

The probe queries the TurboMind kernel registry. The registry needs a CUDA device, so every registry-backed case skips
without one.
"""

import pytest
import torch

import lmdeploy.pytorch.envs as envs
from lmdeploy import turbomind

if not turbomind.is_available():
    pytest.skip('TurboMind is not built', allow_module_level=True)

from lmdeploy.pytorch.backends.awq_modules import LinearW4A16BuildSpec
from lmdeploy.pytorch.backends.cuda.awq_modules import (
    AwqLinearW4A16Impl,
    _build_linear_w4a16,
)
from lmdeploy.pytorch.backends.cuda.tm_gemm_modules import TurbomindGemmW4A16Impl
from lmdeploy.pytorch.backends.cuda.tm_probe import turbomind_support_reason

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')

SUPPORTED = dict(in_features=4096, out_features=4096, w_bit=4, group_size=128)
UNSUPPORTED = dict(in_features=4096, out_features=4094, w_bit=4, group_size=128)


def _spec(**kwargs):
    params = dict(bias=False, output_dtype=torch.float16)
    params.update(kwargs)
    return LinearW4A16BuildSpec(**params)


def _set_backend(monkeypatch, value):
    monkeypatch.setattr(envs, 'w4a16_gemm_backend', value)
    # An exported LMDEPLOY_TM_KERNELS would win over the legacy attribute;
    # clear it so the tests exercise exactly the patched value.
    monkeypatch.setenv('LMDEPLOY_TM_KERNELS', '')


# ---------------------------------------------------------------------------
# Env three-state matrix: auto / turbomind / other, supported vs unsupported.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('backend', ['auto', 'turbomind'])
@cuda_required
def test_supported_shape_selects_turbomind(monkeypatch, backend):
    """A registry-supported layer picks TurboMind under auto and forced."""
    _set_backend(monkeypatch, backend)
    impl = _build_linear_w4a16(_spec(**SUPPORTED))
    assert isinstance(impl, TurbomindGemmW4A16Impl)


@cuda_required
def test_auto_falls_back_to_triton_on_unsupported_shape(monkeypatch):
    """Auto falls back per layer when the registry rejects the shape."""
    _set_backend(monkeypatch, 'auto')
    impl = _build_linear_w4a16(_spec(**UNSUPPORTED))
    assert isinstance(impl, AwqLinearW4A16Impl)


@cuda_required
def test_forced_turbomind_raises_on_unsupported_shape(monkeypatch):
    """Forced turbomind raises a readable RuntimeError on rejection."""
    _set_backend(monkeypatch, 'turbomind')
    with pytest.raises(RuntimeError, match='unavailable or incompatible'):
        _build_linear_w4a16(_spec(**UNSUPPORTED))


@pytest.mark.parametrize('backend', ['triton'])
@pytest.mark.parametrize('shape', [SUPPORTED, UNSUPPORTED], ids=['supported', 'unsupported'])
def test_other_values_always_select_triton(monkeypatch, backend, shape):
    """Any non-TM value keeps the historical all-Triton behavior."""
    _set_backend(monkeypatch, backend)
    impl = _build_linear_w4a16(_spec(**shape))
    assert isinstance(impl, AwqLinearW4A16Impl)


# ---------------------------------------------------------------------------
# Probe rejection table: each rejection carries a readable reason.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'kwargs, fragment',
    [
        (dict(w_bit=8), 'w_bit must be 4'),
        (dict(group_size=0), 'group_size must be positive'),
    ],
    ids=['w_bit', 'group_size'],
)
def test_probe_rejects_without_device(kwargs, fragment, monkeypatch):
    """Cheap preconditions reject before any device access."""
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    reason = turbomind_support_reason(dtype=torch.float16, **{**SUPPORTED, **kwargs})
    assert reason is not None
    assert fragment in reason


@cuda_required
@pytest.mark.parametrize(
    'kwargs, fragment',
    [
        (dict(in_features=4094), 'K must be a multiple of'),
        (dict(out_features=4094), 'N must be a multiple of'),
        (dict(dtype=torch.float32), 'no GEMM family accepts'),
        (dict(dtype=torch.bfloat16), 'no GEMM family accepts'),
    ],
    ids=['align_k', 'align_n', 'dtype_fp32', 'dtype_bf16'],
)
def test_probe_registry_rejections(kwargs, fragment):
    """The registry rejects unsupported combos with a readable reason."""
    reason = turbomind_support_reason(**{**SUPPORTED, **kwargs})
    assert reason is not None
    assert fragment in reason


@cuda_required
def test_probe_accepts_supported_layer():
    """A supported layer yields no reason on this device."""
    assert turbomind_support_reason(**SUPPORTED, dtype=torch.float16) is None


@cuda_required
def test_probe_cuda_unavailable(monkeypatch):
    """Missing CUDA reports a readable reason."""
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    reason = turbomind_support_reason(**SUPPORTED, dtype=torch.float16)
    assert reason == 'CUDA is unavailable'
