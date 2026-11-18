# Copyright (c) Shanghai AI Laboratory. All rights reserved.
"""Engine-level routing tests: the linear builders hand AWQ W4A16 modules
to the TurboMind GEMM family, per module, under the env semantics.

All cases run on tiny shapes; no real checkpoint is needed. The registry
needs a CUDA device, so registry-backed cases skip without one.
"""

import pytest
import torch

from lmdeploy import turbomind

if not turbomind.is_available():
    pytest.skip('TurboMind is not built', allow_module_level=True)

import lmdeploy.pytorch.envs as envs
from lmdeploy.pytorch.backends.cuda.awq_modules import AwqLinearW4A16Impl
from lmdeploy.pytorch.backends.cuda.tm_gemm_modules import TurbomindGemmW4A16Impl
from lmdeploy.pytorch.config import QuantizationConfig
from lmdeploy.pytorch.model_inputs import BuildModelContext
from lmdeploy.pytorch.models.patch import build_model_context
from lmdeploy.pytorch.nn.linear import (
    AwqLinear,
    MergedAwqLinear,
    QKVAwqLinear,
    build_linear,
    build_merged_colwise_linear,
    build_qkv_proj,
)
from lmdeploy.pytorch.nn.linear.tm_gemm import MergedTmGemmLinear, QKVTmGemmLinear, TmGemmLinear

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')

SUPPORTED = dict(in_features=512, out_features=256)
# N stays a multiple of 8 (AWQ packing keeps the module constructible), so
# the module builds. The test injects the rejection below because this
# registry accepts every shape the AWQ module family can construct on SM86.
FALLBACK = dict(in_features=512, out_features=4104)


def _reject_one_shape(monkeypatch):
    """Reject exactly the fallback shape at the probe the builders query."""
    import lmdeploy.pytorch.backends.cuda.tm_gemm_modules as tm_modules

    original = tm_modules.turbomind_support_reason

    def fake(in_features, out_features, w_bit, group_size, dtype=None):
        if (in_features, out_features) == (FALLBACK['in_features'], FALLBACK['out_features']):
            return 'injected: shape rejected by the test'
        return original(in_features, out_features, w_bit, group_size, dtype)

    monkeypatch.setattr(tm_modules, 'turbomind_support_reason', fake)


def _awq_quant_config() -> QuantizationConfig:
    return QuantizationConfig(quant_method='awq', bits=4, group_size=128)


def _set_backend(monkeypatch, value):
    monkeypatch.setattr(envs, 'w4a16_gemm_backend', value)
    # An exported LMDEPLOY_TM_KERNELS would win over the legacy attribute;
    # clear it so the tests exercise exactly the patched value.
    monkeypatch.setenv('LMDEPLOY_TM_KERNELS', '')


# ---------------------------------------------------------------------------
# Builder dispatch: auto routes every AWQ linear shape to the TM family.
# ---------------------------------------------------------------------------


@cuda_required
def test_auto_build_linear_routes_to_tm(monkeypatch):
    """build_linear returns a TurboMind module with the TM operator."""
    _set_backend(monkeypatch, 'auto')
    with build_model_context(BuildModelContext(quant_config=_awq_quant_config())):
        module = build_linear(512, 256, bias=False, quant_config={}, prefix='model.layers.0.mlp.down_proj')
    assert type(module) is TmGemmLinear
    assert isinstance(module.impl, TurbomindGemmW4A16Impl)
    assert module.tm_prefix == 'model.layers.0.mlp.down_proj'


@cuda_required
def test_auto_build_merged_routes_to_tm(monkeypatch):
    """build_merged_colwise_linear returns the merged TurboMind module."""
    _set_backend(monkeypatch, 'auto')
    with build_model_context(BuildModelContext(quant_config=_awq_quant_config())):
        module = build_merged_colwise_linear(512, [128, 128], bias=False, quant_config={},
                                             prefix='model.layers.0.mlp.gate_up_proj')
    assert type(module) is MergedTmGemmLinear
    assert isinstance(module.impl, TurbomindGemmW4A16Impl)


@cuda_required
def test_auto_build_qkv_routes_to_tm(monkeypatch):
    """build_qkv_proj returns the QKV TurboMind module."""
    _set_backend(monkeypatch, 'auto')
    with build_model_context(BuildModelContext(quant_config=_awq_quant_config())):
        module = build_qkv_proj(512,
                                num_q_heads=4,
                                num_kv_heads=4,
                                head_size=64,
                                quant_config={},
                                prefix='model.layers.0.self_attn.qkv_proj')
    assert type(module) is QKVTmGemmLinear
    assert isinstance(module.impl, TurbomindGemmW4A16Impl)


@cuda_required
def test_auto_falls_back_per_module(monkeypatch):
    """A shape the registry rejects keeps the TM module but swaps in the Triton
    operator, while a supported neighbor stays on TurboMind."""
    _set_backend(monkeypatch, 'auto')
    _reject_one_shape(monkeypatch)
    with build_model_context(BuildModelContext(quant_config=_awq_quant_config())):
        fallback = build_linear(FALLBACK['in_features'], FALLBACK['out_features'], bias=False, quant_config={})
        supported = build_linear(SUPPORTED['in_features'], SUPPORTED['out_features'], bias=False, quant_config={})
    assert type(fallback) is TmGemmLinear
    assert isinstance(fallback.impl, AwqLinearW4A16Impl)
    assert isinstance(supported.impl, TurbomindGemmW4A16Impl)


@cuda_required
def test_triton_env_keeps_historical_modules(monkeypatch):
    """The explicit triton value builds the historical AWQ modules."""
    _set_backend(monkeypatch, 'triton')
    with build_model_context(BuildModelContext(quant_config=_awq_quant_config())):
        plain = build_linear(512, 256, bias=False, quant_config={})
        merged = build_merged_colwise_linear(512, [128, 128], bias=False, quant_config={})
        qkv = build_qkv_proj(512, num_q_heads=4, num_kv_heads=4, head_size=64, quant_config={})
    assert type(plain) is AwqLinear
    assert type(merged) is MergedAwqLinear
    assert type(qkv) is QKVAwqLinear


@cuda_required
def test_forced_turbomind_raises_on_unsupported(monkeypatch):
    """Forced turbomind raises on a rejected layer instead of slowing down."""
    _set_backend(monkeypatch, 'turbomind')
    _reject_one_shape(monkeypatch)
    with build_model_context(BuildModelContext(quant_config=_awq_quant_config())):
        with pytest.raises(RuntimeError, match='unavailable or incompatible'):
            build_linear(FALLBACK['in_features'], FALLBACK['out_features'], bias=False, quant_config={})


@cuda_required
def test_routing_choice_is_logged_per_module(monkeypatch, caplog):
    """The per-module routing decision lands in the engine log."""
    import logging

    _set_backend(monkeypatch, 'auto')
    _reject_one_shape(monkeypatch)
    with caplog.at_level(logging.INFO, logger='lmdeploy'):
        with build_model_context(BuildModelContext(quant_config=_awq_quant_config())):
            build_linear(512,
                         256,
                         bias=False,
                         quant_config={},
                         prefix='model.layers.0.mlp.down_proj')
            build_linear(512,
                         4104,
                         bias=False,
                         quant_config={},
                         prefix='model.layers.1.mlp.down_proj')
    turbomind_lines = [m for m in caplog.messages if '-> turbomind' in m]
    fallback_lines = [m for m in caplog.messages if '-> awq triton' in m]
    assert any('model.layers.0.mlp.down_proj [512x256]' in m for m in turbomind_lines)
    assert any('model.layers.1.mlp.down_proj [512x4104]' in m and 'fallback' in m for m in fallback_lines)
