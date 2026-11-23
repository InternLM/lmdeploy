# Copyright (c) OpenMMLab. All rights reserved.
"""Gated norm ownership, backend contracts, and CUDA compatibility with FLA."""
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.backends.norm import RMSNormGatedBuildSpec, RMSNormGatedImpl
from lmdeploy.pytorch.devices import DeviceContext, get_device_manager
from lmdeploy.pytorch.nn import RMSNormGated


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('activation', ['silu', 'swish', 'sigmoid'])
@pytest.mark.parametrize('affine', [True, False])
def test_public_weight_ownership(dtype, activation, affine):
    with get_device_manager().context(DeviceContext('cuda')):
        layer = RMSNormGated(128, activation=activation, dtype=dtype, device='cpu', elementwise_affine=affine)
    assert isinstance(layer.impl, RMSNormGatedImpl)
    assert type(layer.impl).__module__ == 'lmdeploy.pytorch.backends.cuda.norm'
    assert not isinstance(layer.impl, torch.nn.Module)
    assert dict(layer.named_children()) == {}
    assert set(layer.state_dict()) == ({'weight'} if affine else set())
    assert layer.bias is None
    if affine:
        assert layer.weight.dtype == dtype
        assert torch.equal(layer.weight, torch.ones(128, dtype=dtype))
        scale = torch.linspace(-1, 1, 128, dtype=dtype)
        layer.load_state_dict({'weight': scale}, strict=True)
        assert torch.equal(layer.weight, scale)
        layer.reset_parameters()
        assert torch.equal(layer.weight, torch.ones_like(scale))
    else:
        assert layer.weight is None


@pytest.mark.parametrize('activation', ['silu', 'swish', 'sigmoid'])
def test_fla_forwarding_and_lazy_import(monkeypatch, activation):
    calls = []
    keys = ['D', 'NB']
    module = ModuleType('fla.modules.fused_norm_gate')

    def forward(x, z, weight, bias, gate_activation, **kwargs):
        calls.append((x, z, weight, bias, gate_activation, kwargs))
        return x, kwargs['residual']

    module.rms_norm_gated = forward
    module.layer_norm_gated_fwd_kernel = SimpleNamespace(fn=SimpleNamespace(keys=keys))
    with get_device_manager().context(DeviceContext('cuda')):
        layer = RMSNormGated(8, eps=1e-5, activation=activation, device='cpu')
    assert layer.impl._fla_forward is None
    monkeypatch.setitem(sys.modules, 'fla.modules.fused_norm_gate', module)
    x, z, residual = (torch.randn(2, 8) for _ in range(3))
    result = layer(x, z, residual, prenorm=True, residual_in_fp32=True)
    assert result[0] is x and result[1] is residual
    args = calls[0]
    assert args[0] is x and args[1] is z and args[2] is layer.weight and args[3] is None
    assert args[4] == activation
    assert args[5] == dict(residual=residual, eps=1e-5, prenorm=True, residual_in_fp32=True)
    assert keys == ['D']
    assert layer.impl._fla_forward is forward


@pytest.mark.parametrize('activation', ['relu', 'gelu', 'unknown'])
def test_invalid_activation(activation):
    with pytest.raises(ValueError, match='Unsupported gated RMSNorm activation'):
        RMSNormGatedBuildSpec(128, activation=activation)


@pytest.mark.parametrize('hidden_size', [0, -1])
def test_invalid_width(hidden_size):
    with pytest.raises(ValueError, match='hidden_size must be positive'):
        RMSNormGatedBuildSpec(hidden_size)


def test_default_keeps_existing_fla_fallback():
    from lmdeploy.pytorch.backends.default.op_backend import DefaultOpsBackend

    impl = DefaultOpsBackend.build_op(RMSNormGatedBuildSpec(128, activation='sigmoid'))
    assert isinstance(impl, RMSNormGatedImpl)
    assert type(impl).__module__ == 'lmdeploy.pytorch.backends.default.norm'
    assert impl.activation == 'sigmoid'
    assert impl._fla_forward is None


def _assert_same_bits(actual, expected):
    if isinstance(actual, tuple):
        assert isinstance(expected, tuple) and len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_same_bits(left, right)
        return
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


CUDA_AVAILABLE = torch.version.cuda is not None and torch.cuda.is_available()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason='Requires NVIDIA CUDA for the real FLA kernel')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('activation', ['silu', 'swish', 'sigmoid'])
@pytest.mark.parametrize('case', ['decode', 'prefill', 'strided', 'non_affine', 'residual', 'fp32_residual'])
@torch.inference_mode()
def test_cuda_matches_original_fla(dtype, activation, case):
    from fla.modules import FusedRMSNormGated
    from fla.modules.fused_norm_gate import layer_norm_gated_fwd_kernel

    # Match the original LMDeploy factory before running either implementation.
    keys = layer_norm_gated_fwd_kernel.fn.keys
    if 'NB' in keys:
        keys.remove('NB')
    torch.manual_seed(137)
    shape = (1, 128) if case == 'decode' else (2, 17, 128)
    affine = case != 'non_affine'
    original = FusedRMSNormGated(128, eps=1e-6, activation=activation,
                               elementwise_affine=affine, dtype=dtype, device='cuda')
    with get_device_manager().context(DeviceContext('cuda')):
        layer = RMSNormGated(128, activation=activation, elementwise_affine=affine, dtype=dtype, device='cuda')
    if affine:
        original.weight.copy_(torch.randn_like(original.weight))
        layer.load_state_dict(original.state_dict(), strict=True)
    x, z = (torch.randn(shape, dtype=dtype, device='cuda') for _ in range(2))
    if case == 'strided':
        x, z = (torch.randn((*shape[:-1], shape[-1] * 2), dtype=dtype, device='cuda')[..., ::2]
                for _ in range(2))
    kwargs = {}
    if case in ('residual', 'fp32_residual'):
        residual_dtype = torch.float32 if case == 'fp32_residual' else dtype
        kwargs = dict(residual=torch.randn(shape, dtype=residual_dtype, device='cuda'),
                      prenorm=True, residual_in_fp32=True)
    _assert_same_bits(layer(x, z, **kwargs), original(x, z, **kwargs))


@pytest.mark.skipif(not CUDA_AVAILABLE, reason='Requires NVIDIA CUDA graph capture')
@pytest.mark.parametrize('activation', ['silu', 'sigmoid'])
@torch.inference_mode()
def test_cuda_graph_reuses_public_weight(activation):
    from fla.modules import FusedRMSNormGated

    with get_device_manager().context(DeviceContext('cuda')):
        layer = RMSNormGated(128, activation=activation, dtype=torch.bfloat16, device='cuda')
    original = FusedRMSNormGated(128, eps=1e-6, activation=activation, dtype=torch.bfloat16, device='cuda')
    x, z = (torch.randn(4, 128, dtype=torch.bfloat16, device='cuda') for _ in range(2))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            layer(x, z)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = layer(x, z)
    for scale in (0.25, -0.75):
        x.normal_()
        z.normal_()
        layer.weight.fill_(scale)
        original.weight.copy_(layer.weight)
        graph.replay()
        _assert_same_bits(output, original(x, z))
