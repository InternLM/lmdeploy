# Copyright (c) OpenMMLab. All rights reserved.
"""Real Ascend arithmetic and graph replay for the shared gated norm layer."""
import pytest
import torch

from lmdeploy.pytorch.devices import DeviceContext, get_device_manager
from lmdeploy.pytorch.nn import RMSNormGated


def _reference(x, weight, z, eps=1e-6, activation='silu', bias=None, group_size=None, norm_before_gate=True):
    value = x.float()
    gate = None if z is None else torch.sigmoid(z.float())
    if gate is not None and activation != 'sigmoid':
        gate = z.float() * gate
    if gate is not None and not norm_before_gate:
        value = value * gate
    grouped = value.reshape(*value.shape[:-1], -1, group_size or value.shape[-1])
    value = (grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + eps)).reshape_as(value)
    if weight is not None:
        value = value * weight.float()
    if bias is not None:
        value = value + bias.float()
    if gate is not None and norm_before_gate:
        value = value * gate
    return value.to(x.dtype)


def _tolerance(dtype):
    return {torch.bfloat16: 0.02, torch.float16: 0.003, torch.float32: 1e-5}[dtype]


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize('activation', ['silu', 'swish', 'sigmoid'])
@pytest.mark.parametrize('case', ['decode', 'prefill', 'strided', 'non_affine', 'residual'])
@torch.inference_mode()
def test_gated_norm_reference(dtype, activation, case):
    torch.manual_seed(137)
    shape = (1, 128) if case == 'decode' else (2, 17, 128)
    with get_device_manager().context(DeviceContext('ascend')):
        layer = RMSNormGated(128, activation=activation, dtype=dtype, device='npu',
                            elementwise_affine=case != 'non_affine')
    assert layer.weight is None or layer.weight.dtype == dtype
    if layer.weight is not None:
        layer.weight.copy_(torch.randn_like(layer.weight))
    x, z = (torch.randn(shape, dtype=dtype, device='npu') for _ in range(2))
    if case == 'strided':
        x, z = (torch.randn((*shape[:-1], shape[-1] * 2), dtype=dtype, device='npu')[..., ::2]
                for _ in range(2))
    residual = torch.randn_like(x, dtype=torch.float32) if case == 'residual' else None
    source = x if residual is None else x.float() + residual
    expected = _reference(source, layer.weight, z, activation=activation).to(dtype)
    actual = layer(x, z, residual=residual, prenorm=True, residual_in_fp32=True)
    tol = _tolerance(dtype)
    torch.testing.assert_close(actual[0], expected, rtol=tol, atol=tol)
    torch.testing.assert_close(actual[1], source.float(), rtol=0, atol=0)


@pytest.mark.parametrize('activation', ['silu', 'sigmoid'])
@pytest.mark.parametrize('norm_before_gate', [False, True])
@pytest.mark.parametrize('with_gate', [False, True])
@torch.inference_mode()
def test_kernel_grouping_bias_and_optional_gate(activation, norm_before_gate, with_gate):
    from lmdeploy.pytorch.kernels.ascend.triton_ops.rms_norm_gated import rmsnorm_fn

    torch.manual_seed(137)
    x = torch.randn(17, 192, device='npu', dtype=torch.bfloat16)
    z = torch.randn_like(x) if with_gate else None
    weight, bias = (torch.randn(384, device='npu', dtype=torch.bfloat16)[::2] for _ in range(2))
    kwargs = dict(group_size=64, norm_before_gate=norm_before_gate, activation=activation)
    expected = _reference(x, weight, z, bias=bias, **kwargs)
    actual = rmsnorm_fn(x, weight, bias, z=z, **kwargs)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)


@pytest.mark.parametrize('activation', ['silu', 'sigmoid'])
@torch.inference_mode()
def test_npu_graph_reuses_public_weight(activation):
    with get_device_manager().context(DeviceContext('ascend')):
        layer = RMSNormGated(128, activation=activation, device='npu', dtype=torch.bfloat16)
    x, z = (torch.randn(4, 128, device='npu', dtype=torch.bfloat16) for _ in range(2))
    for _ in range(3):
        layer(x, z)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = layer(x, z)
    for scale in (0.25, -0.75):
        x.normal_()
        z.normal_()
        layer.weight.fill_(scale)
        graph.replay()
        expected = layer(x, z)
        assert torch.equal(output.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
