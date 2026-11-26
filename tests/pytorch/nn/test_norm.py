# Copyright (c) OpenMMLab. All rights reserved.
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

from lmdeploy.pytorch.config import QuantizationConfig
from lmdeploy.pytorch.devices import DeviceContext, get_device_manager
from lmdeploy.pytorch.model_inputs import BuildModelContext
from lmdeploy.pytorch.models.patch import build_model_context
from lmdeploy.pytorch.models.q_modules import QTensor
from lmdeploy.pytorch.nn import RMSNorm
from lmdeploy.pytorch.nn import norm as norm_module
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

DTYPES = [torch.bfloat16, torch.float16, torch.float32]
# 2**-9 fits BF16 and FP16, but 1 + 2**-9 needs more mantissa bits than BF16 has.
OFFSETS = [2**-9, 0.24609375, -1.0, 0.0, 0.0001, -0.37, 3.0, 2**-9]


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('checkpoint_dtype', DTYPES)
def test_offset_weight_scale_is_folded_in_fp32(dtype, checkpoint_dtype):
    norm = RMSNorm(len(OFFSETS), dtype=dtype, device='cpu', offset_weight=True)
    assert norm.weight.dtype == torch.float32

    offset = torch.tensor(OFFSETS, dtype=checkpoint_dtype)
    # Loading the same shard twice must not fold +1 twice.
    for _ in range(2):
        load_weight(norm.weight, offset)
        torch.testing.assert_close(norm.weight, offset.float() + 1.0, rtol=0, atol=0)
    if checkpoint_dtype == torch.bfloat16:
        assert norm.weight[0] != (offset + 1)[0]


def test_offset_weight_rejects_broadcast():
    norm = RMSNorm(4, device='cpu', offset_weight=True)
    with pytest.raises(AssertionError, match='Attempted to load weight'):
        load_weight(norm.weight, torch.tensor([0.25]))


def test_offset_weight_accepts_scalar_for_single_element():
    norm = RMSNorm(1, device='cpu', offset_weight=True)
    load_weight(norm.weight, torch.tensor(0.25))
    torch.testing.assert_close(norm.weight, torch.tensor([1.25]))


@pytest.mark.parametrize('dtype', DTYPES)
def test_direct_scale_keeps_dtype_and_value(dtype):
    norm = RMSNorm(len(OFFSETS), dtype=dtype, device='cpu')
    assert norm.weight.dtype == dtype
    assert not hasattr(norm.weight, 'weight_loader')

    scale = torch.tensor(OFFSETS, dtype=dtype)
    load_weight(norm.weight, scale)
    torch.testing.assert_close(norm.weight, scale, rtol=0, atol=0)


@pytest.mark.parametrize('rank', [0, 1])
@pytest.mark.parametrize('offset_weight', [False, True])
def test_tp_shard_composes_with_offset_weight(rank, offset_weight, monkeypatch):
    monkeypatch.setattr(norm_module, 'get_tp_world_rank', lambda layer_type=None: (2, rank))
    hidden_size = len(OFFSETS)
    norm = RMSNorm(hidden_size, dtype=torch.bfloat16, device='cpu', tp=True, offset_weight=offset_weight)
    assert norm.weight.shape == (hidden_size // 2, )

    loaded = torch.tensor(OFFSETS, dtype=torch.bfloat16)
    load_weight(norm.weight, loaded)
    shard = loaded.chunk(2)[rank]
    expected = shard.float() + 1.0 if offset_weight else shard
    assert norm.weight.dtype == expected.dtype
    torch.testing.assert_close(norm.weight, expected, rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
@pytest.mark.parametrize('checkpoint_dtype', DTYPES)
@pytest.mark.parametrize('with_residual', [False, True])
@pytest.mark.parametrize('smooth_quant', [False, True])
def test_ascend_offset_weight_uses_native_norm(dtype, checkpoint_dtype, with_residual, smooth_quant, monkeypatch):
    if 'torch_npu' not in sys.modules:
        monkeypatch.setitem(sys.modules, 'torch_npu', ModuleType('torch_npu'))
    x = torch.tensor([[1, 1, 1, 3]], dtype=dtype)
    residual = x / 2 if with_residual else None
    summed = x + residual if with_residual else x
    plain_op = Mock(return_value=(x, None))
    residual_op = Mock(return_value=(x, None, summed))
    quantized = x.to(torch.int8)
    scale = torch.ones(1)
    quant_op = Mock(return_value=(quantized, scale))
    monkeypatch.setattr(torch.ops.npu, 'npu_rms_norm', plain_op, raising=False)
    monkeypatch.setattr(torch.ops.npu, 'npu_add_rms_norm', residual_op, raising=False)
    monkeypatch.setattr(torch.ops.npu, 'npu_dynamic_quant', quant_op, raising=False)

    config = QuantizationConfig(quant_method='smooth_quant' if smooth_quant else None, quant_dtype=torch.int8)
    ctx = BuildModelContext(quant_config=config)
    with get_device_manager().context(DeviceContext('ascend')), build_model_context(ctx):
        norm = RMSNorm(x.size(-1), dtype=dtype, device='cpu', offset_weight=True, quant_config=config)
    offset = torch.tensor([2**-9, 0.24609375, -1.0, 0.0], dtype=checkpoint_dtype)
    load_weight(norm.weight, offset)
    torch.testing.assert_close(norm.weight, (offset + 1).to(dtype), rtol=0, atol=0)

    result = norm(x, residual)
    if with_residual:
        plain_op.assert_not_called()
        residual_op.assert_called_once_with(x, residual, norm.weight, norm.eps)
        result, result_residual = result
        assert result_residual is summed
    else:
        residual_op.assert_not_called()
        plain_op.assert_called_once_with(x, norm.weight, norm.eps)
    if smooth_quant:
        quant_op.assert_called_once_with(x, dst_type=torch.int8)
        assert isinstance(result, QTensor)
        assert result.tensor is quantized and result.scale is scale
    else:
        quant_op.assert_not_called()
        assert result is x
