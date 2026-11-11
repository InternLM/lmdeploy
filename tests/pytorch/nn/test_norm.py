# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from lmdeploy.pytorch.nn import RMSNorm
from lmdeploy.pytorch.nn import norm as norm_module
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

DTYPES = [torch.bfloat16, torch.float16, torch.float32]
# 2**-9 fits BF16 and FP16, but 1 + 2**-9 needs more mantissa bits than BF16 has.
OFFSETS = [2**-9, 0.24609375, -1.0, 0.0, 0.0001, -0.37, 3.0, 2**-9]


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('checkpoint_dtype', DTYPES)
def test_zero_centered_scale_is_folded_in_fp32(dtype, checkpoint_dtype):
    norm = RMSNorm(len(OFFSETS), dtype=dtype, device='cpu', zero_centered=True)
    assert norm.weight.dtype == torch.float32

    offset = torch.tensor(OFFSETS, dtype=checkpoint_dtype)
    # Loading the same shard twice must not fold +1 twice.
    for _ in range(2):
        load_weight(norm.weight, offset)
        torch.testing.assert_close(norm.weight, offset.float() + 1.0, rtol=0, atol=0)
    if checkpoint_dtype == torch.bfloat16:
        assert norm.weight[0] != (offset + 1)[0]


@pytest.mark.parametrize('dtype', DTYPES)
def test_direct_scale_keeps_dtype_and_value(dtype):
    norm = RMSNorm(len(OFFSETS), dtype=dtype, device='cpu')
    assert norm.weight.dtype == dtype
    assert not hasattr(norm.weight, 'weight_loader')

    scale = torch.tensor(OFFSETS, dtype=dtype)
    load_weight(norm.weight, scale)
    torch.testing.assert_close(norm.weight, scale, rtol=0, atol=0)


@pytest.mark.parametrize('rank', [0, 1])
@pytest.mark.parametrize('zero_centered', [False, True])
def test_tp_shard_composes_with_zero_centered(rank, zero_centered, monkeypatch):
    monkeypatch.setattr(norm_module, 'get_tp_world_rank', lambda layer_type=None: (2, rank))
    hidden_size = len(OFFSETS)
    norm = RMSNorm(hidden_size, dtype=torch.bfloat16, device='cpu', tp=True, zero_centered=zero_centered)
    assert norm.weight.shape == (hidden_size // 2, )

    loaded = torch.tensor(OFFSETS, dtype=torch.bfloat16)
    load_weight(norm.weight, loaded)
    shard = loaded.chunk(2)[rank]
    expected = shard.float() + 1.0 if zero_centered else shard
    assert norm.weight.dtype == expected.dtype
    torch.testing.assert_close(norm.weight, expected, rtol=0, atol=0)
