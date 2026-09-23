# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn.functional as F

from lmdeploy.pytorch.nn.norm import FP32LayerNorm


@pytest.mark.parametrize('device', [
    'cpu', pytest.param('cuda', marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason='requires CUDA')),
])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('bias', [False, True])
def test_fp32_layer_norm_preserves_parameters_and_output_dtype(dtype, bias, device):
    norm = FP32LayerNorm(16, bias=bias, device=device)
    generator = torch.Generator().manual_seed(123)
    inputs = torch.randn(2, 3, 16, generator=generator).to(device=device, dtype=dtype)
    norm.weight.data.copy_(torch.randn(16, generator=generator))
    if bias:
        norm.bias.data.copy_(torch.randn(16, generator=generator))

    expected = F.layer_norm(inputs.float(), (16,), norm.weight, norm.bias, norm.eps).to(dtype)
    actual = norm(inputs)

    assert norm.weight.dtype == torch.float32
    assert norm.bias is None or norm.bias.dtype == torch.float32
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
