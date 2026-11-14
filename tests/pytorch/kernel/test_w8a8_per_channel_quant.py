import importlib.util
from pathlib import Path

import pytest
import torch

MODULE_PATH = Path(__file__).resolve().parents[3] / 'lmdeploy/pytorch/kernels/default/w8a8_kernels.py'
SPEC = importlib.util.spec_from_file_location('w8a8_kernels', MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize('dtype', [torch.int8, torch.float8_e4m3fn])
def test_per_channel_quant_zero_row_has_finite_dequantization(dtype):
    x = torch.tensor([[0.0, 0.0, 0.0], [0.5, -1.0, 0.25]])

    quantized, scale = MODULE.per_channel_quant(x, dtype)
    restored = quantized.float() * scale

    assert torch.isfinite(quantized.float()).all()
    assert torch.isfinite(restored).all()
    assert (scale > 0).all()
    torch.testing.assert_close(restored[0], x[0], rtol=0, atol=0)
    torch.testing.assert_close(restored[1], x[1], rtol=0.02, atol=0.02)
