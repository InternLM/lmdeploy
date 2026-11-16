from __future__ import annotations

import pytest
import torch

from lmdeploy import turbomind

if not turbomind.is_available():
    pytest.skip('TurboMind is not built', allow_module_level=True)

from lmdeploy.turbomind import _tm
from lmdeploy.turbomind.weight_format import NVFP4Format


def _valid_tensors(out: int = 8, inp: int = 32):
    return {
        '.weight_packed': torch.zeros((out, inp // 2), dtype=torch.uint8),
        '.weight_scale': torch.zeros((out, inp // 16), dtype=torch.float8_e4m3fn),
        '.weight_global_scale': torch.ones((1, ), dtype=torch.float32),
    }


def test_nvfp4_format_contract():
    fmt = NVFP4Format()
    assert fmt.name == 'nvfp4'
    assert fmt.block_in == 16
    assert fmt.block_out is None
    assert fmt.weight_dtype == _tm.DataType.TYPE_FP4_E2M1
    assert fmt.scales_dtype == _tm.DataType.TYPE_FP16
    assert fmt.zeros_dtype == _tm.DataType.TYPE_INVALID

    with pytest.raises(ValueError, match='group_size 16'):
        NVFP4Format(block_in=32)


def test_nvfp4_format_accepts():
    fmt = NVFP4Format()
    assert fmt.accepts(_valid_tensors())

    missing_global = _valid_tensors()
    missing_global.pop('.weight_global_scale')
    assert not fmt.accepts(missing_global)

    wrong_weight = _valid_tensors()
    wrong_weight['.weight_packed'] = torch.zeros((8, 16), dtype=torch.int32)
    assert not fmt.accepts(wrong_weight)

    wrong_scales = _valid_tensors()
    wrong_scales['.weight_scale'] = torch.zeros((8, 2), dtype=torch.float16)
    assert not fmt.accepts(wrong_scales)

    wrong_group = _valid_tensors()
    wrong_group['.weight_scale'] = torch.zeros((8, 1), dtype=torch.float8_e4m3fn)
    assert not fmt.accepts(wrong_group)


def test_nvfp4_normalize_layout():
    fmt = NVFP4Format()

    packed = torch.tensor([[0x10, 0x32]], dtype=torch.uint8)
    weight = fmt.normalize(packed, 'weight')
    assert weight.dtype == torch.uint8
    assert weight.shape == (4, 1)
    assert weight.flatten().tolist() == [0, 1, 2, 3]

    scales = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float8_e4m3fn)
    assert fmt.normalize(scales, 'scales').shape == (2, 2)

    one_d = torch.tensor([1.0], dtype=torch.float32)
    assert fmt.normalize(one_d, 'global_scale').shape == (1, )


def test_nvfp4_finalize_folds_global_scale():
    fmt = NVFP4Format()
    tensors = {
        'weight': torch.zeros((4, 1), dtype=torch.uint8),
        'scales': torch.tensor([[2.0, 4.0]], dtype=torch.float32),
        'global_scale': torch.tensor([2.0], dtype=torch.float32),
    }

    folded = fmt.finalize(tensors)
    assert 'global_scale' not in folded
    assert folded['scales'].dtype == torch.float16
    assert torch.equal(folded['scales'], torch.tensor([[1.0, 2.0]], dtype=torch.float16))


def test_nvfp4_dequant_e2m1_table():
    fmt = NVFP4Format()
    codes = torch.arange(16, dtype=torch.uint8).reshape(16, 1)
    scales = torch.ones((1, 1), dtype=torch.float16)

    weight = fmt.dequant({'weight': codes, 'scales': scales}, torch.float32)['weight']
    expected = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
    assert torch.equal(weight.flatten(), expected)


def test_nvfp4_dequant_fractional_code_is_not_truncated():
    fmt = NVFP4Format()
    codes = torch.zeros((16, 1), dtype=torch.uint8)
    codes[0, 0] = 1
    codes[1, 0] = 7
    codes[2, 0] = 9
    codes[3, 0] = 15
    scales = torch.ones((1, 1), dtype=torch.float16)

    weight = fmt.dequant({'weight': codes, 'scales': scales}, torch.float32)['weight']
    expected = torch.tensor([0.5, 6.0, -0.5, -6.0] + [0.0] * 12)
    assert torch.equal(weight.flatten(), expected)


def test_nvfp4_dequant_preserves_bias():
    fmt = NVFP4Format()
    codes = torch.zeros((16, 1), dtype=torch.uint8)
    scales = torch.ones((1, 1), dtype=torch.float16)
    bias = torch.tensor([1.0, 2.0])

    out = fmt.dequant({'weight': codes, 'scales': scales, 'bias': bias}, torch.float32)
    assert torch.equal(out['bias'], bias)


def test_nvfp4_converter_mapping():
    from lmdeploy.turbomind.converter import (
        _DEFAULT_GROUP_SIZES,
        _SUPPORTED_GROUP_SIZES,
        _build_quantized_formats,
    )

    formats = _build_quantized_formats('nvfp4', 16)
    assert len(formats) == 1
    assert isinstance(formats[0], NVFP4Format)
    assert formats[0].block_in == 16

    default_formats = _build_quantized_formats('nvfp4', None)
    assert default_formats[0].block_in == 16

    assert _DEFAULT_GROUP_SIZES['nvfp4'] == 16
    assert _SUPPORTED_GROUP_SIZES['nvfp4'] == frozenset({16})

    with pytest.raises(ValueError, match='unknown model_format'):
        _build_quantized_formats('not-a-format', None)
