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
    # Declared qparam format: the checkpoint's raw e4m3 scales.
    assert fmt.scales_dtype == _tm.DataType.TYPE_FP8_E4M3
    assert fmt.zeros_dtype == _tm.DataType.TYPE_INVALID
    data_format = fmt.make_data_format()
    assert data_format.dtype == _tm.DataType.TYPE_FP4_E2M1
    assert data_format.scales.dtype == _tm.DataType.TYPE_FP8_E4M3

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


def test_nvfp4_finalize_keeps_raw_scales_and_stores_reciprocal():
    fmt = NVFP4Format()
    scales = torch.tensor([[2.0, 4.0]], dtype=torch.float8_e4m3fn)
    tensors = {
        'weight': torch.zeros((4, 1), dtype=torch.uint8),
        'scales': scales,
        'global_scale': torch.tensor([2.0], dtype=torch.float32),
    }

    finalized = fmt.finalize(tensors)
    # Raw e4m3 scales pass through untouched; the bundle keeps global_scale
    # holding the epilogue multiplier 1/gs.
    assert 'global_scale' in finalized
    assert finalized['scales'].dtype == torch.float8_e4m3fn
    assert torch.equal(finalized['scales'].view(torch.uint8), scales.view(torch.uint8))
    assert finalized['global_scale'].dtype == torch.float32
    assert torch.allclose(finalized['global_scale'], torch.tensor([0.5], dtype=torch.float32))
    # The caller's dict is not mutated.
    assert torch.equal(tensors['global_scale'], torch.tensor([2.0], dtype=torch.float32))


def test_nvfp4_finalize_requires_per_tensor_global_scale():
    fmt = NVFP4Format()
    tensors = {
        'weight': torch.zeros((4, 1), dtype=torch.uint8),
        'scales': torch.zeros((1, 2), dtype=torch.float8_e4m3fn),
        'global_scale': torch.ones((2, ), dtype=torch.float32),
    }
    with pytest.raises(ValueError, match='per-tensor'):
        fmt.finalize(tensors)


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


def test_nvfp4_dequant_raw_representation_divides_by_global_scale():
    """Raw bundle dequant: codes * scales * (1/gs) == codes * scales / gs."""
    fmt = NVFP4Format()
    codes = torch.zeros((16, 1), dtype=torch.uint8)
    codes[0, 0] = 6  # 4.0
    codes[1, 0] = 7  # 6.0
    scales = torch.tensor([[2.0]], dtype=torch.float8_e4m3fn)
    global_scale = torch.tensor([0.25], dtype=torch.float32)  # 1/gs with gs=4

    weight = fmt.dequant(
        {
            'weight': codes,
            'scales': scales,
            'global_scale': global_scale,
        }, torch.float32)['weight']
    # codes 4.0 * scale 2.0 * 0.25 = 2.0; 6.0 * 2.0 * 0.25 = 3.0
    assert weight.flatten().tolist() == [2.0, 3.0] + [0.0] * 14


class _FoldedNvFp4Shim(NVFP4Format):
    """The retired SM75 folded representation: fp16 scales, no global."""

    scales_dtype = _tm.DataType.TYPE_FP16


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_nvfp4_plan_selects_native_global_scale_lane():
    """On SM80+ devices the raw-format plan query hits the native lane and the
    bridge widens scales to fp16; the legacy folded fp16 format has no family
    anymore."""
    import lmdeploy.turbomind as tm

    if not tm.is_available():
        pytest.skip('TurboMind is not built')

    from lmdeploy.turbomind.linear import get_linear

    device = torch.device('cuda')
    props = torch.cuda.get_device_properties(device)
    if props.major < 8:
        pytest.skip(f'NVFP4 native lane requires SM80+, got SM{props.major}{props.minor}')

    linear = get_linear()
    with torch.cuda.device(device):
        plan = linear.get_weight_plan(weight_format=NVFP4Format(), dtype=torch.float16)
        # Only the native bridge matches the raw e4m3 layout: no replication,
        # scales widened to fp16, no zeros.
        assert plan._impl.bridge == (1, 1, _tm.DataType.TYPE_FP16, _tm.DataType.TYPE_INVALID)

        # The retired folded representation no longer matches any family.
        with pytest.raises(NotImplementedError, match='no GEMM family'):
            linear.get_weight_plan(weight_format=_FoldedNvFp4Shim(), dtype=torch.float16)
