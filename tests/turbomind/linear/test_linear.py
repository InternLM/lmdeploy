from __future__ import annotations

import pytest
import torch

from lmdeploy import turbomind

if not turbomind.is_available():
    pytest.skip('TurboMind is not built', allow_module_level=True)

from lmdeploy.turbomind import _tm

from .cases import case_by_name, expand_suite
from .fixture import LinearFixture

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_weight_format_dtype_contract():
    from lmdeploy.turbomind.builders.linear import _build_linear
    from lmdeploy.turbomind.weight_format import (
        _GENERIC_FLOAT_DTYPES,
        AWQFormat,
        CompressedTensorFormat,
        FP8Format,
        GPTQFormat,
        TrivialFormat,
    )

    expected = frozenset({
        torch.float16,
        torch.bfloat16,
        torch.float32,
    })
    assert _GENERIC_FLOAT_DTYPES == expected

    for dtype in (*expected, torch.float64, torch.int32):
        awq_scales = torch.ones((1, 8), dtype=dtype)
        gptq_scales = torch.ones((1, 8), dtype=dtype)
        compressed_scales = torch.ones((1, 8), dtype=dtype)
        fp8_scales = torch.ones((1, 1), dtype=dtype)

        cases = (
            (
                AWQFormat(block_in=128),
                {
                    '.qweight': torch.zeros(
                        (128, 1),
                        dtype=torch.int32,
                    ),
                    '.scales': awq_scales,
                    '.qzeros': torch.zeros(
                        (1, 1),
                        dtype=torch.int32,
                    ),
                },
            ),
            (
                GPTQFormat(block_in=128),
                {
                    '.qweight': torch.zeros(
                        (16, 8),
                        dtype=torch.int32,
                    ),
                    '.scales': gptq_scales,
                },
            ),
            (
                CompressedTensorFormat(block_in=128),
                {
                    '.weight_packed': torch.zeros(
                        (16, 8),
                        dtype=torch.int32,
                    ),
                    '.weight_scale': compressed_scales,
                },
            ),
            (
                FP8Format(block_out=128),
                {
                    '.weight': torch.zeros(
                        (128, 128),
                        dtype=torch.uint8,
                    ),
                    '.weight_scale_inv': fp8_scales,
                },
            ),
        )

        accepted = dtype in expected
        for weight_format, available in cases:
            assert weight_format.accepts(available) is accepted
            if accepted:
                linear = _build_linear(weight_format, available)
                assert linear.tensors['scales'].dtype == dtype
                if (
                    weight_format.zeros_dtype
                    != _tm.DataType.TYPE_INVALID
                ):
                    assert linear.tensors['zeros'].dtype == dtype
                data_format = weight_format.make_data_format()
                assert (
                    data_format.scales.dtype
                    == _tm.DataType.TYPE_GENERIC_FLOAT
                )

    assert not AWQFormat(block_in=128).accepts({
        '.qweight': torch.zeros((128, 1), dtype=torch.int32),
        '.scales': torch.ones((1, 8), dtype=torch.float16),
    })

    fp16 = TrivialFormat(weight_dtype=_tm.DataType.TYPE_FP16)
    bf16 = TrivialFormat(weight_dtype=_tm.DataType.TYPE_BF16)
    assert fp16 != bf16
    assert len({fp16, bf16}) == 2


@cuda_required
@pytest.mark.parametrize('run', expand_suite('smoke'), ids=lambda r: f'{r.case.name}_m{r.batch_size}')
def test_smoke_linear_correctness(run):
    try:
        fx = LinearFixture(run.case)
    except NotImplementedError as e:
        pytest.skip(str(e))
    try:
        fx.prepare_batch(run.batch_size)
        fx.run_reference()
        fx.run_linear()
        assert fx.returned_output is fx.output
        assert fx.returned_scales is fx.output_scales
        minimum, alignment = fx.weight_plan.shape_constraints
        assert all(value > 0 for value in (*minimum, *alignment))
        if fx.output_scales is not None:
            rows = fx.output.shape[0]
            groups = (fx.output.shape[-1] + 127) // 128
            aligned_rows = (rows + 3) // 4 * 4
            assert fx.output_scales.shape == (groups, rows)
            assert fx.output_scales.stride() == (aligned_rows, 1)
        fx.check_tolerances(fx.compare())
    finally:
        fx.close()


@cuda_required
def test_rank_three_dense_input():
    case = case_by_name()['llama2_7b_o__bf16_bf16_bf16']
    fx = LinearFixture(case)
    try:
        fx.prepare_batch(6)
        x = fx.x_original.view(2, 3, case.input_dim)
        fx.x_original = x
        fx.x_source = x
        fx.exec_plan = fx.linear.get_exec_plan(x, fx.w_quant)
        fx.output = None
        fx.output_scales = None
        fx.run_reference()
        fx.run_linear()
        assert fx.output.shape == (2, 3, case.output_dim)
        fx.check_tolerances(fx.compare())
    finally:
        fx.close()
