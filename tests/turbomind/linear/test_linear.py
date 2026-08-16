from __future__ import annotations

import pytest
import torch

from . import linear as linear_mod
from .cases import LinearCase, expand_suite
from .fixture import LinearFixture

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
tm_required = pytest.mark.skipif(not linear_mod.is_available(), reason='_turbomind required')


@cuda_required
@tm_required
@pytest.mark.parametrize('run', expand_suite('smoke'), ids=lambda r: f'{r.case.name}_m{r.batch_size}')
def test_smoke_linear_correctness(run):
    fx = LinearFixture(run.case)
    try:
        fx.prepare_batch(run.batch_size)
        fx.run_reference()
        fx.run_linear()
        fx.check_tolerances(fx.compare())
    finally:
        fx.close()


@cuda_required
@tm_required
def test_fp8_weight_only_fallback():
    major, _ = torch.cuda.get_device_capability()
    data_type = 'bf16' if major >= 8 else 'fp16'
    case = LinearCase(
        name='fp8_weight_only_fallback',
        input_dim=256,
        output_dim=256,
        data_type=data_type,
        weight_type='fp8_e4m3',
        input_type=data_type,
        group_size=128,
        expert_num=0,
        experts_per_token=0,
        combine_experts=False,
        moe_indexed=False,
        type_name=f'{data_type}_e4m3k128_{data_type}',
        shape_name='fp8_weight_only_fallback',
        tp_axis='output',
        max_tp=1,
        max_ep=1,
    )
    fx = LinearFixture(case, force_nonnative_fp8=True)
    try:
        assert fx.w_quant is not None
        assert fx.w_quant._impl.weight_format.block_sizes == [128, 1]
        assert fx.w_quant._impl.weight_format.scales.dtype == linear_mod.to_tm_dtype(data_type)
        fx.prepare_batch(3)
        fx.run_reference()
        fx.run_linear()
        fx.check_tolerances(fx.compare())
    finally:
        fx.close()
