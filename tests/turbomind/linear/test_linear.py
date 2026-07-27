from __future__ import annotations

import pytest
import torch

from tests.turbomind.linear import linear as linear_mod
from tests.turbomind.linear.cases import expand_suite
from tests.turbomind.linear.fixture import LinearFixture

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
