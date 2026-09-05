import pytest
from tools.common_case_config import TURBOMIND_FALLBACK_TEST_MLLM_GPU1
from utils.pytest_layout_utils import LOCAL_TP_LAYOUTS, build_layout_params, layout_mark
from utils.run_restful_chat import run_mllm_test

BACKEND = 'turbomind'


def _mllm_layout_marks(layout: dict[str, int]):
    if layout == {'tp': 1}:
        return [pytest.mark.test_3090]
    return []


_MLLM_PARAMS = build_layout_params(
    BACKEND,
    LOCAL_TP_LAYOUTS,
    model_type='vl_model',
    layout_extra_marks=_mllm_layout_marks,
)


@pytest.mark.parametrize('run_config', _MLLM_PARAMS)
def test_restful_mllm_chat(config, run_config, worker_id):
    run_mllm_test(config, run_config, worker_id)


@pytest.mark.other
@layout_mark({'tp': 1})
@pytest.mark.parametrize('run_config', TURBOMIND_FALLBACK_TEST_MLLM_GPU1)
def test_restful_mllm_fallback_backend_tp1(config, run_config, worker_id):
    run_mllm_test(config, run_config, worker_id)
