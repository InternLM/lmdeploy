import pytest
from utils.pytest_layout_utils import LOCAL_TP_LAYOUTS, build_layout_params
from utils.run_restful_chat import run_mllm_test

BACKEND = 'pytorch'


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
