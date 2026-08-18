import pytest
from utils.config_utils import get_func_config_list
from utils.pipeline_chat import run_pipeline_mllm_test
from utils.pytest_layout_utils import LOCAL_TP_LAYOUTS, build_layout_params

BACKEND = 'pytorch'
_MLLM_EXTRA = {'session_len': 8192}


def _mllm_layout_marks(layout: dict[str, int]):
    if layout == {'tp': 1}:
        return [pytest.mark.test_3090]
    return []


_MLLM_PARAMS = build_layout_params(
    BACKEND,
    LOCAL_TP_LAYOUTS,
    model_type='vl_model',
    extra=_MLLM_EXTRA,
    layout_extra_marks=_mllm_layout_marks,
)


@pytest.mark.parametrize('run_config', _MLLM_PARAMS)
def test_pipeline_mllm_chat(config, run_config, worker_id):
    run_pipeline_mllm_test(config, run_config, worker_id)
