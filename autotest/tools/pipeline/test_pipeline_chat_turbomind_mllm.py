import pytest
from tools.common_case_config import (
    TURBOMIND_FALLBACK_TEST_MLLM_GPU1,
    TURBOMIND_PR_TEST_MLLM_GPU1,
    TURBOMIND_PR_TEST_MLLM_GPU2,
)
from utils.config_utils import get_workerid
from utils.pipeline_chat import run_pipeline_mllm_test
from utils.pytest_layout_utils import LOCAL_TP_LAYOUTS, build_layout_params, layout_mark

BACKEND = 'turbomind'
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


@pytest.mark.other
@layout_mark({'tp': 1})
@pytest.mark.parametrize('run_config', TURBOMIND_FALLBACK_TEST_MLLM_GPU1)
def test_pipeline_mllm_fallback_backend_tp1(config, run_config, worker_id):
    run_pipeline_mllm_test(config, run_config, worker_id)


@pytest.mark.other
@pytest.mark.pr_test
@layout_mark({'tp': 1})
@pytest.mark.parametrize('run_config', TURBOMIND_PR_TEST_MLLM_GPU1)
def test_pipeline_mllm_pr_tp1(config, run_config, worker_id):
    worker_id = 'gw' + str(6 + get_workerid(worker_id))
    run_pipeline_mllm_test(config, run_config, worker_id, is_smoke=True)


@pytest.mark.other
@pytest.mark.pr_test
@layout_mark({'tp': 2})
@pytest.mark.parametrize('run_config', TURBOMIND_PR_TEST_MLLM_GPU2)
def test_pipeline_mllm_pr_tp2(config, run_config, worker_id):
    worker_id = 'gw' + str(3 + get_workerid(worker_id))
    run_pipeline_mllm_test(config, run_config, worker_id, is_smoke=True)
