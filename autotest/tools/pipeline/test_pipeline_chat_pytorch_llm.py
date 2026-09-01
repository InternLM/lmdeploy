import pytest
from tools.common_case_config import (
    MODELSCOPE_CONFIG,
    PYTORCH_LORA_TEST_LLM_GPU1,
    PYTORCH_PR_TEST_LLM_GPU1,
    PYTORCH_PR_TEST_LLM_GPU2,
)
from utils.config_utils import get_workerid
from utils.pipeline_chat import run_pipeline_llm_test
from utils.pytest_layout_utils import LOCAL_TP_LAYOUTS, build_layout_params, layout_mark

BACKEND = 'pytorch'
_PREFIX_CACHE_EXTRA = {'enable-prefix-caching': None}


def _chat_layout_marks(layout: dict[str, int]):
    marks = [pytest.mark.test_ascend]
    if layout == {'tp': 1}:
        marks.append(pytest.mark.test_3090)
    return marks


_CHAT_PARAMS = build_layout_params(
    BACKEND,
    LOCAL_TP_LAYOUTS,
    layout_extra_marks=_chat_layout_marks,
)
_PREFIX_CACHE_PARAMS = build_layout_params(
    BACKEND,
    LOCAL_TP_LAYOUTS[:2],
    extra=_PREFIX_CACHE_EXTRA,
    layout_extra_marks=lambda _layout: [pytest.mark.test_ascend],
)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.parametrize('run_config', _CHAT_PARAMS)
def test_pipeline_chat(config, run_config, common_case_config, worker_id):
    run_pipeline_llm_test(config, run_config, common_case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.parametrize('run_config', _PREFIX_CACHE_PARAMS)
def test_pipeline_chat_prefix_cache(config, run_config, common_case_config, worker_id):
    run_pipeline_llm_test(config, run_config, common_case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pr_test
@layout_mark({'tp': 2})
@pytest.mark.parametrize('run_config', PYTORCH_PR_TEST_LLM_GPU2)
def test_hf_pytorch_chat_pr_tp2(config, run_config, common_case_config, worker_id):
    worker_id = 'gw' + str(3 + get_workerid(worker_id))
    run_pipeline_llm_test(config, run_config, common_case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pr_test
@layout_mark({'tp': 1})
@pytest.mark.parametrize('run_config', PYTORCH_PR_TEST_LLM_GPU1)
def test_hf_pytorch_chat_pr_tp1(config, run_config, common_case_config, worker_id):
    worker_id = 'gw' + str(6 + get_workerid(worker_id))
    run_pipeline_llm_test(config, run_config, common_case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@layout_mark({'tp': 1})
@pytest.mark.parametrize('run_config', [item for item in MODELSCOPE_CONFIG if item['backend'] == BACKEND])
def test_modelscope_pipeline_chat_tp1(config, run_config, common_case_config, worker_id):
    case_config = {k: v for k, v in common_case_config.items() if k == 'memory_test'}
    run_pipeline_llm_test(config, run_config, case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@layout_mark({'tp': 1})
@pytest.mark.parametrize('run_config', PYTORCH_LORA_TEST_LLM_GPU1)
def test_pytorch_chat_with_lora_tp1(config, run_config, common_case_config, worker_id):
    run_pipeline_llm_test(config, run_config, common_case_config, worker_id)
