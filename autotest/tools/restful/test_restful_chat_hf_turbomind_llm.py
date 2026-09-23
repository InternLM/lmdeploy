import pytest
from tools.common_case_config import (
    MODELSCOPE_CONFIG,
    TURBOMIND_FALLBACK_TEST_LLM_GPU1,
    TURBOMIND_FALLBACK_TEST_LLM_GPU2,
    TURBOMIND_LOGPROBS_TEST_LLM_GPU2,
    TURBOMIND_PR_TEST_LLM_GPU1,
    TURBOMIND_PR_TEST_LLM_GPU2,
)
from utils.config_utils import get_workerid
from utils.pytest_layout_utils import LOCAL_TP_LAYOUTS, build_layout_params, layout_mark
from utils.run_restful_chat import run_llm_test, run_logprob_test

BACKEND = 'turbomind'
_PREFIX_CACHE_EXTRA = {'enable-prefix-caching': None}


def _chat_layout_marks(layout: dict[str, int]):
    if layout == {'tp': 1}:
        return [pytest.mark.test_3090]
    return []


_CHAT_PARAMS = build_layout_params(
    BACKEND,
    LOCAL_TP_LAYOUTS[:4],
    layout_extra_marks=_chat_layout_marks,
)
_PREFIX_CACHE_PARAMS = build_layout_params(
    BACKEND,
    ({'tp': 2},),
    extra=_PREFIX_CACHE_EXTRA,
)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.parametrize('run_config', _CHAT_PARAMS)
def test_restful_chat(config, run_config, common_case_config, worker_id):
    run_llm_test(config, run_config, common_case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.parametrize('run_config', _PREFIX_CACHE_PARAMS)
def test_restful_chat_prefix_cache(config, run_config, common_case_config, worker_id):
    run_llm_test(config, run_config, common_case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@layout_mark({'tp': 1})
@pytest.mark.parametrize('run_config', TURBOMIND_FALLBACK_TEST_LLM_GPU1)
def test_restful_chat_fallback_backend_tp1(config, run_config, common_case_config, worker_id):
    case_config = {k: v for k, v in common_case_config.items() if k == 'memory_test'}
    run_llm_test(config, run_config, case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@layout_mark({'tp': 2})
@pytest.mark.parametrize('run_config', TURBOMIND_FALLBACK_TEST_LLM_GPU2)
def test_restful_chat_fallback_backend_tp2(config, run_config, common_case_config, worker_id):
    case_config = {k: v for k, v in common_case_config.items() if k == 'memory_test'}
    run_llm_test(config, run_config, case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.flaky(reruns=0)
@pytest.mark.pr_test
@layout_mark({'tp': 2})
@pytest.mark.parametrize('run_config', TURBOMIND_PR_TEST_LLM_GPU2)
def test_restful_chat_pr_tp2(config, run_config, common_case_config, worker_id):
    worker_id = 'gw' + str(3 + get_workerid(worker_id))
    case_config = {k: v for k, v in common_case_config.items() if k == 'memory_test'}
    run_llm_test(config, run_config, case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.flaky(reruns=0)
@pytest.mark.pr_test
@layout_mark({'tp': 1})
@pytest.mark.parametrize('run_config', TURBOMIND_PR_TEST_LLM_GPU1)
def test_restful_chat_pr_tp1(config, run_config, common_case_config, worker_id):
    worker_id = 'gw' + str(6 + get_workerid(worker_id))
    case_config = {k: v for k, v in common_case_config.items() if k == 'memory_test'}
    run_llm_test(config, run_config, case_config, worker_id)


@pytest.mark.flaky(reruns=0)
@pytest.mark.pr_test
@layout_mark({'tp': 2})
@pytest.mark.parametrize('run_config', TURBOMIND_LOGPROBS_TEST_LLM_GPU2)
def test_restful_logprobs(config, run_config, worker_id):
    worker_id = 'gw' + str(3 + get_workerid(worker_id))
    run_logprob_test(config, run_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@layout_mark({'tp': 1})
@pytest.mark.parametrize('run_config', [item for item in MODELSCOPE_CONFIG if item['backend'] == BACKEND])
def test_modelscope_restful_chat_tp1(config, run_config, common_case_config, worker_id):
    case_config = {k: v for k, v in common_case_config.items() if k == 'memory_test'}
    run_llm_test(config, run_config, case_config, worker_id)
