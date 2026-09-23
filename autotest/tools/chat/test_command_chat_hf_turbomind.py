import pytest
from tools.common_case_config import (
    MODELSCOPE_CONFIG,
    TURBOMIND_FALLBACK_TEST_LLM_GPU1,
    TURBOMIND_FALLBACK_TEST_LLM_GPU2,
    TURBOMIND_PR_TEST_LLM_GPU1,
    TURBOMIND_PR_TEST_LLM_GPU2,
)
from utils.config_utils import get_workerid
from utils.pytest_layout_utils import (
    BASE_TP_LAYOUTS,
    LOCAL_TP_LAYOUTS,
    build_layout_params,
    layout_mark,
)
from utils.run_client_chat import run_tests

BACKEND = 'turbomind'


def _chat_layout_marks(layout: dict[str, int]):
    if layout == {'tp': 1}:
        return [pytest.mark.test_3090]
    return []


_CHAT_PARAMS = build_layout_params(
    BACKEND,
    LOCAL_TP_LAYOUTS,
    layout_extra_marks=_chat_layout_marks,
)
_BASE_PARAMS = build_layout_params(
    BACKEND,
    BASE_TP_LAYOUTS,
    model_type='base_model',
)


@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.parametrize('run_config', _CHAT_PARAMS)
def test_hf_turbomind_chat(config, run_config, cli_case_config, worker_id):
    run_tests(config, 'chat_testcase', cli_case_config, run_config, worker_id)


@pytest.mark.usefixtures('cli_case_config')
@layout_mark({'tp': 1})
@pytest.mark.parametrize('run_config', TURBOMIND_FALLBACK_TEST_LLM_GPU1)
def test_hf_turbomind_chat_fallback_backend_tp1(config, run_config, cli_case_config, worker_id):
    run_tests(config, 'chat_testcase', cli_case_config, run_config, worker_id)


@pytest.mark.usefixtures('cli_case_config')
@layout_mark({'tp': 2})
@pytest.mark.parametrize('run_config', TURBOMIND_FALLBACK_TEST_LLM_GPU2)
def test_hf_turbomind_chat_fallback_backend_tp2(config, run_config, cli_case_config, worker_id):
    run_tests(config, 'chat_testcase', cli_case_config, run_config, worker_id)


@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.parametrize('run_config', _BASE_PARAMS)
def test_hf_turbomind_base(config, run_config, cli_case_config, worker_id):
    run_tests(config, 'base_testcase', cli_case_config, run_config, worker_id)


@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.pr_test
@layout_mark({'tp': 2})
@pytest.mark.parametrize('run_config', TURBOMIND_PR_TEST_LLM_GPU2)
def test_hf_turbomind_chat_pr_tp2(config, run_config, cli_case_config, worker_id):
    worker_id = 'gw' + str(3 + get_workerid(worker_id))
    run_tests(config, 'chat_testcase', cli_case_config, run_config, worker_id)


@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.pr_test
@layout_mark({'tp': 1})
@pytest.mark.parametrize('run_config', TURBOMIND_PR_TEST_LLM_GPU1)
def test_hf_turbomind_chat_pr_tp1(config, run_config, cli_case_config, worker_id):
    worker_id = 'gw' + str(6 + get_workerid(worker_id))
    run_tests(config, 'chat_testcase', cli_case_config, run_config, worker_id)


@pytest.mark.usefixtures('cli_case_config')
@layout_mark({'tp': 1})
@pytest.mark.parametrize('run_config', [item for item in MODELSCOPE_CONFIG if item['backend'] == BACKEND])
def test_modelscope_turbomind_chat_tp1(config, run_config, cli_case_config, worker_id):
    run_tests(config, 'chat_testcase', cli_case_config, run_config, worker_id)
