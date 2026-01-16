import pytest
from tools.common_case_config import (MODELSCOPE_CONFIG, REASONING_TEST_LLM, TOOLCALL_TEST_LLM,
                                      TURBOMIND_FALLBACK_TEST_LLM_GPU1, TURBOMIND_FALLBACK_TEST_LLM_GPU2,
                                      TURBOMIND_LOGPROBS_TEST_LLM_GPU2, TURBOMIND_PR_TEST_LLM_GPU1,
                                      TURBOMIND_PR_TEST_LLM_GPU2)
from utils.config_utils import get_func_config_list, get_workerid
from utils.run_restful_chat import run_llm_test, run_logprob_test, run_reasoning_case, run_tools_case

BACKEND = 'turbomind'


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_1
@pytest.mark.test_3090
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 1}))
def test_restful_chat_tp1(config, run_config, common_case_config, worker_id):
    run_llm_test(config, run_config, common_case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 2}))
def test_restful_chat_tp2(config, run_config, common_case_config, worker_id):
    run_llm_test(config, run_config, common_case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_4
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 4}))
def test_restful_chat_tp4(config, run_config, common_case_config, worker_id):
    run_llm_test(config, run_config, common_case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_8
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 8}))
def test_restful_chat_tp8(config, run_config, common_case_config, worker_id):
    run_llm_test(config, run_config, common_case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 2}, extra={'enable_prefix_caching': True}))
def test_restful_chat_prefix_cache_tp2(config, run_config, common_case_config, worker_id):
    run_llm_test(config, run_config, common_case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('run_config', TURBOMIND_FALLBACK_TEST_LLM_GPU1)
def test_restful_chat_fallback_backend_tp1(config, run_config, common_case_config, worker_id):
    case_config = {k: v for k, v in common_case_config.items() if k == 'memory_test'}
    run_llm_test(config, run_config, run_config, case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('run_config', TURBOMIND_FALLBACK_TEST_LLM_GPU2)
def test_restful_chat_fallback_backend_tp2(config, run_config, common_case_config, worker_id):
    case_config = {k: v for k, v in common_case_config.items() if k == 'memory_test'}
    run_llm_test(config, run_config, run_config, case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.pr_test
@pytest.mark.parametrize('run_config', TURBOMIND_PR_TEST_LLM_GPU2)
def test_restful_chat_pr_tp2(config, run_config, common_case_config, worker_id):
    worker_id = 'gw' + str(5 + get_workerid(worker_id))
    case_config = {k: v for k, v in common_case_config.items() if k == 'memory_test'}
    run_llm_test(config, run_config, case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.pr_test
@pytest.mark.parametrize('run_config', TURBOMIND_PR_TEST_LLM_GPU1)
def test_restful_chat_pr_tp1(config, run_config, common_case_config, worker_id):
    worker_id = 'gw' + str(5 + get_workerid(worker_id))
    case_config = {k: v for k, v in common_case_config.items() if k == 'memory_test'}
    run_llm_test(config, run_config, case_config, worker_id)


@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_2
@pytest.mark.pr_test
@pytest.mark.parametrize('run_config', TURBOMIND_LOGPROBS_TEST_LLM_GPU2)
def test_restful_logprobs(config, run_config, worker_id):
    worker_id = 'gw' + str(5 + get_workerid(worker_id))
    run_logprob_test(config, run_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('run_config', [item for item in MODELSCOPE_CONFIG if item['backend'] == BACKEND])
def test_modelscope_restful_chat_tp1(config, run_config, common_case_config, worker_id):
    case_config = {k: v for k, v in common_case_config.items() if k == 'memory_test'}
    run_llm_test(config, run_config, case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_1
@pytest.mark.parametrize(
    'run_config',
    [item for item in REASONING_TEST_LLM if item['backend'] == BACKEND and item['parallel_config']['tp'] == 1])
def test_restful_chat_reasoning_tp1(config, run_config, worker_id):
    run_reasoning_case(config, run_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_2
@pytest.mark.parametrize(
    'run_config',
    [item for item in REASONING_TEST_LLM if item['backend'] == BACKEND and item['parallel_config']['tp'] == 2])
def test_restful_chat_reasoning_tp2(config, run_config, worker_id):
    run_reasoning_case(config, run_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_1
@pytest.mark.parametrize(
    'run_config',
    [item for item in TOOLCALL_TEST_LLM if item['backend'] == BACKEND and item['parallel_config']['tp'] == 1])
def test_restful_chat_tools_tp1(config, run_config, worker_id):
    run_tools_case(config, run_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_2
@pytest.mark.parametrize(
    'run_config',
    [item for item in TOOLCALL_TEST_LLM if item['backend'] == BACKEND and item['parallel_config']['tp'] == 2])
def test_restful_chat_tools_tp2(config, run_config, worker_id):
    run_tools_case(config, run_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_4
@pytest.mark.parametrize(
    'run_config',
    [item for item in TOOLCALL_TEST_LLM if item['backend'] == BACKEND and item['parallel_config']['tp'] == 4])
def test_restful_chat_tools_tp4(config, run_config, worker_id):
    run_tools_case(config, run_config, worker_id)
