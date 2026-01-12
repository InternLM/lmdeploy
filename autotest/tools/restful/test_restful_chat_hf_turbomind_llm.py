import pytest
from tools.common_case_config import (TURBOMIND_FALLBACK_TEST_LLM_GPU1, TURBOMIND_FALLBACK_TEST_LLM_GPU2,
                                      TURBOMIND_LOGPROBS_TEST_LLM_GPU2, TURBOMIND_MODELSCOPE_CONFIG,
                                      TURBOMIND_PR_TEST_LLM_GPU1, TURBOMIND_PR_TEST_LLM_GPU2,
                                      TURBOMIND_REASONING_TEST_LLM, TURBOMIND_TOOLCALL_TEST_LLM)
from utils.config_utils import get_func_config_list, get_workerid
from utils.constant import DEFAULT_PORT
from utils.run_restful_chat import (run_all_step, run_reasoning_case, run_tools_case, start_openai_service,
                                    terminate_restful_api, test_logprobs)


@pytest.fixture(scope='function', autouse=True)
def prepare_environment(request, config, worker_id):
    if hasattr(request, 'param'):
        run_config = request.param

        pid, startRes = start_openai_service(config, run_config, worker_id)
        try:
            yield run_config
        finally:
            if pid > 0:
                terminate_restful_api(worker_id, run_config)
    else:
        yield


BACKEND = 'turbomind'


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_1
@pytest.mark.test_3090
@pytest.mark.parametrize('prepare_environment', get_func_config_list(BACKEND, {'tp': 1}), indirect=True)
def test_restful_chat_tp1(config, common_case_config, worker_id):
    run_all_step(config, common_case_config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('prepare_environment', get_func_config_list(BACKEND, {'tp': 2}), indirect=True)
def test_restful_chat_tp2(config, common_case_config, worker_id):
    run_all_step(config, common_case_config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_4
@pytest.mark.parametrize('prepare_environment', get_func_config_list(BACKEND, {'tp': 4}), indirect=True)
def test_restful_chat_tp4(config, common_case_config, worker_id):
    run_all_step(config, common_case_config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_8
@pytest.mark.parametrize('prepare_environment', get_func_config_list(BACKEND, {'tp': 8}), indirect=True)
def test_restful_chat_tp8(config, common_case_config, worker_id):
    run_all_step(config, common_case_config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('prepare_environment',
                         get_func_config_list(BACKEND, {'tp': 2}, extra={'enable_prefix_caching': None}),
                         indirect=True)
def test_restful_chat_prefix_cache_tp2(config, common_case_config, worker_id):
    run_all_step(config, common_case_config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('prepare_environment', TURBOMIND_FALLBACK_TEST_LLM_GPU1, indirect=True)
def test_restful_chat_fallback_backend_tp1(config, common_case_config, worker_id):
    case_config = {k: v for k, v in common_case_config.items() if k == 'memory_test'}
    run_all_step(config, case_config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('prepare_environment', TURBOMIND_FALLBACK_TEST_LLM_GPU2, indirect=True)
def test_restful_chat_fallback_backend_tp2(config, common_case_config, worker_id):
    case_config = {k: v for k, v in common_case_config.items() if k == 'memory_test'}
    run_all_step(config, case_config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.pr_test
@pytest.mark.parametrize('prepare_environment', TURBOMIND_PR_TEST_LLM_GPU2, indirect=True)
def test_restful_chat_pr_tp2(config, common_case_config, worker_id):
    case_config = {k: v for k, v in common_case_config.items() if k == 'memory_test'}
    run_all_step(config, case_config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.pr_test
@pytest.mark.parametrize('prepare_environment', TURBOMIND_PR_TEST_LLM_GPU1, indirect=True)
def test_restful_chat_pr_tp1(config, common_case_config, worker_id):
    case_config = {k: v for k, v in common_case_config.items() if k == 'memory_test'}
    run_all_step(config, case_config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_2
@pytest.mark.pr_test
@pytest.mark.parametrize('prepare_environment', TURBOMIND_LOGPROBS_TEST_LLM_GPU2, indirect=True)
def test_restful_logprobs(worker_id):
    test_logprobs(worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('prepare_environment', TURBOMIND_MODELSCOPE_CONFIG, indirect=True)
def test_modelscope_restful_chat_tp1(config, common_case_config, worker_id):
    case_config = {k: v for k, v in common_case_config.items() if k == 'memory_test'}
    run_all_step(config, case_config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('prepare_environment',
                         [item for item in TURBOMIND_REASONING_TEST_LLM if item['parallel_config']['tp'] == 1],
                         indirect=True)
def test_restful_chat_reasoning_tp1(config, worker_id):
    run_reasoning_case(config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('prepare_environment',
                         [item for item in TURBOMIND_REASONING_TEST_LLM if item['parallel_config']['tp'] == 2],
                         indirect=True)
def test_restful_chat_reasoning_tp2(config, worker_id):
    run_reasoning_case(config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('prepare_environment',
                         [item for item in TURBOMIND_TOOLCALL_TEST_LLM if item['parallel_config']['tp'] == 1],
                         indirect=True)
def test_restful_chat_tools_tp1(config, worker_id):
    run_tools_case(config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('prepare_environment',
                         [item for item in TURBOMIND_TOOLCALL_TEST_LLM if item['parallel_config']['tp'] == 2],
                         indirect=True)
def test_restful_chat_tools_tp2(config, worker_id):
    run_tools_case(config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_4
@pytest.mark.parametrize('prepare_environment',
                         [item for item in TURBOMIND_TOOLCALL_TEST_LLM if item['parallel_config']['tp'] == 4],
                         indirect=True)
def test_restful_chat_tools_tp4(config, worker_id):
    run_tools_case(config, port=DEFAULT_PORT + get_workerid(worker_id))
