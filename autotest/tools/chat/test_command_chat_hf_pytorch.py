import pytest
from tools.common_case_config import (MODELSCOPE_CONFIG, PYTORCH_LORA_TEST_LLM_GPU1, PYTORCH_LORA_TEST_LLM_GPU2,
                                      PYTORCH_PR_TEST_LLM_GPU1, PYTORCH_PR_TEST_LLM_GPU2)
from utils.config_utils import get_func_config_list, get_workerid
from utils.run_client_chat import run_tests

BACKEND = 'pytorch'


@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.gpu_num_1
@pytest.mark.test_3090
@pytest.mark.test_ascend
@pytest.mark.parametrize('run_config', get_func_config_list('pytorch', {'tp': 1}))
def test_hf_pytorch_chat_tp1(config, run_config, cli_case_config, worker_id):
    run_tests(config, 'chat_testcase', cli_case_config, run_config, worker_id)


@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.gpu_num_2
@pytest.mark.test_ascend
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 2}))
def test_hf_pytorch_chat_tp2(config, run_config, cli_case_config, worker_id):
    run_tests(config, 'chat_testcase', cli_case_config, run_config, worker_id)


@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.gpu_num_4
@pytest.mark.test_ascend
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 4}))
def test_hf_pytorch_chat_tp4(config, run_config, cli_case_config, worker_id):
    run_tests(config, 'chat_testcase', cli_case_config, run_config, worker_id)


@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.gpu_num_8
@pytest.mark.test_ascend
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 8}))
def test_hf_pytorch_chat_tp8(config, run_config, cli_case_config, worker_id):
    run_tests(config, 'chat_testcase', cli_case_config, run_config, worker_id)


@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.gpu_num_16
@pytest.mark.test_ascend
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 16}))
def test_hf_pytorch_chat_tp16(config, run_config, cli_case_config, worker_id):
    run_tests(config, 'chat_testcase', cli_case_config, run_config, worker_id)


@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.gpu_num_1
@pytest.mark.test_ascend
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 1}, 'base_model'))
def test_hf_pytorch_base_tp1(config, run_config, cli_case_config, worker_id):
    run_tests(config, 'base_testcase', cli_case_config, run_config, worker_id)


@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.gpu_num_2
@pytest.mark.test_ascend
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 2}, 'base_model'))
def test_hf_pytorch_base_tp2(config, run_config, cli_case_config, worker_id):
    run_tests(config, 'base_testcase', cli_case_config, run_config, worker_id)


@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.gpu_num_2
@pytest.mark.pr_test
@pytest.mark.parametrize('run_config', PYTORCH_PR_TEST_LLM_GPU2)
def test_hf_turbomind_chat_pr_tp2(config, run_config, cli_case_config, worker_id):
    worker_id = 'gw' + str(3 + get_workerid(worker_id))
    run_tests(config, 'chat_testcase', cli_case_config, run_config, worker_id)


@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.gpu_num_1
@pytest.mark.pr_test
@pytest.mark.parametrize('run_config', PYTORCH_PR_TEST_LLM_GPU1)
def test_hf_turbomind_chat_pr_tp1(config, run_config, cli_case_config, worker_id):
    worker_id = 'gw' + str(6 + get_workerid(worker_id))
    run_tests(config, 'chat_testcase', cli_case_config, run_config, worker_id)


@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('run_config', [item for item in MODELSCOPE_CONFIG if item['backend'] == BACKEND])
def test_modelscope_turbomind_chat_tp1(config, run_config, cli_case_config, worker_id):
    run_config['env'] = {'LMDEPLOY_USE_MODELSCOPE': 'True'}
    run_tests(config, 'chat_testcase', cli_case_config, run_config, worker_id)


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_pytorch_chat
@pytest.mark.gpu_num_1
@pytest.mark.other
@pytest.mark.parametrize('run_config', PYTORCH_LORA_TEST_LLM_GPU1)
def test_pytorch_chat_with_lora_tp1(config, run_config, cli_case_config, worker_id):
    run_tests(config, 'chat_testcase', cli_case_config, run_config, worker_id)


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_pytorch_chat
@pytest.mark.gpu_num_1
@pytest.mark.other
@pytest.mark.parametrize('run_config', PYTORCH_LORA_TEST_LLM_GPU2)
def test_pytorch_chat_with_lora_tp2(config, run_config, cli_case_config, worker_id):
    run_tests(config, 'chat_testcase', cli_case_config, run_config, worker_id)
