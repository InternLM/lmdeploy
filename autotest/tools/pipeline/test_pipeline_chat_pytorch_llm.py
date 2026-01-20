import pytest
from tools.common_case_config import (MODELSCOPE_CONFIG, PYTORCH_LORA_TEST_LLM_GPU1, PYTORCH_LORA_TEST_LLM_GPU2,
                                      PYTORCH_PR_TEST_LLM_GPU1, PYTORCH_PR_TEST_LLM_GPU2)
from utils.config_utils import get_func_config_list, get_workerid
from utils.pipeline_chat import run_pipeline_llm_test

BACKEND = 'pytorch'


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_1
@pytest.mark.test_3090
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 1}))
def test_pipeline_chat_tp1(config, run_config, common_case_config, worker_id):
    run_pipeline_llm_test(config, run_config, common_case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 2}))
def test_pipeline_chat_tp2(config, run_config, common_case_config, worker_id):
    run_pipeline_llm_test(config, run_config, common_case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_4
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 4}))
def test_pipeline_chat_tp4(config, run_config, common_case_config, worker_id):
    run_pipeline_llm_test(config, run_config, common_case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_8
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 8}))
def test_pipeline_chat_tp8(config, run_config, common_case_config, worker_id):
    run_pipeline_llm_test(config, run_config, common_case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_16
@pytest.mark.test_ascend
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 16}))
def test_pipeline_chat_tp16(config, run_config, common_case_config, worker_id):
    run_pipeline_llm_test(config, run_config, common_case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 2}, extra={'enable_prefix_caching': True}))
def test_pipeline_chat_turbomind_prefix_cache_tp2(config, run_config, common_case_config, worker_id):
    run_pipeline_llm_test(config, run_config, common_case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_2
@pytest.mark.pr_test
@pytest.mark.parametrize('run_config', PYTORCH_PR_TEST_LLM_GPU2)
def test_hf_turbomind_chat_pr_tp2(config, run_config, common_case_config, worker_id):
    worker_id = 'gw' + str(3 + get_workerid(worker_id))
    run_pipeline_llm_test(config, run_config, common_case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_1
@pytest.mark.pr_test
@pytest.mark.parametrize('run_config', PYTORCH_PR_TEST_LLM_GPU1)
def test_hf_turbomind_chat_pr_tp1(config, run_config, common_case_config, worker_id):
    worker_id = 'gw' + str(6 + get_workerid(worker_id))
    run_pipeline_llm_test(config, run_config, common_case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('run_config', [item for item in MODELSCOPE_CONFIG if item['backend'] == BACKEND])
def test_modelscope_pipeline_chat_tp1(config, run_config, common_case_config, worker_id):
    case_config = {k: v for k, v in common_case_config.items() if k == 'memory_test'}
    run_pipeline_llm_test(config, run_config, case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('run_config', PYTORCH_LORA_TEST_LLM_GPU1)
def test_pytorch_chat_with_lora_tp1(config, run_config, common_case_config, worker_id):
    run_pipeline_llm_test(config, run_config, common_case_config, worker_id)


@pytest.mark.usefixtures('common_case_config')
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('run_config', PYTORCH_LORA_TEST_LLM_GPU2)
def test_pytorch_chat_with_lora_tp2(config, run_config, common_case_config, worker_id):
    run_pipeline_llm_test(config, run_config, common_case_config, worker_id)
