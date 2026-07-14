import pytest
from utils.config_utils import get_func_config_list
from utils.pipeline_chat import run_pipeline_mllm_test

BACKEND = 'pytorch'


def get_models(parallel_config):
    return get_func_config_list(BACKEND, parallel_config, model_type='vl_model', extra={'session_len': 8192})


@pytest.mark.gpu_num_1
@pytest.mark.test_3090
@pytest.mark.parametrize('run_config', get_models({'tp': 1}))
def test_restful_chat_tp1(config, run_config, worker_id):
    run_pipeline_mllm_test(config, run_config, worker_id)


@pytest.mark.gpu_num_2
@pytest.mark.parametrize('run_config', get_models({'tp': 2}))
def test_restful_chat_tp2(config, run_config, worker_id):
    run_pipeline_mllm_test(config, run_config, worker_id)


@pytest.mark.gpu_num_4
@pytest.mark.parametrize('run_config', get_models({'tp': 4}))
def test_restful_chat_tp4(config, run_config, worker_id):
    run_pipeline_mllm_test(config, run_config, worker_id)


@pytest.mark.gpu_num_8
@pytest.mark.parametrize('run_config', get_models({'tp': 8}))
def test_restful_chat_tp8(config, run_config, worker_id):
    run_pipeline_mllm_test(config, run_config, worker_id)


@pytest.mark.gpu_num_16
@pytest.mark.parametrize('run_config', get_models({'tp': 16}))
def test_restful_chat_tp16(config, run_config, worker_id):
    run_pipeline_mllm_test(config, run_config, worker_id)
