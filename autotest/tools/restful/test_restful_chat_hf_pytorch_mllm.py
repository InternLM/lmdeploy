import pytest
from utils.config_utils import get_func_config_list, get_workerid
from utils.constant import DEFAULT_PORT
from utils.run_restful_chat import run_mllm_case

BACKEND = 'pytorch'


@pytest.mark.gpu_num_1
@pytest.mark.test_3090
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 1}, model_type='vl_model'))
def test_restful_chat_tp1(config, run_config, worker_id):
    run_mllm_case(config, run_config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.gpu_num_2
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 2}, model_type='vl_model'))
def test_restful_chat_tp2(config, run_config, worker_id):
    run_mllm_case(config, run_config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.gpu_num_4
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 4}, model_type='vl_model'))
def test_restful_chat_tp4(config, run_config, worker_id):
    run_mllm_case(config, run_config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.gpu_num_8
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 8}, model_type='vl_model'))
def test_restful_chat_tp8(config, run_config, worker_id):
    run_mllm_case(config, run_config, port=DEFAULT_PORT + get_workerid(worker_id))


@pytest.mark.gpu_num_16
@pytest.mark.parametrize('run_config', get_func_config_list(BACKEND, {'tp': 16}, model_type='vl_model'))
def test_restful_chat_tp16(config, run_config, worker_id):
    run_mllm_case(config, run_config, port=DEFAULT_PORT + get_workerid(worker_id))
