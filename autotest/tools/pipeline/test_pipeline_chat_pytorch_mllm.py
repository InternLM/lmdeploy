import os

import pytest
from utils.config_utils import get_torch_model_list, set_device_env_variable
from utils.pipeline_chat import run_pipeline_vl_chat_test

BACKEND = 'pytorch'
BACKEND_KVINT = 'pytorch-kvint'


@pytest.mark.order(6)
@pytest.mark.pipeline_chat
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_1
@pytest.mark.test_3090
@pytest.mark.test_ascend
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=1, model_type='vl_model'))
def test_pipeline_chat_tp1(config, model, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id)
    run_pipeline_vl_chat_test(config, model, BACKEND, worker_id)


@pytest.mark.order(6)
@pytest.mark.pipeline_chat
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_2
@pytest.mark.test_ascend
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=2, model_type='vl_model'))
def test_pipeline_chat_tp2(config, model, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, parallel_config=2)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    run_pipeline_vl_chat_test(config, model, BACKEND, worker_id)


@pytest.mark.order(6)
@pytest.mark.pipeline_chat
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_4
@pytest.mark.test_ascend
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=4, model_type='vl_model'))
def test_pipeline_chat_tp4(config, model, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, parallel_config=4)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    run_pipeline_vl_chat_test(config, model, BACKEND, worker_id)


@pytest.mark.order(6)
@pytest.mark.pipeline_chat
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_8
@pytest.mark.test_ascend
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=8, model_type='vl_model'))
def test_pipeline_chat_tp8(config, model, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, parallel_config=8)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    run_pipeline_vl_chat_test(config, model, BACKEND, worker_id)


@pytest.mark.order(6)
@pytest.mark.pipeline_chat
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_16
@pytest.mark.test_ascend
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=16, model_type='vl_model'))
def test_pipeline_chat_tp16(config, model, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, parallel_config=16)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    run_pipeline_vl_chat_test(config, model, BACKEND, worker_id)


@pytest.mark.order(6)
@pytest.mark.pipeline_chat
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_1
@pytest.mark.test_3090
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=1, quant_policy=4, model_type='vl_model'))
def test_pipeline_chat_kvint4_tp1(config, model, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id)
    run_pipeline_vl_chat_test(config, model, BACKEND_KVINT, worker_id, {'quant_policy': 4})


@pytest.mark.order(6)
@pytest.mark.pipeline_chat
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=2, quant_policy=4, model_type='vl_model'))
def test_pipeline_chat_kvint4_tp2(config, model, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, parallel_config=2)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    run_pipeline_vl_chat_test(config, model, BACKEND_KVINT, worker_id, {'quant_policy': 4})


@pytest.mark.order(6)
@pytest.mark.pipeline_chat
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_4
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=4, quant_policy=4, model_type='vl_model'))
def test_pipeline_chat_kvint4_tp4(config, model, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, parallel_config=4)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    run_pipeline_vl_chat_test(config, model, BACKEND_KVINT, worker_id, {'quant_policy': 4})


@pytest.mark.order(6)
@pytest.mark.pipeline_chat
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_1
@pytest.mark.test_3090
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=1, quant_policy=8, model_type='vl_model'))
def test_pipeline_chat_kvint8_tp1(config, model, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id)
    run_pipeline_vl_chat_test(config, model, BACKEND_KVINT, worker_id, {'quant_policy': 8})


@pytest.mark.order(6)
@pytest.mark.pipeline_chat
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=2, quant_policy=8, model_type='vl_model'))
def test_pipeline_chat_kvint8_tp2(config, model, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, parallel_config=2)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    run_pipeline_vl_chat_test(config, model, BACKEND_KVINT, worker_id, {'quant_policy': 8})


@pytest.mark.order(6)
@pytest.mark.pipeline_chat
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_4
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=4, quant_policy=8, model_type='vl_model'))
def test_pipeline_chat_kvint8_tp4(config, model, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, parallel_config=4)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    run_pipeline_vl_chat_test(config, model, BACKEND_KVINT, worker_id, {'quant_policy': 8})
