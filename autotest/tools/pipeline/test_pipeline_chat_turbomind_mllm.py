import os

import pytest
from utils.config_utils import get_communicator_list, get_cuda_id_by_workerid, get_turbomind_model_list
from utils.pipeline_chat import run_pipeline_vl_chat_test

BACKEND = 'turbomind'
BACKEND_KVINT = 'turbomind-kvint'


@pytest.mark.order(6)
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('model', get_turbomind_model_list(tp_num=1, model_type='vl_model'))
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_pipeline_chat_tp1(config, model, communicator, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id)
    run_pipeline_vl_chat_test(config, model, BACKEND, worker_id, {'communicator': communicator})


@pytest.mark.order(6)
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('model', get_turbomind_model_list(tp_num=2, model_type='vl_model'))
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_pipeline_chat_tp2(config, model, communicator, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id, tp_num=2)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    run_pipeline_vl_chat_test(config, model, BACKEND, worker_id, {'communicator': communicator})


@pytest.mark.order(6)
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_4
@pytest.mark.parametrize('model', get_turbomind_model_list(tp_num=4, model_type='vl_model'))
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_pipeline_chat_tp4(config, model, communicator, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id, tp_num=4)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    run_pipeline_vl_chat_test(config, model, BACKEND, worker_id, {'communicator': communicator})


@pytest.mark.order(6)
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('model', get_turbomind_model_list(tp_num=1, quant_policy=4, model_type='vl_model'))
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_pipeline_chat_kvint4_tp1(config, model, communicator, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id)
    run_pipeline_vl_chat_test(config, model, BACKEND_KVINT, worker_id, {
        'quant_policy': 4,
        'communicator': communicator
    })


@pytest.mark.order(6)
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('model', get_turbomind_model_list(tp_num=2, quant_policy=4, model_type='vl_model'))
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_pipeline_chat_kvint4_tp2(config, model, communicator, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id, tp_num=2)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    run_pipeline_vl_chat_test(config, model, BACKEND_KVINT, worker_id, {
        'quant_policy': 4,
        'communicator': communicator
    })


@pytest.mark.order(6)
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_4
@pytest.mark.parametrize('model', get_turbomind_model_list(tp_num=4, quant_policy=4, model_type='vl_model'))
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_pipeline_chat_kvint4_tp4(config, model, communicator, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id, tp_num=4)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    run_pipeline_vl_chat_test(config, model, BACKEND_KVINT, worker_id, {
        'quant_policy': 4,
        'communicator': communicator
    })


@pytest.mark.order(6)
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('model', get_turbomind_model_list(tp_num=1, quant_policy=8, model_type='vl_model'))
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_pipeline_chat_kvint8_tp1(config, model, communicator, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id)
    run_pipeline_vl_chat_test(config, model, BACKEND_KVINT, worker_id, {
        'quant_policy': 8,
        'communicator': communicator
    })


@pytest.mark.order(6)
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('model', get_turbomind_model_list(tp_num=2, quant_policy=8, model_type='vl_model'))
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_pipeline_chat_kvint8_tp2(config, model, communicator, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id, tp_num=2)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    run_pipeline_vl_chat_test(config, model, BACKEND_KVINT, worker_id, {
        'quant_policy': 8,
        'communicator': communicator
    })


@pytest.mark.order(6)
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_4
@pytest.mark.parametrize('model', get_turbomind_model_list(tp_num=4, quant_policy=8, model_type='vl_model'))
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_pipeline_chat_kvint8_tp4(config, model, communicator, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id, tp_num=4)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    run_pipeline_vl_chat_test(config, model, BACKEND_KVINT, worker_id, {
        'quant_policy': 8,
        'communicator': communicator
    })


@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_1
@pytest.mark.pr_test
@pytest.mark.parametrize('model', [
    'liuhaotian/llava-v1.6-vicuna-7b', 'OpenGVLab/InternVL2-4B', 'OpenGVLab/InternVL2-8B',
    'internlm/internlm-xcomposer2d5-7b'
])
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_pipeline_pr_test(config, model, communicator, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(int(get_cuda_id_by_workerid(worker_id)) + 5)
    run_pipeline_vl_chat_test(config, model, BACKEND, worker_id, {'communicator': communicator}, is_pr_test=True)
