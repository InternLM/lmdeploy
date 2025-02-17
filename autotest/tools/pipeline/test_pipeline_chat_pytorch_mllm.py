import os
from multiprocessing import get_context

import pytest
from utils.config_utils import get_cuda_id_by_workerid, get_torch_model_list
from utils.pipeline_chat import assert_pipeline_vl_chat_log, run_pipeline_vl_chat_test

BACKEND = 'pytorch'


@pytest.mark.order(6)
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('model', get_torch_model_list(tp_num=1, model_type='vl_model'))
def test_pipeline_chat_tp1(config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id)
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_vl_chat_test, args=(config, model, BACKEND, worker_id))
    p.start()
    p.join()
    assert_pipeline_vl_chat_log(config, model, worker_id)


@pytest.mark.order(6)
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('model', get_torch_model_list(tp_num=2, model_type='vl_model'))
def test_pipeline_chat_tp2(config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id, tp_num=2)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_vl_chat_test, args=(config, model, BACKEND, worker_id))
    p.start()
    p.join()
    assert_pipeline_vl_chat_log(config, model, worker_id)


@pytest.mark.order(6)
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_4
@pytest.mark.parametrize('model', get_torch_model_list(tp_num=4, model_type='vl_model'))
def test_pipeline_chat_tp4(config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id, tp_num=4)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_vl_chat_test, args=(config, model, BACKEND, worker_id))
    p.start()
    p.join()
    assert_pipeline_vl_chat_log(config, model, worker_id)


@pytest.mark.order(6)
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('model', get_torch_model_list(tp_num=1, quant_policy=4, model_type='vl_model'))
def test_pipeline_chat_kvint4_tp1(config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id)
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_vl_chat_test, args=(config, model, BACKEND, worker_id, 4))
    p.start()
    p.join()
    assert_pipeline_vl_chat_log(config, model, worker_id)


@pytest.mark.order(6)
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('model', get_torch_model_list(tp_num=2, quant_policy=4, model_type='vl_model'))
def test_pipeline_chat_kvint4_tp2(config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id, tp_num=2)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_vl_chat_test, args=(config, model, BACKEND, worker_id, 4))
    p.start()
    p.join()
    assert_pipeline_vl_chat_log(config, model, worker_id)


@pytest.mark.order(6)
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_4
@pytest.mark.parametrize('model', get_torch_model_list(tp_num=4, quant_policy=4, model_type='vl_model'))
def test_pipeline_chat_kvint4_tp4(config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id, tp_num=4)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_vl_chat_test, args=(config, model, BACKEND, worker_id, 4))
    p.start()
    p.join()
    assert_pipeline_vl_chat_log(config, model, worker_id)


@pytest.mark.order(6)
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('model', get_torch_model_list(tp_num=1, quant_policy=8, model_type='vl_model'))
def test_pipeline_chat_kvint8_tp1(config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id)
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_vl_chat_test, args=(config, model, BACKEND, worker_id, 8))
    p.start()
    p.join()
    assert_pipeline_vl_chat_log(config, model, worker_id)


@pytest.mark.order(6)
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('model', get_torch_model_list(tp_num=2, quant_policy=8, model_type='vl_model'))
def test_pipeline_chat_kvint8_tp2(config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id, tp_num=2)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_vl_chat_test, args=(config, model, BACKEND, worker_id, 8))
    p.start()
    p.join()
    assert_pipeline_vl_chat_log(config, model, worker_id)


@pytest.mark.order(6)
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_4
@pytest.mark.parametrize('model', get_torch_model_list(tp_num=4, quant_policy=8, model_type='vl_model'))
def test_pipeline_chat_kvint8_tp4(config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id, tp_num=4)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_vl_chat_test, args=(config, model, BACKEND, worker_id, 8))
    p.start()
    p.join()
    assert_pipeline_vl_chat_log(config, model, worker_id)
