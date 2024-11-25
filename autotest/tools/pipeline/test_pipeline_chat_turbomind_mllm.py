import os
from multiprocessing import get_context

import pytest
from utils.config_utils import get_all_model_list, get_cuda_id_by_workerid
from utils.pipeline_chat import (assert_pipeline_vl_chat_log,
                                 run_pipeline_vl_chat_test)

BACKEND = 'turbomind'


@pytest.mark.order(6)
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('model',
                         get_all_model_list(tp_num=1, model_type='vl_model'))
def test_pipeline_chat_tp1(config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id)
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_vl_chat_test,
                              args=(config, model, BACKEND, worker_id))
    p.start()
    p.join()
    assert_pipeline_vl_chat_log(config, model, worker_id)


@pytest.mark.order(6)
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('model',
                         get_all_model_list(tp_num=2, model_type='vl_model'))
def test_pipeline_chat_tp2(config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id,
                                                                     tp_num=2)
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_vl_chat_test,
                              args=(config, model, BACKEND, worker_id))
    p.start()
    p.join()
    assert_pipeline_vl_chat_log(config, model, worker_id)


@pytest.mark.order(6)
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('model',
                         get_all_model_list(tp_num=1,
                                            quant_policy=4,
                                            model_type='vl_model'))
def test_pipeline_chat_kvint4_tp1(config, model, worker_id):
    if 'Qwen2' in model:
        return  # kvint4 for qwen2 is not support
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id)
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_vl_chat_test,
                              args=(config, model, BACKEND, worker_id, 4))
    p.start()
    p.join()
    assert_pipeline_vl_chat_log(config, model, worker_id)


@pytest.mark.order(6)
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('model',
                         get_all_model_list(tp_num=2,
                                            quant_policy=4,
                                            model_type='vl_model'))
def test_pipeline_chat_kvint4_tp2(config, model, worker_id):
    if 'Qwen2' in model:
        return  # kvint4 for qwen2 is not support
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id,
                                                                     tp_num=2)
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_vl_chat_test,
                              args=(config, model, BACKEND, worker_id, 4))
    p.start()
    p.join()
    assert_pipeline_vl_chat_log(config, model, worker_id)


@pytest.mark.order(6)
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('model',
                         get_all_model_list(tp_num=1,
                                            quant_policy=8,
                                            model_type='vl_model'))
def test_pipeline_chat_kvint8_tp1(config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id)
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_vl_chat_test,
                              args=(config, model, BACKEND, worker_id, 8))
    p.start()
    p.join()
    assert_pipeline_vl_chat_log(config, model, worker_id)


@pytest.mark.order(6)
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('model',
                         get_all_model_list(tp_num=2,
                                            quant_policy=8,
                                            model_type='vl_model'))
def test_pipeline_chat_kvint8_tp2(config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id,
                                                                     tp_num=2)
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_vl_chat_test,
                              args=(config, model, BACKEND, worker_id, 8))
    p.start()
    p.join()
    assert_pipeline_vl_chat_log(config, model, worker_id)


@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_1
@pytest.mark.pr_test
@pytest.mark.parametrize('model', [
    'liuhaotian/llava-v1.6-vicuna-7b', 'OpenGVLab/InternVL2-4B',
    'OpenGVLab/InternVL2-8B', 'internlm/internlm-xcomposer2d5-7b'
])
def test_pipeline_pr_test(config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(
            int(get_cuda_id_by_workerid(worker_id)) + 5)
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_vl_chat_test,
                              args=(config, model, BACKEND, worker_id))
    p.start()
    p.join()
    assert_pipeline_vl_chat_log(config, model, worker_id)
