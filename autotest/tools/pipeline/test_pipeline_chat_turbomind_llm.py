import os
from multiprocessing import get_context

import pytest
from utils.config_utils import get_all_model_list, get_cuda_id_by_workerid
from utils.pipeline_chat import assert_pipeline_chat_log, run_pipeline_chat_test


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_all_model_list(tp_num=1))
def test_pipeline_chat_tp1(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id)
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_chat_test,
                              args=(config, common_case_config, model, 'turbomind', worker_id))
    p.start()
    p.join()
    assert_pipeline_chat_log(config, common_case_config, model, 'turbomind', worker_id)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_all_model_list(tp_num=2))
def test_pipeline_chat_tp2(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id, tp_num=2)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_chat_test,
                              args=(config, common_case_config, model, 'turbomind', worker_id))
    p.start()
    p.join()
    assert_pipeline_chat_log(config, common_case_config, model, 'turbomind', worker_id)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_4
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_all_model_list(tp_num=4))
def test_pipeline_chat_tp4(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id, tp_num=4)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_chat_test,
                              args=(config, common_case_config, model, 'turbomind', worker_id))
    p.start()
    p.join()
    assert_pipeline_chat_log(config, common_case_config, model, 'turbomind', worker_id)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_all_model_list(tp_num=1, quant_policy=4))
def test_pipeline_chat_kvint4_tp1(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id)
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_chat_test,
                              args=(config, common_case_config, model, 'turbomind-kvint', worker_id, {
                                  'quant_policy': 4
                              }))
    p.start()
    p.join()
    assert_pipeline_chat_log(config, common_case_config, model, 'turbomind-kvint', worker_id)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_all_model_list(tp_num=2, quant_policy=4))
def test_pipeline_chat_kvint4_tp2(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id, tp_num=2)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_chat_test,
                              args=(config, common_case_config, model, 'turbomind-kvint', worker_id, {
                                  'quant_policy': 4
                              }))
    p.start()
    p.join()
    assert_pipeline_chat_log(config, common_case_config, model, 'turbomind-kvint', worker_id)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_4
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_all_model_list(tp_num=4, quant_policy=4))
def test_pipeline_chat_kvint4_tp4(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id, tp_num=4)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_chat_test,
                              args=(config, common_case_config, model, 'turbomind-kvint', worker_id, {
                                  'quant_policy': 4
                              }))
    p.start()
    p.join()
    assert_pipeline_chat_log(config, common_case_config, model, 'turbomind-kvint', worker_id)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_all_model_list(tp_num=1, quant_policy=8))
def test_pipeline_chat_kvint8_tp1(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id)
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_chat_test,
                              args=(config, common_case_config, model, 'turbomind-kvint', worker_id, {
                                  'quant_policy': 8
                              }))
    p.start()
    p.join()
    assert_pipeline_chat_log(config, common_case_config, model, 'turbomind-kvint', worker_id)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_all_model_list(tp_num=2, quant_policy=8))
def test_pipeline_chat_kvint8_tp2(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id, tp_num=2)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_chat_test,
                              args=(config, common_case_config, model, 'turbomind-kvint', worker_id, {
                                  'quant_policy': 8
                              }))
    p.start()
    p.join()
    assert_pipeline_chat_log(config, common_case_config, model, 'turbomind-kvint', worker_id)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_4
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_all_model_list(tp_num=4, quant_policy=8))
def test_pipeline_chat_kvint8_tp4(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id, tp_num=4)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_chat_test,
                              args=(config, common_case_config, model, 'turbomind-kvint', worker_id, {
                                  'quant_policy': 8
                              }))
    p.start()
    p.join()
    assert_pipeline_chat_log(config, common_case_config, model, 'turbomind-kvint', worker_id)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_2
@pytest.mark.pr_test
@pytest.mark.parametrize('model', [
    'internlm/internlm2_5-20b-chat', 'internlm/internlm2_5-20b-chat-inner-4bits', 'mistralai/Mixtral-8x7B-Instruct-v0.1'
])
def test_pipeline_chat_pr(config, common_case_config, model):
    spawn_context = get_context('spawn')
    case_config = {k: v for k, v in common_case_config.items() if k == 'memory_test'}
    p = spawn_context.Process(target=run_pipeline_chat_test, args=(config, case_config, model, 'turbomind'))
    p.start()
    p.join()
    assert_pipeline_chat_log(config, case_config, model, 'turbomind')


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', ['Qwen/Qwen2.5-7B-Instruct'])
def test_modelscope_pipeline_chat_tp1(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id)
    os.environ['LMDEPLOY_USE_MODELSCOPE'] = 'True'
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_chat_test,
                              args=(config, common_case_config, model, 'turbomind', worker_id, None, False))
    p.start()
    p.join()
    del os.environ['LMDEPLOY_USE_MODELSCOPE']
    assert_pipeline_chat_log(config, common_case_config, model, 'turbomind', worker_id)
