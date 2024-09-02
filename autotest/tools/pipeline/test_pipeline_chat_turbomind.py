import os
from multiprocessing import Process

import pytest
from utils.config_utils import (get_all_model_list, get_cuda_id_by_workerid,
                                get_kvint_model_list)
from utils.pipeline_chat import (assert_pipeline_chat_log,
                                 run_pipeline_chat_test)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_all_model_list(tp_num=1))
def test_pipeline_chat_tp1(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id)
    p = Process(target=run_pipeline_chat_test,
                args=(config, common_case_config, model, 'turbomind',
                      worker_id))
    p.start()
    p.join()
    assert_pipeline_chat_log(config, common_case_config, model, 'turbomind',
                             worker_id)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_all_model_list(tp_num=2))
def test_pipeline_chat_tp2(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id,
                                                                     tp_num=2)
    p = Process(target=run_pipeline_chat_test,
                args=(config, common_case_config, model, 'turbomind',
                      worker_id))
    p.start()
    p.join()
    assert_pipeline_chat_log(config, common_case_config, model, 'turbomind',
                             worker_id)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_kvint_model_list(tp_num=1))
@pytest.mark.parametrize('quant_policy', (4, 8))
def test_pipeline_chat_kvint_tp1(config, common_case_config, model,
                                 quant_policy, worker_id):
    if quant_policy == 4 and 'Qwen2' in model:
        return  # kvint4 for qwen2 is not support
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id)
    p = Process(target=run_pipeline_chat_test,
                args=(config, common_case_config, model, 'turbomind-kvint',
                      worker_id, {
                          'quant_policy': quant_policy
                      }))
    p.start()
    p.join()
    assert_pipeline_chat_log(config, common_case_config, model,
                             'turbomind-kvint', worker_id)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_kvint_model_list(tp_num=2))
@pytest.mark.parametrize('quant_policy', (4, 8))
def test_pipeline_chat_kvint_tp2(config, common_case_config, model,
                                 quant_policy, worker_id):
    if quant_policy == 4 and 'Qwen2' in model:
        return  # kvint4 for qwen2 is not support
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id,
                                                                     tp_num=2)
    p = Process(target=run_pipeline_chat_test,
                args=(config, common_case_config, model, 'turbomind-kvint',
                      worker_id, {
                          'quant_policy': quant_policy
                      }))
    p.start()
    p.join()
    assert_pipeline_chat_log(config, common_case_config, model,
                             'turbomind-kvint', worker_id)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_2
@pytest.mark.pr_test
@pytest.mark.parametrize('model', [
    'internlm/internlm2_5-20b-chat',
    'internlm/internlm2_5-20b-chat-inner-4bits'
])
def test_pipeline_chat_pr(config, common_case_config, model):
    p = Process(target=run_pipeline_chat_test,
                args=(config, common_case_config, model, 'turbomind'))
    p.start()
    p.join()
    assert_pipeline_chat_log(config, common_case_config, model, 'turbomind')


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', ['Qwen/Qwen-7B-Chat'])
def test_modelscope_pipeline_chat_tp1(config, common_case_config, model,
                                      worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id)
    os.environ['LMDEPLOY_USE_MODELSCOPE'] = 'True'
    p = Process(target=run_pipeline_chat_test,
                args=(config, common_case_config, model, 'turbomind',
                      worker_id, None, False))
    p.start()
    p.join()
    del os.environ['LMDEPLOY_USE_MODELSCOPE']
    assert_pipeline_chat_log(config, common_case_config, model, 'turbomind',
                             worker_id)
