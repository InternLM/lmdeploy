import os
from multiprocessing import Process

import pytest
from utils.config_utils import get_cuda_id_by_workerid, get_torch_model_list
from utils.pipeline_chat import (assert_pipeline_chat_log,
                                 run_pipeline_chat_test)


def getModelList(tp_num):
    return [
        item for item in get_torch_model_list(tp_num)
        if 'falcon' not in item.lower() and 'chatglm2' not in item.lower()
    ]


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat_pytorch
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', getModelList(tp_num=1))
def test_pipeline_chat_pytorch_tp1(config, common_case_config, model,
                                   worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id)
    p = Process(target=run_pipeline_chat_test,
                args=(config, common_case_config, model, 'pytorch'))
    p.start()
    p.join()

    # assert script
    assert_pipeline_chat_log(config, common_case_config, model, 'pytorch')


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat_pytorch
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', getModelList(tp_num=2))
def test_pipeline_chat_pytorch_tp2(config, common_case_config, model,
                                   worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id,
                                                                     tp_num=2)
    p = Process(target=run_pipeline_chat_test,
                args=(config, common_case_config, model, 'pytorch'))
    p.start()
    p.join()

    # assert script
    assert_pipeline_chat_log(config, common_case_config, model, 'pytorch')


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat_pytorch
@pytest.mark.flaky(reruns=0)
@pytest.mark.pr_test
@pytest.mark.parametrize('model', ['internlm/internlm2-chat-20b'])
def test_pipeline_chat_pytorch_pr(config, common_case_config, model):
    p = Process(target=run_pipeline_chat_test,
                args=(config, common_case_config, model, 'pytorch'))
    p.start()
    p.join()

    # assert script
    assert_pipeline_chat_log(config, common_case_config, model, 'pytorch')


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat_pytorch
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', ['Qwen/Qwen-7B-Chat'])
def test_modelscope_pipeline_chat_pytorch_tp1(config, common_case_config,
                                              model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id)
    os.environ['LMDEPLOY_USE_MODELSCOPE'] = 'True'
    p = Process(target=run_pipeline_chat_test,
                args=(config, common_case_config, model, 'pytorch', None,
                      False))
    p.start()
    p.join()
    del os.environ['LMDEPLOY_USE_MODELSCOPE']

    # assert script
    assert_pipeline_chat_log(config, common_case_config, model, 'pytorch')


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat_pytorch
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', ['meta-llama/Llama-2-7b-chat-hf'])
def test_pipeline_chat_pytorch_with_lora_tp1(config, common_case_config, model,
                                             worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id)
    p = Process(target=run_pipeline_chat_test,
                args=(config, common_case_config, model, 'pytorch_lora', {
                    'adapters': {
                        'adapter0': 'lora/Llama2-Chinese-7b-Chat-LoRA'
                    }
                }))
    p.start()
    p.join()

    # assert script
    assert_pipeline_chat_log(config, common_case_config, model, 'pytorch_lora')


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat_pytorch
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', ['baichuan-inc/Baichuan2-13B-Chat'])
def test_pipeline_chat_pytorch_with_lora_tp2(config, common_case_config, model,
                                             worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id,
                                                                     tp_num=2)
    p = Process(target=run_pipeline_chat_test,
                args=(config, common_case_config, model, 'pytorch_lora', {
                    'adapters': {
                        'adapter0': 'lora/2024-01-25_self_dup',
                        'adapter1': 'lora/2024-01-25_self'
                    }
                }))
    p.start()
    p.join()

    # assert script
    assert_pipeline_chat_log(config, common_case_config, model, 'pytorch_lora')
