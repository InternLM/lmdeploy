import os
from multiprocessing import get_context

import pytest
from utils.config_utils import get_cuda_id_by_workerid, get_torch_model_list
from utils.pipeline_chat import assert_pipeline_chat_log, run_pipeline_chat_test


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat_pytorch
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_torch_model_list(tp_num=1, exclude_dup=True))
def test_pipeline_chat_pytorch_tp1(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id)
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_chat_test,
                              args=(config, common_case_config, model, 'pytorch', worker_id))
    p.start()
    p.join()

    # assert script
    assert_pipeline_chat_log(config, common_case_config, model, 'pytorch', worker_id)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat_pytorch
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_torch_model_list(tp_num=2, exclude_dup=True))
def test_pipeline_chat_pytorch_tp2(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id, tp_num=2)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_chat_test,
                              args=(config, common_case_config, model, 'pytorch', worker_id))
    p.start()
    p.join()

    # assert script
    assert_pipeline_chat_log(config, common_case_config, model, 'pytorch', worker_id)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat_pytorch
@pytest.mark.gpu_num_4
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_torch_model_list(tp_num=4, exclude_dup=True))
def test_pipeline_chat_pytorch_tp4(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id, tp_num=4)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_chat_test,
                              args=(config, common_case_config, model, 'pytorch', worker_id))
    p.start()
    p.join()

    # assert script
    assert_pipeline_chat_log(config, common_case_config, model, 'pytorch', worker_id)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_torch_model_list(tp_num=1, quant_policy=4, exclude_dup=True))
def test_pipeline_chat_kvint4_tp1(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id)
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_chat_test,
                              args=(config, common_case_config, model, 'pytorch-kvint', worker_id, {
                                  'quant_policy': 4
                              }))
    p.start()
    p.join()
    assert_pipeline_chat_log(config, common_case_config, model, 'pytorch-kvint', worker_id)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_torch_model_list(tp_num=2, quant_policy=4, exclude_dup=True))
def test_pipeline_chat_kvint4_tp2(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id, tp_num=2)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_chat_test,
                              args=(config, common_case_config, model, 'pytorch-kvint', worker_id, {
                                  'quant_policy': 4
                              }))
    p.start()
    p.join()
    assert_pipeline_chat_log(config, common_case_config, model, 'pytorch-kvint', worker_id)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_4
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_torch_model_list(tp_num=4, quant_policy=4, exclude_dup=True))
def test_pipeline_chat_kvint4_tp4(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id, tp_num=4)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_chat_test,
                              args=(config, common_case_config, model, 'pytorch-kvint', worker_id, {
                                  'quant_policy': 4
                              }))
    p.start()
    p.join()
    assert_pipeline_chat_log(config, common_case_config, model, 'pytorch-kvint', worker_id)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_torch_model_list(tp_num=1, quant_policy=8, exclude_dup=True))
def test_pipeline_chat_kvint8_tp1(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id)
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_chat_test,
                              args=(config, common_case_config, model, 'pytorch-kvint', worker_id, {
                                  'quant_policy': 8
                              }))
    p.start()
    p.join()
    assert_pipeline_chat_log(config, common_case_config, model, 'pytorch-kvint', worker_id)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_torch_model_list(tp_num=2, quant_policy=8, exclude_dup=True))
def test_pipeline_chat_kvint8_tp2(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id, tp_num=2)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_chat_test,
                              args=(config, common_case_config, model, 'pytorch-kvint', worker_id, {
                                  'quant_policy': 8
                              }))
    p.start()
    p.join()
    assert_pipeline_chat_log(config, common_case_config, model, 'pytorch-kvint', worker_id)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_4
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_torch_model_list(tp_num=4, quant_policy=8, exclude_dup=True))
def test_pipeline_chat_kvint8_tp4(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id, tp_num=4)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_chat_test,
                              args=(config, common_case_config, model, 'pytorch-kvint', worker_id, {
                                  'quant_policy': 8
                              }))
    p.start()
    p.join()
    assert_pipeline_chat_log(config, common_case_config, model, 'pytorch-kvint', worker_id)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat_pytorch
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_2
@pytest.mark.pr_test
@pytest.mark.parametrize('model', ['internlm/internlm2_5-20b-chat', 'mistralai/Mixtral-8x7B-Instruct-v0.1'])
def test_pipeline_chat_pytorch_pr(config, common_case_config, model):
    spawn_context = get_context('spawn')
    case_config = {k: v for k, v in common_case_config.items() if k == 'memory_test'}
    p = spawn_context.Process(target=run_pipeline_chat_test, args=(config, case_config, model, 'pytorch'))
    p.start()
    p.join()

    # assert script
    assert_pipeline_chat_log(config, case_config, model, 'pytorch')


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat_pytorch
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', ['Qwen/Qwen2.5-7B-Instruct'])
def test_modelscope_pipeline_chat_pytorch_tp1(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id)
    os.environ['LMDEPLOY_USE_MODELSCOPE'] = 'True'
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_chat_test,
                              args=(config, common_case_config, model, 'pytorch', worker_id, None, False))
    p.start()
    p.join()
    del os.environ['LMDEPLOY_USE_MODELSCOPE']

    # assert script
    assert_pipeline_chat_log(config, common_case_config, model, 'pytorch', worker_id)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat_pytorch
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', ['meta-llama/Llama-2-7b-chat-hf'])
def test_pipeline_chat_pytorch_with_lora_tp1(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id)
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_chat_test,
                              args=(config, common_case_config, model, 'pytorch_lora', worker_id, {
                                  'adapters': {
                                      'adapter0': 'lora/Llama2-Chinese-7b-Chat-LoRA'
                                  }
                              }))
    p.start()
    p.join()

    # assert script
    assert_pipeline_chat_log(config, common_case_config, model, 'pytorch_lora', worker_id)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat_pytorch
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', ['baichuan-inc/Baichuan2-13B-Chat'])
def test_pipeline_chat_pytorch_with_lora_tp2(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_id_by_workerid(worker_id, tp_num=2)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    spawn_context = get_context('spawn')
    p = spawn_context.Process(target=run_pipeline_chat_test,
                              args=(config, common_case_config, model, 'pytorch_lora', worker_id, {
                                  'adapters': {
                                      'adapter0': 'lora/2024-01-25_self_dup',
                                      'adapter1': 'lora/2024-01-25_self'
                                  }
                              }))
    p.start()
    p.join()

    # assert script
    assert_pipeline_chat_log(config, common_case_config, model, 'pytorch_lora', worker_id)
