import os

import pytest
from utils.config_utils import get_torch_model_list, set_device_env_variable
from utils.pipeline_chat import run_pipeline_chat_test


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.prefix_cache_test
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=1, exclude_dup=True))
def test_pipeline_chat_pytorch_prefix_cache_tp1(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id)
    run_pipeline_chat_test(config,
                           common_case_config,
                           model,
                           'pytorch',
                           worker_id,
                           extra={'enable_prefix_caching': True})


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat_pytorch
@pytest.mark.gpu_num_1
@pytest.mark.test_3090
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=1, exclude_dup=True))
def test_pipeline_chat_pytorch_tp1(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id)
    run_pipeline_chat_test(config, common_case_config, model, 'pytorch', worker_id)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat_pytorch
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.test_ascend
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=2, exclude_dup=True))
def test_pipeline_chat_pytorch_tp2(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, parallel_config=2)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    run_pipeline_chat_test(config, common_case_config, model, 'pytorch', worker_id)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat_pytorch
@pytest.mark.gpu_num_4
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=4, exclude_dup=True))
def test_pipeline_chat_pytorch_tp4(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, parallel_config=4)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    run_pipeline_chat_test(config, common_case_config, model, 'pytorch', worker_id)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat_pytorch
@pytest.mark.gpu_num_8
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=8, exclude_dup=True))
def test_pipeline_chat_pytorch_tp8(config, common_case_config, model, worker_id):
    run_pipeline_chat_test(config, common_case_config, model, 'pytorch', worker_id)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat_pytorch
@pytest.mark.gpu_num_16
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=16, exclude_dup=True))
def test_pipeline_chat_pytorch_tp16(config, common_case_config, model, worker_id):
    run_pipeline_chat_test(config, common_case_config, model, 'pytorch', worker_id)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_1
@pytest.mark.test_3090
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=1, quant_policy=4, exclude_dup=True))
def test_pipeline_chat_kvint4_tp1(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id)
    run_pipeline_chat_test(config, common_case_config, model, 'pytorch-kvint', worker_id, {'quant_policy': 4})


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=2, quant_policy=4, exclude_dup=True))
def test_pipeline_chat_kvint4_tp2(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, parallel_config=2)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    run_pipeline_chat_test(config, common_case_config, model, 'pytorch-kvint', worker_id, {'quant_policy': 4})


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_4
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=4, quant_policy=4, exclude_dup=True))
def test_pipeline_chat_kvint4_tp4(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, parallel_config=4)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    run_pipeline_chat_test(config, common_case_config, model, 'pytorch-kvint', worker_id, {'quant_policy': 4})


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_1
@pytest.mark.test_3090
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=1, quant_policy=8, exclude_dup=True))
def test_pipeline_chat_kvint8_tp1(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id)
    run_pipeline_chat_test(config, common_case_config, model, 'pytorch-kvint', worker_id, {'quant_policy': 8})


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=2, quant_policy=8, exclude_dup=True))
def test_pipeline_chat_kvint8_tp2(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, parallel_config=2)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    run_pipeline_chat_test(config, common_case_config, model, 'pytorch-kvint', worker_id, {'quant_policy': 8})


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_4
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=4, quant_policy=8, exclude_dup=True))
def test_pipeline_chat_kvint8_tp4(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, parallel_config=4)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    run_pipeline_chat_test(config, common_case_config, model, 'pytorch-kvint', worker_id, {'quant_policy': 8})


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_8
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=8, quant_policy=8, exclude_dup=True))
def test_pipeline_chat_kvint8_tp8(config, common_case_config, model, worker_id):
    run_pipeline_chat_test(config, common_case_config, model, 'pytorch-kvint', worker_id, {'quant_policy': 8})


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat_pytorch
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_2
@pytest.mark.pr_test
@pytest.mark.parametrize('model', ['internlm/internlm2_5-20b-chat', 'mistralai/Mixtral-8x7B-Instruct-v0.1'])
def test_pipeline_chat_pytorch_pr(config, common_case_config, model, worker_id):
    run_pipeline_chat_test(config, common_case_config, model, 'pytorch', worker_id, is_smoke=True)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat_pytorch
@pytest.mark.gpu_num_1
@pytest.mark.other
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', ['Qwen/Qwen2.5-7B-Instruct'])
def test_modelscope_pipeline_chat_pytorch_tp1(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id)
    os.environ['LMDEPLOY_USE_MODELSCOPE'] = 'True'
    run_pipeline_chat_test(config, common_case_config, model, 'pytorch', worker_id, use_local_model=True)
    del os.environ['LMDEPLOY_USE_MODELSCOPE']


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat_pytorch
@pytest.mark.gpu_num_1
@pytest.mark.other
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', ['meta-llama/Llama-2-7b-chat-hf'])
def test_pipeline_chat_pytorch_with_lora_tp1(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id)
    run_pipeline_chat_test(config, common_case_config, model, 'pytorch_lora', worker_id,
                           {'adapters': {
                               'adapter0': 'lora/Llama2-Chinese-7b-Chat-LoRA'
                           }})


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat_pytorch
@pytest.mark.gpu_num_2
@pytest.mark.other
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', ['baichuan-inc/Baichuan2-13B-Chat'])
def test_pipeline_chat_pytorch_with_lora_tp2(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, parallel_config=2)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    run_pipeline_chat_test(config, common_case_config, model, 'pytorch_lora', worker_id,
                           {'adapters': {
                               'adapter0': 'lora/2024-01-25_self_dup',
                               'adapter1': 'lora/2024-01-25_self'
                           }})


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat_pytorch
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_2
@pytest.mark.pr_test
@pytest.mark.parametrize('model', ['internlm/internlm2_5-20b-chat'])
def test_pipeline_chat_pytorch_kvint_pr(config, common_case_config, model, worker_id):
    run_pipeline_chat_test(config,
                           common_case_config,
                           model,
                           'pytorch-kvint',
                           worker_id,
                           extra={'quant_policy': 4},
                           is_smoke=True)
