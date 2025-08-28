import os

import pytest
from utils.config_utils import get_communicator_list, set_device_env_variable, get_turbomind_model_list
from utils.pipeline_chat import run_pipeline_chat_test


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_1
@pytest.mark.test_3090
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_turbomind_model_list(tp_num=1))
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_pipeline_chat_tp1(config, common_case_config, model, communicator, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id)
    run_pipeline_chat_test(config, common_case_config, model, 'turbomind', worker_id, {'communicator': communicator})


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_turbomind_model_list(tp_num=2))
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_pipeline_chat_tp2(config, common_case_config, model, communicator, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=2)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    run_pipeline_chat_test(config, common_case_config, model, 'turbomind', worker_id, {'communicator': communicator})


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_4
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_turbomind_model_list(tp_num=4))
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_pipeline_chat_tp4(config, common_case_config, model, communicator, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=4)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    run_pipeline_chat_test(config, common_case_config, model, 'turbomind', worker_id, {'communicator': communicator})


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_8
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_turbomind_model_list(tp_num=8))
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_pipeline_chat_tp8(config, common_case_config, model, communicator, worker_id):
    run_pipeline_chat_test(config, common_case_config, model, 'turbomind', worker_id, {'communicator': communicator})


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_1
@pytest.mark.test_3090
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_turbomind_model_list(tp_num=1, quant_policy=4))
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_pipeline_chat_kvint4_tp1(config, common_case_config, model, communicator, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id)
    run_pipeline_chat_test(config, common_case_config, model, 'turbomind-kvint', worker_id, {
        'quant_policy': 4,
        'communicator': communicator
    })


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_turbomind_model_list(tp_num=2, quant_policy=4))
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_pipeline_chat_kvint4_tp2(config, common_case_config, model, communicator, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=2)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    run_pipeline_chat_test(config, common_case_config, model, 'turbomind-kvint', worker_id, {
        'quant_policy': 4,
        'communicator': communicator
    })


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_4
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_turbomind_model_list(tp_num=4, quant_policy=4))
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_pipeline_chat_kvint4_tp4(config, common_case_config, model, communicator, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=4)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    run_pipeline_chat_test(config, common_case_config, model, 'turbomind-kvint', worker_id, {
        'quant_policy': 4,
        'communicator': communicator
    })


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_1
@pytest.mark.test_3090
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_turbomind_model_list(tp_num=1, quant_policy=8))
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_pipeline_chat_kvint8_tp1(config, common_case_config, model, communicator, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id)
    run_pipeline_chat_test(config, common_case_config, model, 'turbomind-kvint', worker_id, {
        'quant_policy': 8,
        'communicator': communicator
    })


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_turbomind_model_list(tp_num=2, quant_policy=8))
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_pipeline_chat_kvint8_tp2(config, common_case_config, model, communicator, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=2)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    run_pipeline_chat_test(config, common_case_config, model, 'turbomind-kvint', worker_id, {
        'quant_policy': 8,
        'communicator': communicator
    })


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_4
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_turbomind_model_list(tp_num=4, quant_policy=8))
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_pipeline_chat_kvint8_tp4(config, common_case_config, model, communicator, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=4)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    run_pipeline_chat_test(config, common_case_config, model, 'turbomind-kvint', worker_id, {
        'quant_policy': 8,
        'communicator': communicator
    })


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_8
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', get_turbomind_model_list(tp_num=8, quant_policy=8))
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_pipeline_chat_kvint8_tp8(config, common_case_config, model, communicator, worker_id):
    run_pipeline_chat_test(config, common_case_config, model, 'turbomind-kvint', worker_id, {
        'quant_policy': 8,
        'communicator': communicator
    })


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('model', [
    'microsoft/Phi-3-mini-4k-instruct', 'microsoft/Phi-3-mini-4k-instruct-inner-4bits',
    'microsoft/Phi-3-mini-4k-instruct-inner-w8a8'
])
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_pipeline_chat_fallback_backend_tp1(config, common_case_config, model, communicator, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=1)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    run_pipeline_chat_test(config,
                           common_case_config,
                           model,
                           'turbomind',
                           worker_id, {'communicator': communicator},
                           is_smoke=True)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('model', [
    'microsoft/Phi-3-mini-4k-instruct', 'microsoft/Phi-3-mini-4k-instruct-inner-4bits',
    'microsoft/Phi-3-mini-4k-instruct-inner-w8a8'
])
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_pipeline_chat_fallback_backend_kvint8_tp1(config, common_case_config, model, communicator, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=1)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    run_pipeline_chat_test(config,
                           common_case_config,
                           model,
                           'turbomind-kvint',
                           worker_id, {
                               'quant_policy': 8,
                               'communicator': communicator
                           },
                           is_smoke=True)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('model',
                         ['google/gemma-2-27b-it', 'deepseek-ai/deepseek-moe-16b-chat', 'Qwen/Qwen2.5-VL-32B-Instruct'])
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_pipeline_chat_fallback_backend_tp2(config, common_case_config, model, communicator, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=2)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    run_pipeline_chat_test(config,
                           common_case_config,
                           model,
                           'turbomind',
                           worker_id, {'communicator': communicator},
                           is_smoke=True)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('model',
                         ['google/gemma-2-27b-it', 'deepseek-ai/deepseek-moe-16b-chat', 'Qwen/Qwen2.5-VL-32B-Instruct'])
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_pipeline_chat_fallback_backend_kvint8_tp2(config, common_case_config, model, communicator, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id, tp_num=2)
        os.environ['MASTER_PORT'] = str(int(worker_id.replace('gw', '')) + 29500)
    run_pipeline_chat_test(config,
                           common_case_config,
                           model,
                           'turbomind-kvint',
                           worker_id, {
                               'quant_policy': 8,
                               'communicator': communicator
                           },
                           is_smoke=True)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_2
@pytest.mark.pr_test
@pytest.mark.parametrize('model', [
    'internlm/internlm2_5-20b-chat', 'internlm/internlm2_5-20b-chat-inner-4bits', 'mistralai/Mixtral-8x7B-Instruct-v0.1'
])
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_pipeline_chat_pr(config, common_case_config, model, communicator, worker_id):
    run_pipeline_chat_test(config,
                           common_case_config,
                           model,
                           'turbomind',
                           worker_id,
                           extra={'communicator': communicator},
                           is_smoke=True)


@pytest.mark.order(6)
@pytest.mark.usefixtures('common_case_config')
@pytest.mark.pipeline_chat
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('model', ['Qwen/Qwen2.5-7B-Instruct'])
def test_modelscope_pipeline_chat_tp1(config, common_case_config, model, worker_id):
    if 'gw' in worker_id:
        set_device_env_variable(worker_id)
    os.environ['LMDEPLOY_USE_MODELSCOPE'] = 'True'
    run_pipeline_chat_test(config, common_case_config, model, 'turbomind', worker_id, use_local_model=True)
    del os.environ['LMDEPLOY_USE_MODELSCOPE']
