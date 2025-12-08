import os

import allure
import pytest
from utils.config_utils import get_communicator_list, get_cuda_prefix_by_workerid, get_turbomind_model_list
from utils.run_client_chat import hf_command_line_test


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_turbomind_chat
@pytest.mark.gpu_num_1
@pytest.mark.test_3090
@pytest.mark.parametrize('model', get_turbomind_model_list(parallel_config=1))
def test_hf_turbomind_chat_tp1(config, model, cli_case_config, worker_id):
    usercase = 'chat_testcase'
    if 'coder' in model:
        usercase = 'code_testcase'
    result, chat_log, msg = hf_command_line_test(config,
                                                 usercase,
                                                 cli_case_config.get(usercase),
                                                 model,
                                                 'turbomind',
                                                 cuda_prefix=get_cuda_prefix_by_workerid(worker_id))

    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_turbomind_chat
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('model', get_turbomind_model_list(parallel_config=2))
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_hf_turbomind_chat_tp2(config, model, communicator, cli_case_config, worker_id):
    usercase = 'chat_testcase'
    result, chat_log, msg = hf_command_line_test(config,
                                                 usercase,
                                                 cli_case_config.get(usercase),
                                                 model,
                                                 'turbomind',
                                                 cuda_prefix=get_cuda_prefix_by_workerid(worker_id, parallel_config=2),
                                                 extra=f'--communicator {communicator}')

    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_turbomind_chat
@pytest.mark.gpu_num_4
@pytest.mark.parametrize('model', get_turbomind_model_list(parallel_config=4))
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_hf_turbomind_chat_tp4(config, model, communicator, cli_case_config, worker_id):
    usercase = 'chat_testcase'
    result, chat_log, msg = hf_command_line_test(config,
                                                 usercase,
                                                 cli_case_config.get(usercase),
                                                 model,
                                                 'turbomind',
                                                 cuda_prefix=get_cuda_prefix_by_workerid(worker_id, parallel_config=4),
                                                 extra=f'--communicator {communicator}')

    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_turbomind_chat
@pytest.mark.gpu_num_8
@pytest.mark.parametrize('model', get_turbomind_model_list(parallel_config=8))
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_hf_turbomind_chat_tp8(config, model, communicator, cli_case_config, worker_id):
    usercase = 'chat_testcase'
    result, chat_log, msg = hf_command_line_test(config,
                                                 usercase,
                                                 cli_case_config.get(usercase),
                                                 model,
                                                 'turbomind',
                                                 cuda_prefix=get_cuda_prefix_by_workerid(worker_id, parallel_config=8),
                                                 extra=f'--communicator {communicator}')

    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_turbomind_chat
@pytest.mark.gpu_num_1
@pytest.mark.test_3090
@pytest.mark.parametrize('model', get_turbomind_model_list(parallel_config=1, quant_policy=4))
def test_hf_turbomind_chat_kvint4_tp1(config, model, cli_case_config, worker_id):
    usercase = 'chat_testcase'
    if 'coder' in model:
        usercase = 'code_testcase'
    result, chat_log, msg = hf_command_line_test(config,
                                                 usercase,
                                                 cli_case_config.get(usercase),
                                                 model,
                                                 'turbomind',
                                                 cuda_prefix=get_cuda_prefix_by_workerid(worker_id),
                                                 extra='--quant-policy 4')

    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_turbomind_chat
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('model', get_turbomind_model_list(parallel_config=2, quant_policy=4))
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_hf_turbomind_chat_kvint4_tp2(config, model, communicator, cli_case_config, worker_id):
    usercase = 'chat_testcase'
    result, chat_log, msg = hf_command_line_test(config,
                                                 usercase,
                                                 cli_case_config.get(usercase),
                                                 model,
                                                 'turbomind',
                                                 cuda_prefix=get_cuda_prefix_by_workerid(worker_id, parallel_config=2),
                                                 extra=f'--communicator {communicator} --quant-policy 4')

    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_turbomind_chat
@pytest.mark.gpu_num_4
@pytest.mark.parametrize('model', get_turbomind_model_list(parallel_config=4, quant_policy=4))
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_hf_turbomind_chat_kvint4_tp4(config, model, communicator, cli_case_config, worker_id):
    usercase = 'chat_testcase'
    result, chat_log, msg = hf_command_line_test(config,
                                                 usercase,
                                                 cli_case_config.get(usercase),
                                                 model,
                                                 'turbomind',
                                                 cuda_prefix=get_cuda_prefix_by_workerid(worker_id, parallel_config=4),
                                                 extra=f'--communicator {communicator} --quant-policy 4')

    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_turbomind_chat
@pytest.mark.gpu_num_1
@pytest.mark.test_3090
@pytest.mark.parametrize('model', get_turbomind_model_list(parallel_config=1, quant_policy=8))
def test_hf_turbomind_chat_kvint8_tp1(config, model, cli_case_config, worker_id):
    usercase = 'chat_testcase'
    if 'coder' in model:
        usercase = 'code_testcase'
    result, chat_log, msg = hf_command_line_test(config,
                                                 usercase,
                                                 cli_case_config.get(usercase),
                                                 model,
                                                 'turbomind',
                                                 cuda_prefix=get_cuda_prefix_by_workerid(worker_id),
                                                 extra='--quant-policy 8')

    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_turbomind_chat
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('model', get_turbomind_model_list(parallel_config=2, quant_policy=8))
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_hf_turbomind_chat_kvint8_tp2(config, model, communicator, cli_case_config, worker_id):
    usercase = 'chat_testcase'
    result, chat_log, msg = hf_command_line_test(config,
                                                 usercase,
                                                 cli_case_config.get(usercase),
                                                 model,
                                                 'turbomind',
                                                 cuda_prefix=get_cuda_prefix_by_workerid(worker_id, parallel_config=2),
                                                 extra=f'--communicator {communicator} --quant-policy 8')

    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_turbomind_chat
@pytest.mark.gpu_num_4
@pytest.mark.parametrize('model', get_turbomind_model_list(parallel_config=4, quant_policy=8))
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_hf_turbomind_chat_kvint8_tp4(config, model, communicator, cli_case_config, worker_id):
    usercase = 'chat_testcase'
    result, chat_log, msg = hf_command_line_test(config,
                                                 usercase,
                                                 cli_case_config.get(usercase),
                                                 model,
                                                 'turbomind',
                                                 cuda_prefix=get_cuda_prefix_by_workerid(worker_id, parallel_config=4),
                                                 extra=f'--communicator {communicator} --quant-policy 8')

    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_turbomind_chat
@pytest.mark.gpu_num_8
@pytest.mark.parametrize('model', get_turbomind_model_list(parallel_config=8, quant_policy=8))
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_hf_turbomind_chat_kvint4_tp8(config, model, communicator, cli_case_config, worker_id):
    usercase = 'chat_testcase'
    result, chat_log, msg = hf_command_line_test(config,
                                                 usercase,
                                                 cli_case_config.get(usercase),
                                                 model,
                                                 'turbomind',
                                                 cuda_prefix=get_cuda_prefix_by_workerid(worker_id, parallel_config=8),
                                                 extra=f'--communicator {communicator} --quant-policy 8')

    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_turbomind_chat
@pytest.mark.gpu_num_1
@pytest.mark.other
@pytest.mark.parametrize('model', [
    'microsoft/Phi-3-mini-4k-instruct', 'microsoft/Phi-3-mini-4k-instruct-inner-4bits',
    'microsoft/Phi-3-mini-4k-instruct-inner-w8a8'
])
def test_hf_turbomind_chat_fallback_backend_tp1(config, model, cli_case_config, worker_id):
    usercase = 'chat_testcase'
    if 'coder' in model:
        usercase = 'code_testcase'
    result, chat_log, msg = hf_command_line_test(config,
                                                 usercase,
                                                 cli_case_config.get(usercase),
                                                 model,
                                                 'turbomind',
                                                 cuda_prefix=get_cuda_prefix_by_workerid(worker_id))

    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_turbomind_chat
@pytest.mark.gpu_num_1
@pytest.mark.other
@pytest.mark.parametrize('model', [
    'microsoft/Phi-3-mini-4k-instruct', 'microsoft/Phi-3-mini-4k-instruct-inner-4bits',
    'microsoft/Phi-3-mini-4k-instruct-inner-w8a8'
])
def test_hf_turbomind_chat_fallback_backend_kvint8_tp1(config, model, cli_case_config, worker_id):
    usercase = 'chat_testcase'
    if 'coder' in model:
        usercase = 'code_testcase'
    result, chat_log, msg = hf_command_line_test(config,
                                                 usercase,
                                                 cli_case_config.get(usercase),
                                                 model,
                                                 'turbomind',
                                                 cuda_prefix=get_cuda_prefix_by_workerid(worker_id),
                                                 extra='--quant-policy 8')

    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_turbomind_chat
@pytest.mark.gpu_num_2
@pytest.mark.other
@pytest.mark.parametrize('model', ['google/gemma-2-27b-it', 'deepseek-ai/deepseek-moe-16b-chat'])
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_hf_turbomind_chat_fallback_backend_tp2(config, model, communicator, cli_case_config, worker_id):
    usercase = 'chat_testcase'
    result, chat_log, msg = hf_command_line_test(config,
                                                 usercase,
                                                 cli_case_config.get(usercase),
                                                 model,
                                                 'turbomind',
                                                 cuda_prefix=get_cuda_prefix_by_workerid(worker_id, parallel_config=2),
                                                 extra=f'--communicator {communicator}')

    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_turbomind_chat
@pytest.mark.gpu_num_2
@pytest.mark.other
@pytest.mark.parametrize('model', ['google/gemma-2-27b-it', 'deepseek-ai/deepseek-moe-16b-chat'])
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_hf_turbomind_chat_fallback_backend_kvint8_tp2(config, model, communicator, cli_case_config, worker_id):
    usercase = 'chat_testcase'
    result, chat_log, msg = hf_command_line_test(config,
                                                 usercase,
                                                 cli_case_config.get(usercase),
                                                 model,
                                                 'turbomind',
                                                 cuda_prefix=get_cuda_prefix_by_workerid(worker_id, parallel_config=2),
                                                 extra=f'--communicator {communicator} --quant-policy 8')

    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_turbomind_chat
@pytest.mark.gpu_num_1
@pytest.mark.test_3090
@pytest.mark.parametrize('model', get_turbomind_model_list(parallel_config=1, model_type='base_model'))
def test_hf_turbomind_base_tp1(config, model, cli_case_config, worker_id):
    usercase = 'base_testcase'
    result, chat_log, msg = hf_command_line_test(config,
                                                 usercase,
                                                 cli_case_config.get(usercase),
                                                 model,
                                                 'turbomind',
                                                 cuda_prefix=get_cuda_prefix_by_workerid(worker_id))

    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_turbomind_chat
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('model', get_turbomind_model_list(parallel_config=2, model_type='base_model'))
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_hf_turbomind_base_tp2(config, model, communicator, cli_case_config, worker_id):
    usercase = 'base_testcase'
    result, chat_log, msg = hf_command_line_test(config,
                                                 usercase,
                                                 cli_case_config.get(usercase),
                                                 model,
                                                 'turbomind',
                                                 cuda_prefix=get_cuda_prefix_by_workerid(worker_id, parallel_config=2),
                                                 extra=f'--communicator {communicator}')

    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_turbomind_chat
@pytest.mark.gpu_num_2
@pytest.mark.pr_test
@pytest.mark.parametrize('model', [
    'internlm/internlm2_5-20b-chat', 'internlm/internlm2_5-20b-chat-inner-4bits', 'mistralai/Mixtral-8x7B-Instruct-v0.1'
])
@pytest.mark.parametrize('communicator', get_communicator_list())
def test_hf_turbomind_chat_pr(config, model, communicator, cli_case_config):
    usercase = 'chat_testcase'
    result, chat_log, msg = hf_command_line_test(config,
                                                 usercase,
                                                 cli_case_config.get(usercase),
                                                 model,
                                                 'turbomind',
                                                 cuda_prefix='CUDA_VISIBLE_DEVICES=5,6',
                                                 extra=f'--communicator {communicator}')

    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_turbomind_chat
@pytest.mark.gpu_num_1
@pytest.mark.pr_test
@pytest.mark.parametrize('model', ['OpenGVLab/InternVL3-8B'])
def test_hf_turbomind_chat_pr_gpu1(config, model, cli_case_config):
    usercase = 'chat_testcase'
    device_type = os.environ.get('DEVICE', 'cuda')
    if device_type == 'ascend':
        env_var = 'ASCEND_RT_VISIBLE_DEVICES='
    else:
        env_var = 'CUDA_VISIBLE_DEVICES='
    result, chat_log, msg = hf_command_line_test(config,
                                                 usercase,
                                                 cli_case_config.get(usercase),
                                                 model,
                                                 'turbomind',
                                                 cuda_prefix=env_var + '5,6')
    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_turbomind_chat
@pytest.mark.gpu_num_1
@pytest.mark.other
@pytest.mark.parametrize('model', ['Qwen/Qwen2.5-7B-Instruct'])
def test_modelscope_turbomind_chat_tp1(config, model, cli_case_config, worker_id):
    os.environ['LMDEPLOY_USE_MODELSCOPE'] = 'True'
    usercase = 'chat_testcase'
    result, chat_log, msg = hf_command_line_test(config,
                                                 usercase,
                                                 cli_case_config.get(usercase),
                                                 model,
                                                 'turbomind',
                                                 cuda_prefix=get_cuda_prefix_by_workerid(worker_id),
                                                 use_local_model=False)
    del os.environ['LMDEPLOY_USE_MODELSCOPE']

    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)
    assert result, msg
