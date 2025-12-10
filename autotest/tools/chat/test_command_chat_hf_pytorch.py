import os

import allure
import pytest
from utils.config_utils import get_cuda_prefix_by_workerid, get_torch_model_list
from utils.run_client_chat import hf_command_line_test


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_pytorch_chat
@pytest.mark.gpu_num_1
@pytest.mark.test_3090
@pytest.mark.test_ascend
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=1))
def test_hf_pytorch_chat_tp1(config, model, cli_case_config, worker_id):
    usercase = 'chat_testcase'
    if 'coder' in model:
        usercase = 'code_testcase'
    result, chat_log, msg = hf_command_line_test(config,
                                                 usercase,
                                                 cli_case_config.get(usercase),
                                                 model,
                                                 'pytorch',
                                                 cuda_prefix=get_cuda_prefix_by_workerid(worker_id))
    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_pytorch_chat
@pytest.mark.gpu_num_2
@pytest.mark.test_ascend
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=2))
def test_hf_pytorch_chat_tp2(config, model, cli_case_config, worker_id):
    usercase = 'chat_testcase'
    result, chat_log, msg = hf_command_line_test(config,
                                                 usercase,
                                                 cli_case_config.get(usercase),
                                                 model,
                                                 'pytorch',
                                                 cuda_prefix=get_cuda_prefix_by_workerid(worker_id, parallel_config=2))
    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_pytorch_chat
@pytest.mark.gpu_num_4
@pytest.mark.test_ascend
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=4))
def test_hf_pytorch_chat_tp4(config, model, cli_case_config, worker_id):
    usercase = 'chat_testcase'
    result, chat_log, msg = hf_command_line_test(config,
                                                 usercase,
                                                 cli_case_config.get(usercase),
                                                 model,
                                                 'pytorch',
                                                 cuda_prefix=get_cuda_prefix_by_workerid(worker_id, parallel_config=4))
    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_pytorch_chat
@pytest.mark.gpu_num_8
@pytest.mark.test_ascend
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=8))
def test_hf_pytorch_chat_tp8(config, model, cli_case_config, worker_id):
    usercase = 'chat_testcase'
    result, chat_log, msg = hf_command_line_test(config,
                                                 usercase,
                                                 cli_case_config.get(usercase),
                                                 model,
                                                 'pytorch',
                                                 cuda_prefix=get_cuda_prefix_by_workerid(worker_id, parallel_config=8))
    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_pytorch_chat
@pytest.mark.gpu_num_16
@pytest.mark.test_ascend
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=16))
def test_hf_pytorch_chat_tp16(config, model, cli_case_config, worker_id):
    usercase = 'chat_testcase'
    result, chat_log, msg = hf_command_line_test(config,
                                                 usercase,
                                                 cli_case_config.get(usercase),
                                                 model,
                                                 'pytorch',
                                                 cuda_prefix=get_cuda_prefix_by_workerid(worker_id, parallel_config=16))
    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_pytorch_chat
@pytest.mark.gpu_num_1
@pytest.mark.test_3090
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=1, quant_policy=4))
def test_hf_pytorch_chat_kvin4_tp1(config, model, cli_case_config, worker_id):
    usercase = 'chat_testcase'
    if 'coder' in model:
        usercase = 'code_testcase'
    result, chat_log, msg = hf_command_line_test(config,
                                                 usercase,
                                                 cli_case_config.get(usercase),
                                                 model,
                                                 'pytorch',
                                                 cuda_prefix=get_cuda_prefix_by_workerid(worker_id),
                                                 extra='--quant-policy 4')
    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_pytorch_chat
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=2, quant_policy=4))
def test_hf_pytorch_chat_kvin4_tp2(config, model, cli_case_config, worker_id):
    usercase = 'chat_testcase'
    result, chat_log, msg = hf_command_line_test(config,
                                                 usercase,
                                                 cli_case_config.get(usercase),
                                                 model,
                                                 'pytorch',
                                                 cuda_prefix=get_cuda_prefix_by_workerid(worker_id, parallel_config=2),
                                                 extra='--quant-policy 4')
    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_pytorch_chat
@pytest.mark.gpu_num_4
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=4, quant_policy=4))
def test_hf_pytorch_chat_kvin4_tp4(config, model, cli_case_config, worker_id):
    usercase = 'chat_testcase'
    result, chat_log, msg = hf_command_line_test(config,
                                                 usercase,
                                                 cli_case_config.get(usercase),
                                                 model,
                                                 'pytorch',
                                                 cuda_prefix=get_cuda_prefix_by_workerid(worker_id, parallel_config=4),
                                                 extra='--quant-policy 4')
    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_pytorch_chat
@pytest.mark.gpu_num_1
@pytest.mark.test_3090
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=1, quant_policy=8))
def test_hf_pytorch_chat_kvin8_tp1(config, model, cli_case_config, worker_id):
    usercase = 'chat_testcase'
    if 'coder' in model:
        usercase = 'code_testcase'
    result, chat_log, msg = hf_command_line_test(config,
                                                 usercase,
                                                 cli_case_config.get(usercase),
                                                 model,
                                                 'pytorch',
                                                 cuda_prefix=get_cuda_prefix_by_workerid(worker_id),
                                                 extra='--quant-policy 8')
    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_pytorch_chat
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=2, quant_policy=8))
def test_hf_pytorch_chat_kvin8_tp2(config, model, cli_case_config, worker_id):
    usercase = 'chat_testcase'
    result, chat_log, msg = hf_command_line_test(config,
                                                 usercase,
                                                 cli_case_config.get(usercase),
                                                 model,
                                                 'pytorch',
                                                 cuda_prefix=get_cuda_prefix_by_workerid(worker_id, parallel_config=2),
                                                 extra='--quant-policy 8')
    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_pytorch_chat
@pytest.mark.gpu_num_4
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=4, quant_policy=8))
def test_hf_pytorch_chat_kvin8_tp4(config, model, cli_case_config, worker_id):
    usercase = 'chat_testcase'
    result, chat_log, msg = hf_command_line_test(config,
                                                 usercase,
                                                 cli_case_config.get(usercase),
                                                 model,
                                                 'pytorch',
                                                 cuda_prefix=get_cuda_prefix_by_workerid(worker_id, parallel_config=4),
                                                 extra='--quant-policy 8')
    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_pytorch_chat
@pytest.mark.gpu_num_8
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=8, quant_policy=8))
def test_hf_pytorch_chat_kvint8_tp8(config, model, cli_case_config, worker_id):
    usercase = 'chat_testcase'
    result, chat_log, msg = hf_command_line_test(config,
                                                 usercase,
                                                 cli_case_config.get(usercase),
                                                 model,
                                                 'pytorch',
                                                 cuda_prefix=get_cuda_prefix_by_workerid(worker_id, parallel_config=8),
                                                 extra='--quant-policy 8')
    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_turbomind_chat
@pytest.mark.test_3090
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=1, model_type='base_model'))
def test_hf_pytorch_base_tp1(config, model, cli_case_config, worker_id):
    usercase = 'base_testcase'
    result, chat_log, msg = hf_command_line_test(config,
                                                 usercase,
                                                 cli_case_config.get(usercase),
                                                 model,
                                                 'pytorch',
                                                 cuda_prefix=get_cuda_prefix_by_workerid(worker_id))

    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_turbomind_chat
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('model', get_torch_model_list(parallel_config=2, model_type='base_model'))
def test_hf_pytorch_base_tp2(config, model, cli_case_config, worker_id):
    usercase = 'base_testcase'
    result, chat_log, msg = hf_command_line_test(config,
                                                 usercase,
                                                 cli_case_config.get(usercase),
                                                 model,
                                                 'pytorch',
                                                 cuda_prefix=get_cuda_prefix_by_workerid(worker_id, parallel_config=2))

    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_pytorch_chat
@pytest.mark.gpu_num_2
@pytest.mark.pr_test
@pytest.mark.parametrize('model', ['internlm/internlm2_5-20b-chat', 'mistralai/Mixtral-8x7B-Instruct-v0.1'])
def test_hf_pytorch_chat_pr(config, model, cli_case_config):
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
                                                 'pytorch',
                                                 cuda_prefix=f'{env_var}5,6')
    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_pytorch_chat
@pytest.mark.gpu_num_1
@pytest.mark.other
@pytest.mark.parametrize('model', ['Qwen/Qwen2.5-7B-Instruct'])
def test_modelscope_pytorch_chat_tp1(config, model, cli_case_config, worker_id):
    os.environ['LMDEPLOY_USE_MODELSCOPE'] = 'True'
    usercase = 'chat_testcase'
    result, chat_log, msg = hf_command_line_test(config,
                                                 usercase,
                                                 cli_case_config.get(usercase),
                                                 model,
                                                 'pytorch',
                                                 cuda_prefix=get_cuda_prefix_by_workerid(worker_id),
                                                 use_local_model=False)
    del os.environ['LMDEPLOY_USE_MODELSCOPE']

    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_pytorch_chat
@pytest.mark.gpu_num_1
@pytest.mark.other
@pytest.mark.parametrize('model', ['meta-llama/Llama-2-7b-chat-hf'])
def test_pytorch_chat_with_lora_tp1(config, model, cli_case_config, worker_id):
    usercase = 'chat_testcase'
    result, chat_log, msg = hf_command_line_test(config,
                                                 usercase,
                                                 cli_case_config.get(usercase),
                                                 model,
                                                 'pytorch',
                                                 cuda_prefix=get_cuda_prefix_by_workerid(worker_id),
                                                 extra='--adapters lora/Llama2-Chinese-7b-Chat-LoRA')

    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_pytorch_chat
@pytest.mark.gpu_num_2
@pytest.mark.other
@pytest.mark.parametrize('model', ['baichuan-inc/Baichuan2-13B-Chat'])
def test_pytorch_chat_with_lora_tp2(config, model, cli_case_config, worker_id):
    usercase = 'chat_testcase'
    result, chat_log, msg = hf_command_line_test(config,
                                                 usercase,
                                                 cli_case_config.get(usercase),
                                                 model,
                                                 'pytorch',
                                                 cuda_prefix=get_cuda_prefix_by_workerid(worker_id, parallel_config=2),
                                                 extra='--adapters a=lora/2024-01-25_self_dup b=lora/2024-01-25_self')

    if chat_log is not None:
        allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg
