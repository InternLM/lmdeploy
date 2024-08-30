import os

import allure
import pytest
from utils.config_utils import (get_cuda_prefix_by_workerid,
                                get_turbomind_model_list)
from utils.run_client_chat import hf_command_line_test


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_turbomind_chat
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('model', get_turbomind_model_list(tp_num=1))
def test_hf_turbomind_chat_tp1(config, model, cli_case_config, worker_id):
    usercase = 'chat_testcase'
    if 'coder' in model:
        usercase = 'code_testcase'
    result, chat_log, msg = hf_command_line_test(
        config,
        usercase,
        cli_case_config.get(usercase),
        model,
        'turbomind',
        cuda_prefix=get_cuda_prefix_by_workerid(worker_id))

    if chat_log is not None:
        allure.attach.file(chat_log,
                           attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_turbomind_chat
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('model', get_turbomind_model_list(tp_num=2))
def test_hf_turbomind_chat_tp2(config, model, cli_case_config, worker_id):
    usercase = 'chat_testcase'
    result, chat_log, msg = hf_command_line_test(
        config,
        usercase,
        cli_case_config.get(usercase),
        model,
        'turbomind',
        cuda_prefix=get_cuda_prefix_by_workerid(worker_id, tp_num=2))

    if chat_log is not None:
        allure.attach.file(chat_log,
                           attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_turbomind_chat
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('model',
                         get_turbomind_model_list(tp_num=1,
                                                  model_type='base_model'))
def test_hf_turbomind_base_tp1(config, model, cli_case_config, worker_id):
    usercase = 'base_testcase'
    result, chat_log, msg = hf_command_line_test(
        config,
        usercase,
        cli_case_config.get(usercase),
        model,
        'turbomind',
        cuda_prefix=get_cuda_prefix_by_workerid(worker_id))

    if chat_log is not None:
        allure.attach.file(chat_log,
                           attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_turbomind_chat
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('model',
                         get_turbomind_model_list(tp_num=2,
                                                  model_type='base_model'))
def test_hf_turbomind_base_tp2(config, model, cli_case_config, worker_id):
    usercase = 'base_testcase'
    result, chat_log, msg = hf_command_line_test(
        config,
        usercase,
        cli_case_config.get(usercase),
        model,
        'turbomind',
        cuda_prefix=get_cuda_prefix_by_workerid(worker_id, tp_num=2))

    if chat_log is not None:
        allure.attach.file(chat_log,
                           attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_turbomind_chat
@pytest.mark.gpu_num_2
@pytest.mark.pr_test
@pytest.mark.parametrize('model', [
    'internlm/internlm2_5-20b-chat',
    'internlm/internlm2_5-20b-chat-inner-4bits'
])
def test_hf_turbomind_chat_pr(config, model, cli_case_config):
    usercase = 'chat_testcase'

    result, chat_log, msg = hf_command_line_test(
        config,
        usercase,
        cli_case_config.get(usercase),
        model,
        'turbomind',
        cuda_prefix='CUDA_VISIBLE_DEVICES=5,6')

    if chat_log is not None:
        allure.attach.file(chat_log,
                           attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_turbomind_chat
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('model', ['Qwen/Qwen-7B-Chat'])
def test_modelscope_turbomind_chat_tp1(config, model, cli_case_config,
                                       worker_id):
    os.environ['LMDEPLOY_USE_MODELSCOPE'] = 'True'
    usercase = 'chat_testcase'
    result, chat_log, msg = hf_command_line_test(
        config,
        usercase,
        cli_case_config.get(usercase),
        model,
        'turbomind',
        cuda_prefix=get_cuda_prefix_by_workerid(worker_id),
        use_local_model=False)
    del os.environ['LMDEPLOY_USE_MODELSCOPE']

    if chat_log is not None:
        allure.attach.file(chat_log,
                           attachment_type=allure.attachment_type.TEXT)
    assert result, msg
