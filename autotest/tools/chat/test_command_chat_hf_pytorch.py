import allure
import pytest
from utils.config_utils import (get_cuda_prefix_by_workerid,
                                get_torch_model_list)
from utils.run_client_chat import hf_command_line_test


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_pytorch_chat
@pytest.mark.gpu_num_1
@pytest.mark.parametrize('model', get_torch_model_list(tp_num=1))
def test_hf_pytorch_chat_tp1(config, model, cli_case_config, worker_id):
    usercase = 'chat_testcase'
    if 'deepseek-coder' in model:
        usercase = 'code_testcase'
    result, chat_log, msg = hf_command_line_test(
        config,
        usercase,
        cli_case_config.get(usercase),
        model,
        'torch',
        cuda_prefix=get_cuda_prefix_by_workerid(worker_id))
    if chat_log is not None:
        allure.attach.file(chat_log,
                           attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_pytorch_chat
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('model', get_torch_model_list(tp_num=2))
def test_hf_pytorch_chat_tp2(config, model, cli_case_config, worker_id):
    usercase = 'chat_testcase'
    result, chat_log, msg = hf_command_line_test(
        config,
        usercase,
        cli_case_config.get(usercase),
        model,
        'torch',
        cuda_prefix=get_cuda_prefix_by_workerid(worker_id, tp_num=2))
    if chat_log is not None:
        allure.attach.file(chat_log,
                           attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_pytorch_chat
@pytest.mark.pr_test
@pytest.mark.xdist_group(name='pr_test')
@pytest.mark.parametrize('model', ['internlm/internlm2-chat-20b'])
def test_hf_pytorch_chat_pr(config, model, cli_case_config, usercase):
    usercase = 'chat_testcase'
    result, chat_log, msg = hf_command_line_test(
        config,
        usercase,
        cli_case_config.get(usercase),
        model,
        'torch',
        cuda_prefix='CUDA_VISIBLE_DEVICES=5,6')
    if chat_log is not None:
        allure.attach.file(chat_log,
                           attachment_type=allure.attachment_type.TEXT)

    assert result, msg
