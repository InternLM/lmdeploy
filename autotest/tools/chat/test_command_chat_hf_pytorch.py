import allure
import conftest
import pytest
from utils.config_utils import get_torch_model_list
from utils.run_client_chat import hf_command_line_test

conftest._init_cli_case_list()
case_list = conftest.global_cli_case_List


def getCaseList():
    return case_list


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_pytorch_chat
@pytest.mark.parametrize('usercase', getCaseList())
@pytest.mark.parametrize('model', get_torch_model_list())
def test_hf_pytorch_chat(config, model, cli_case_config, usercase):
    result, chat_log, msg = hf_command_line_test(config, usercase,
                                                 cli_case_config.get(usercase),
                                                 model, 'torch')
    if chat_log is not None:
        allure.attach.file(chat_log,
                           attachment_type=allure.attachment_type.TEXT)

    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_pytorch_chat
@pytest.mark.pr_test
@pytest.mark.xdist_group(name='pr_test')
@pytest.mark.parametrize('usercase', getCaseList())
@pytest.mark.parametrize('model', ['internlm2-chat-20b'])
def test_hf_pytorch_chat_pr(config, model, cli_case_config, usercase):
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
