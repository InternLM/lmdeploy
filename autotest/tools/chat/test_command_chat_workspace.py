import allure
import conftest
import pytest
from utils.config_utils import get_turbomind_model_list
from utils.run_client_chat import command_line_test

conftest._init_cli_case_list()
prompt_list = conftest.global_cli_case_List


def getPromptCaseList():
    return prompt_list


def getModelList():
    return [
        item for item in get_turbomind_model_list()
        if 'kvint8' not in item.lower()
    ]


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.command_chat
@pytest.mark.parametrize('usercase', getPromptCaseList())
@pytest.mark.parametrize('model', getModelList())
def test_workspace_chat(config, cli_case_config, usercase, model):
    result, chat_log, msg = command_line_test(config, usercase,
                                              cli_case_config.get(usercase),
                                              model, 'turbomind', None)
    if chat_log is not None:
        allure.attach.file(chat_log,
                           attachment_type=allure.attachment_type.TEXT)
    assert result, msg


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.command_chat
@pytest.mark.pr_test
@pytest.mark.parametrize('usercase', getPromptCaseList())
@pytest.mark.parametrize(
    'model', ['internlm2-chat-20b', 'internlm2-chat-20b-inner-w4a16'])
def test_workspace_chat_pr(config, cli_case_config, usercase, model):
    result, chat_log, msg = command_line_test(
        config,
        usercase,
        cli_case_config.get(usercase),
        model,
        'turbomind',
        None,
        cuda_prefix='CUDA_VISIBLE_DEVICES=5,6')
    if chat_log is not None:
        allure.attach.file(chat_log,
                           attachment_type=allure.attachment_type.TEXT)
    assert result, msg
