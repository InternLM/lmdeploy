import allure
import conftest
import pytest
from utils.config_utils import (get_cuda_prefix_by_workerid,
                                get_turbomind_model_list)
from utils.run_client_chat import hf_command_line_test

conftest._init_cli_case_list()
case_list = conftest.global_cli_case_List


def getCaseList():
    return case_list


@pytest.mark.order(10)
@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.hf_turbomind_chat
@pytest.mark.parametrize('usercase', getCaseList())
@pytest.mark.parametrize('model', get_turbomind_model_list(tp_num=1))
def test_hf_turbomind_chat_tp1(config, model, cli_case_config, usercase,
                               worker_id):
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
@pytest.mark.parametrize('usercase', getCaseList())
@pytest.mark.parametrize('model', get_turbomind_model_list(tp_num=2))
def test_hf_turbomind_chat_tp2(config, model, cli_case_config, usercase,
                               worker_id):
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
@pytest.mark.pr_test
@pytest.mark.xdist_group(name='pr_test')
@pytest.mark.parametrize('usercase', getCaseList())
@pytest.mark.parametrize(
    'model', ['internlm2-chat-20b', 'internlm2-chat-20b-inner-w4a16'])
def test_hf_turbomind_chat_pr(config, model, cli_case_config, usercase):
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
