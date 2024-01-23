import allure
import conftest
import pytest
from utils.run_client_chat import pytorch_command_line_test

conftest._init_cli_case_list()
case_list = conftest.global_cli_case_List


def getCaseList():
    return case_list


@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.timeout(120)
@pytest.mark.command_chat_pytorch
class TestPytorchCommandChat:

    @pytest.mark.internlm2_chat_7b
    @allure.story('internlm2-chat-7b')
    @pytest.mark.parametrize('usercase', getCaseList())
    def test_pytorch_chat_internlm2_chat_7b(self, config, cli_case_config,
                                            usercase):
        run_command_line_test(config, usercase, cli_case_config.get(usercase),
                              'internlm2-chat-7b')

    @pytest.mark.internlm2_chat_20b
    @allure.story('internlm2-chat-20b')
    @pytest.mark.parametrize('usercase', getCaseList())
    def test_pytorch_chat_internlm2_chat_20b(self, config, cli_case_config,
                                             usercase):
        run_command_line_test(config, usercase, cli_case_config.get(usercase),
                              'internlm2-chat-20b')

    @pytest.mark.internlm2_chat_20b
    @allure.story('internlm2-chat-20b-inner-w4')
    @pytest.mark.parametrize('usercase', getCaseList())
    def test_pytorch_chat_internlm2_chat_20b_inner_w4(self, config,
                                                      cli_case_config,
                                                      usercase):
        run_command_line_test(config, usercase, cli_case_config.get(usercase),
                              'internlm2-chat-20b-inner-w4')

    @pytest.mark.internlm2_chat_20b
    @allure.story('internlm2-chat-20b-inner-w8a8')
    @pytest.mark.parametrize('usercase', getCaseList())
    def test_chat_internlm2_chat_20b_inner_w8a8(self, config, cli_case_config,
                                                usercase):
        run_command_line_test(config, usercase, cli_case_config.get(usercase),
                              'internlm2-chat-20b-inner-w8a8')


def run_command_line_test(config, case, case_info, model_case):
    model_map = config.get('model_map')

    if model_case not in model_map.keys():
        assert False, 'the model is incorrect'

    result, chat_log, msg = pytorch_command_line_test(config, case, case_info,
                                                      model_case)
    allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg
