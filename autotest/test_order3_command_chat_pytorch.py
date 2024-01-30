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
    @allure.story('internlm2-chat-20b-inner-w8a8')
    @pytest.mark.parametrize('usercase', getCaseList())
    def test_chat_internlm2_chat_20b_inner_w8a8(self, config, cli_case_config,
                                                usercase):
        run_command_line_test(config, usercase, cli_case_config.get(usercase),
                              'internlm2-chat-20b-inner-w8a8')

    @pytest.mark.internlm_chat_7b
    @allure.story('internlm-chat-7b')
    @pytest.mark.parametrize('usercase', getCaseList())
    def test_pytorch_chat_internlm_chat_7b(self, config, cli_case_config,
                                           usercase):
        run_command_line_test(config, usercase, cli_case_config.get(usercase),
                              'internlm-chat-7b')

    @pytest.mark.internlm_chat_20b
    @allure.story('internlm-chat-20b')
    @pytest.mark.parametrize('usercase', getCaseList())
    def test_pytorch_chat_internlm_chat_20b(self, config, cli_case_config,
                                            usercase):
        run_command_line_test(config, usercase, cli_case_config.get(usercase),
                              'internlm-chat-20b')

    @pytest.mark.Baichuan2_7B_Chat
    @allure.story('Baichuan2-7B-Chat')
    @pytest.mark.parametrize('usercase', getCaseList())
    def test_pytorch_chat_Baichuan2_7B_Chat(self, config, cli_case_config,
                                            usercase):
        run_command_line_test(config, usercase, cli_case_config.get(usercase),
                              'Baichuan2-7B-Chat')

    @pytest.mark.Baichuan2_13B_Chat
    @allure.story('Baichuan2-13B-Chat')
    @pytest.mark.parametrize('usercase', getCaseList())
    def test_pytorch_chat_Baichuan2_13B_Chat(self, config, cli_case_config,
                                             usercase):
        run_command_line_test(config, usercase, cli_case_config.get(usercase),
                              'Baichuan2-13B-Chat')

    @pytest.mark.Yi_6B_Chat
    @allure.story('Yi-6B-Chat')
    @pytest.mark.parametrize('usercase', getCaseList())
    def test_pytorch_chat_Yi_6B_Chat(self, config, cli_case_config, usercase):
        run_command_line_test(config, usercase, cli_case_config.get(usercase),
                              'Yi-6B-Chat')

    @pytest.mark.chatglm2_6b
    @allure.story('chatglm2-6b')
    @pytest.mark.parametrize('usercase', getCaseList())
    def test_pytorch_chat_chatglm2_6b(self, config, cli_case_config, usercase):
        run_command_line_test(config, usercase, cli_case_config.get(usercase),
                              'chatglm2-6b')

    @pytest.mark.falcon_7b
    @allure.story('falcon-7b')
    @pytest.mark.parametrize('usercase', getCaseList())
    def test_pytorch_chat_falcon_7b(self, config, cli_case_config, usercase):
        run_command_line_test(config, usercase, cli_case_config.get(usercase),
                              'falcon-7b')


def run_command_line_test(config, case, case_info, model_case):
    model_map = config.get('model_map')

    if model_case not in model_map.keys():
        assert False, 'the model is incorrect'

    result, chat_log, msg = pytorch_command_line_test(config, case, case_info,
                                                      model_case)
    allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg
