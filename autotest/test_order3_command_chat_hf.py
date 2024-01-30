import allure
import conftest
import pytest
from utils.run_client_chat import hf_command_line_test

conftest._init_cli_case_list()
case_list = conftest.global_cli_case_List


def getCaseList():
    return case_list


@pytest.mark.usefixtures('cli_case_config')
@pytest.mark.timeout(90)
@pytest.mark.command_chat_hf
class TestCommandChat:

    @pytest.mark.llama2_chat_7b_w4
    @allure.story('llama2-chat-7b-w4')
    @pytest.mark.parametrize('usercase', getCaseList())
    def test_chat_llama2_chat_7b_w4(self, config, cli_case_config, usercase):
        run_command_line_test(config, usercase, cli_case_config.get(usercase),
                              'llama2-chat-7b-w4')

    @pytest.mark.internlm2_chat_7b
    @allure.story('internlm2-chat-7b')
    @pytest.mark.parametrize('usercase', getCaseList())
    def test_chat_internlm2_chat_7b(self, config, cli_case_config, usercase):
        run_command_line_test(config, usercase, cli_case_config.get(usercase),
                              'internlm2-chat-7b')

    @pytest.mark.internlm2_chat_20b
    @allure.story('internlm2-chat-20b')
    @pytest.mark.parametrize('usercase', getCaseList())
    def test_chat_internlm2_chat_20b(self, config, cli_case_config, usercase):
        run_command_line_test(config, usercase, cli_case_config.get(usercase),
                              'internlm2-chat-20b')

    @pytest.mark.internlm2_chat_20b
    @allure.story('internlm2-chat-20b-inner-w4')
    @pytest.mark.parametrize('usercase', getCaseList())
    def test_chat_internlm2_chat_20b_inner_w4(self, config, cli_case_config,
                                              usercase):
        run_command_line_test(config, usercase, cli_case_config.get(usercase),
                              'internlm2-chat-20b-inner-w4')

    @pytest.mark.internlm_chat_7b
    @allure.story('internlm-chat-7b')
    @pytest.mark.parametrize('usercase', getCaseList())
    def test_chat_internlm_chat_7b(self, config, cli_case_config, usercase):
        run_command_line_test(config, usercase, cli_case_config.get(usercase),
                              'internlm-chat-7b')

    @pytest.mark.internlm_chat_20b
    @allure.story('internlm-chat-20b')
    @pytest.mark.parametrize('usercase', getCaseList())
    def test_chat_internlm_chat_20b(self, config, cli_case_config, usercase):
        run_command_line_test(config, usercase, cli_case_config.get(usercase),
                              'internlm-chat-20b')

    @pytest.mark.internlm_chat_20b
    @allure.story('internlm-chat-20b-inner-w4')
    @pytest.mark.parametrize('usercase', getCaseList())
    def test_chat_internlm_chat_20b_inner_w4(self, config, cli_case_config,
                                             usercase):
        run_command_line_test(config, usercase, cli_case_config.get(usercase),
                              'internlm-chat-20b-inner-w4')

    @pytest.mark.Qwen_7B_Chat
    @allure.story('Qwen-7B-Chat')
    @pytest.mark.parametrize('usercase', getCaseList())
    def test_chat_Qwen_7B_Chat(self, config, cli_case_config, usercase):
        run_command_line_test(config, usercase, cli_case_config.get(usercase),
                              'Qwen-7B-Chat')

    @pytest.mark.Qwen_7B_Chat
    @allure.story('Qwen-7B-Chat-inner-w4')
    @pytest.mark.parametrize('usercase', getCaseList())
    def test_chat_Qwen_7B_Chat_inner_w4(self, config, cli_case_config,
                                        usercase):
        run_command_line_test(config, usercase, cli_case_config.get(usercase),
                              'Qwen-7B-Chat-inner-w4')

    @pytest.mark.Qwen_14B_Chat
    @allure.story('Qwen-14B-Chat')
    @pytest.mark.parametrize('usercase', getCaseList())
    def test_chat_Qwen_14B_Chat(self, config, cli_case_config, usercase):
        run_command_line_test(config, usercase, cli_case_config.get(usercase),
                              'Qwen-14B-Chat')

    @pytest.mark.Qwen_14B_Chat
    @allure.story('Qwen-14B-Chat-inner-w4')
    @pytest.mark.parametrize('usercase', getCaseList())
    def test_chat_Qwen_14B_Chat_inner_w4(self, config, cli_case_config,
                                         usercase):
        run_command_line_test(config, usercase, cli_case_config.get(usercase),
                              'Qwen-14B-Chat-inner-w4')

    @pytest.mark.Baichuan2_7B_Chat
    @allure.story('Baichuan2-7B-Chat')
    @pytest.mark.parametrize('usercase', getCaseList())
    def test_chat_Baichuan2_7B_Chat(self, config, cli_case_config, usercase):
        run_command_line_test(config, usercase, cli_case_config.get(usercase),
                              'Baichuan2-7B-Chat')

    @pytest.mark.Baichuan2_7B_Chat
    @allure.story('Baichuan2-7B-Chat-inner-w4')
    @pytest.mark.parametrize('usercase', getCaseList())
    def test_chat_Baichuan2_7B_Chat_inner_w4(self, config, cli_case_config,
                                             usercase):
        run_command_line_test(config, usercase, cli_case_config.get(usercase),
                              'Baichuan2-7B-Chat-inner-w4')

    @pytest.mark.Baichuan2_13B_Chat
    @allure.story('Baichuan2-13B-Chat')
    @pytest.mark.parametrize('usercase', getCaseList())
    def future_test_chat_Baichuan2_13B_Chat(self, config, cli_case_config,
                                            usercase):
        run_command_line_test(config, usercase, cli_case_config.get(usercase),
                              'Baichuan2-13B-Chat')

    @pytest.mark.Baichuan2_13B_Chat
    @allure.story('Baichuan2-13B-Chat-inner-w4')
    @pytest.mark.parametrize('usercase', getCaseList())
    def future_test_chat_Baichuan2_13B_Chat_inner_w4(self, config,
                                                     cli_case_config,
                                                     usercase):
        run_command_line_test(config, usercase, cli_case_config.get(usercase),
                              'Baichuan2-13B-Chat-inner-w4')

    @pytest.mark.CodeLlama_7b_Instruct_hf
    @allure.story('CodeLlama-7b-Instruct-hf')
    @pytest.mark.parametrize('usercase', getCaseList())
    def future_test_chat_CodeLlama_7b_Instruct_hf(self, config,
                                                  cli_case_config, usercase):
        run_command_line_test(config, usercase, cli_case_config.get(usercase),
                              'CodeLlama-7b-Instruct-hf')

    @pytest.mark.llama_2_7b_chat
    @allure.story('llama-2-7b-chat')
    @pytest.mark.parametrize('usercase', getCaseList())
    def test_chat_llama_2_7b_chat(self, config, cli_case_config, usercase):
        run_command_line_test(config, usercase, cli_case_config.get(usercase),
                              'llama-2-7b-chat')

    @pytest.mark.llama_2_7b_chat
    @allure.story('llama-2-7b-chat-inner-w4')
    @pytest.mark.parametrize('usercase', getCaseList())
    def test_chat_llama_2_7b_chat_inner_w4(self, config, cli_case_config,
                                           usercase):
        run_command_line_test(config, usercase, cli_case_config.get(usercase),
                              'llama-2-7b-chat-inner-w4')


def run_command_line_test(config, case, case_info, model_case):
    model_map = config.get('model_map')

    if model_case not in model_map.keys():
        assert False, 'the model is incorrect'
    model_name = model_map.get(model_case)

    result, chat_log, msg = hf_command_line_test(config, case, case_info,
                                                 model_case, model_name)
    allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)

    assert result, msg
