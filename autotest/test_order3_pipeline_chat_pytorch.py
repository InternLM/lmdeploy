import os

import allure
import pytest
from pytest import assume
from utils.pipeline_chat import PipelinePytorchChat
from utils.rule_condition_assert import assert_result


@pytest.mark.timeout(600)
@pytest.mark.pipeline_chat_pytorch
class TestPipelinePytorchChat:

    @pytest.mark.internlm_chat_7b
    @allure.story('internlm-chat-7b')
    def test_chat_internlm_chat_7b(self, config, common_case_config):
        run_pipeline_pytorch_chat_test(config, common_case_config,
                                       'internlm-chat-7b')

    @pytest.mark.internlm2_chat_7b
    @allure.story('internlm2-chat-7b')
    def test_chat_internlm2_chat_7b(self, config, common_case_config):
        run_pipeline_pytorch_chat_test(config, common_case_config,
                                       'internlm2-chat-7b')

    @pytest.mark.internlm_chat_20b
    @allure.story('internlm-chat-20b')
    def future_test_chat_internlm_chat_20b(self, config, common_case_config):
        run_pipeline_pytorch_chat_test(config, common_case_config,
                                       'internlm-chat-20b')

    @pytest.mark.internlm2_chat_20b
    @allure.story('internlm2-chat-20b')
    def future_test_chat_internlm2_chat_20b(self, config, common_case_config):
        run_pipeline_pytorch_chat_test(config, common_case_config,
                                       'internlm2-chat-20b')

    @pytest.mark.Baichuan2_7B_Chat
    @allure.story('Baichuan2-7B-Chat')
    def future_test_chat_Baichuan2_7B_Chat(self, config, common_case_config):
        run_pipeline_pytorch_chat_test(config, common_case_config,
                                       'Baichuan2-7B-Chat')

    @pytest.mark.Qwen_7B_Chat
    @allure.story('Qwen-7B-Chat')
    def future_test_chat_Qwen_7B_Chat(self, config, common_case_config):
        run_pipeline_pytorch_chat_test(config, common_case_config,
                                       'Qwen-7B-Chat')

    @pytest.mark.llama_2_7b_chat
    @allure.story('llama-2-7b-chat')
    def future_test_chat_llama_2_7b_chat(self, config, common_case_config):
        run_pipeline_pytorch_chat_test(config, common_case_config,
                                       'llama-2-7b-chat')


def run_pipeline_pytorch_chat_test(config, cases_info, model_case):
    model_map = config.get('model_map')
    log_path = config.get('log_path')

    tp_config = config.get('tp_config')
    tp_info = 1
    if model_case in tp_config.keys():
        tp_info = tp_config.get(model_case)

    if model_case not in model_map.keys():
        assert False, 'the model is incorrect'
    model_name = model_map.get(model_case)
    model_path = config.get('model_path')
    hf_path = model_path + '/' + model_case

    # init pipeline
    pipe = PipelinePytorchChat(hf_path, tp_info)

    # run testcases
    for case in cases_info.keys():
        case_info = cases_info.get(case)
        msg = ''
        result = True
        with allure.step('case - ' + case):
            pipeline_chat_log = os.path.join(
                log_path, 'pipeline_chat_' + model_case + '_' + case + '.log')

            file = open(pipeline_chat_log, 'w')

            for prompt_detail in case_info:
                if result is False:
                    break
                prompt = list(prompt_detail.keys())[0]
                prompts = [{'role': 'user', 'content': prompt}]
                file.writelines('prompt:' + prompt + '\n')

                response = pipe.default_pipeline_chat(prompts).text

                case_result, reason = assert_result(response,
                                                    prompt_detail.values(),
                                                    model_name)
                file.writelines('output:' + response + '\n')
                file.writelines('result:' + str(case_result) + ',reason:' +
                                reason + '\n')
                if result is False:
                    msg += reason
                result = result & case_result
            file.close()
            allure.attach.file(pipeline_chat_log,
                               attachment_type=allure.attachment_type.TEXT)
            with assume:
                result, msg
