import os
import subprocess
from subprocess import PIPE

import allure
import pytest
from utils.get_run_config import get_command_with_extra


@pytest.mark.convert
@pytest.mark.timeout(120)
class TestConvert:

    @pytest.mark.llama2_chat_7b_w4
    @allure.story('llama2-chat-7b-w4')
    def test_model_convert_llama2_chat_7b_w4(self, config):
        convert(config, 'llama2-chat-7b-w4')

    @pytest.mark.internlm_chat_7b
    @allure.story('internlm-chat-7b')
    def test_model_convert_internlm_chat_7b(self, config):
        convert(config, 'internlm-chat-7b')

    @pytest.mark.internlm2_chat_7b
    @allure.story('internlm2-chat-7b')
    def test_model_convert_internlm2_chat_7b(self, config):
        convert(config, 'internlm2-chat-7b')

    @pytest.mark.internlm_chat_20b
    @allure.story('internlm-chat-20b')
    def test_model_convert_internlm_chat_20b(self, config):
        convert(config, 'internlm-chat-20b')

    @pytest.mark.internlm_chat_20b
    @allure.story('internlm-chat-20b')
    def test_model_convert_internlm_chat_20b_inner_w4(self, config):
        convert(config, 'internlm-chat-20b-inner-w4')

    @pytest.mark.internlm2_chat_20b
    @allure.story('internlm2-chat-20b')
    def test_model_convert_internlm2_chat_20b(self, config):
        convert(config, 'internlm2-chat-20b')

    @pytest.mark.internlm2_chat_20b
    @allure.story('internlm2-chat-20b')
    def future_test_model_convert_internlm2_chat_20b_inner_w4(self, config):
        convert(config, 'internlm2-chat-20b-inner-w4')

    @pytest.mark.Qwen_7B_Chat
    @allure.story('Qwen-7B-Chat')
    def test_model_convert_Qwen_7B_Chat(self, config):
        convert(config, 'Qwen-7B-Chat')

    @pytest.mark.Qwen_14B_Chat
    @allure.story('Qwen-14B-Chat')
    def test_model_convert_Qwen_14B_Chat(self, config):
        convert(config, 'Qwen-14B-Chat')

    @pytest.mark.Qwen_7B_Chat
    @allure.story('Qwen-7B-Chat-inner-w4')
    def test_model_convert_Qwen_7B_Chat_inner_w4(self, config):
        convert(config, 'Qwen-7B-Chat-inner-w4')

    @pytest.mark.Qwen_14B_Chat
    @allure.story('Qwen-14B-Chat-inner-w4')
    def test_model_convert_Qwen_14B_Chat_inner_w4(self, config):
        convert(config, 'Qwen-14B-Chat-inner-w4')

    @pytest.mark.Baichuan2_7B_Chat
    @allure.story('Baichuan2-7B-Chat')
    def test_model_convert_Baichuan2_7B_Chat(self, config):
        convert(config, 'Baichuan2-7B-Chat')

    @pytest.mark.Baichuan2_7B_Chat
    @allure.story('Baichuan2-7B-Chat-inner-w4')
    def test_model_convert_Baichuan2_7B_Chat_inner_w4(self, config):
        convert(config, 'Baichuan2-7B-Chat-inner-w4')

    @pytest.mark.CodeLlama_7b_Instruct_hf
    @allure.story('CodeLlama-7b-Instruct-hf')
    def future_test_model_convert_CodeLlama_7b_Instruct_hf(self, config):
        convert(config, 'CodeLlama-7b-Instruct-hf')

    @pytest.mark.CodeLlama_7b_Instruct_hf
    @allure.story('CodeLlama-7b-Instruct-hf-inner-w4')
    def future_test_model_convert_CodeLlama_7b_Instruct_hf_inner_w4(
            self, config):
        convert(config, 'CodeLlama-7b-Instruct-hf-inner-w4')

    @pytest.mark.llama_2_7b_chat
    @allure.story('llama-2-7b-chat')
    def test_model_convert_llama_2_7b_chat(self, config):
        convert(config, 'llama-2-7b-chat')

    @pytest.mark.llama_2_7b_chat
    @allure.story('llama-2-7b-chat-inner-w4')
    def test_model_convert_llama_2_7b_chat_inner_w4(self, config):
        convert(config, 'llama-2-7b-chat-inner-w4')


def convert(config, model_case):
    model_path = config.get('model_path')
    dst_path = config.get('dst_path')
    log_path = config.get('log_path')
    model_map = config.get('model_map')

    if model_case not in model_map.keys():
        assert False, 'the model is incorrect'
    model_name = model_map.get(model_case)

    if 'w4' in model_case:
        cmd = get_command_with_extra(
            'lmdeploy convert ' + model_name + ' ' + model_path + '/' +
            model_case + ' --model-format awq --group-size 128 --dst-path ' +
            dst_path + '/workspace_' + model_case, config, model_name, True)
    else:
        cmd = get_command_with_extra(
            'lmdeploy convert ' + model_name + ' ' + model_path + '/' +
            model_case + ' --dst-path ' + dst_path + '/workspace_' +
            model_case, config, model_name, True)

    convert_log = os.path.join(log_path, 'convert_' + model_case + '.log')
    with open(convert_log, 'w') as f:
        f.writelines('commondLine: ' + cmd + '\n')
        # convert
        convertRes = subprocess.run([cmd],
                                    stdout=f,
                                    stderr=PIPE,
                                    shell=True,
                                    text=True,
                                    encoding='utf-8')
        f.writelines(convertRes.stderr)
        # check result
        result = convertRes.returncode == 0

    allure.attach.file(convert_log,
                       attachment_type=allure.attachment_type.TEXT)

    assert result, convertRes.stderr
