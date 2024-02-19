import os
import subprocess
from subprocess import PIPE

import allure
import pytest
from utils.get_run_config import get_command_with_extra


@pytest.mark.quantization
@pytest.mark.timeout(600)
class TestQuantization:

    @pytest.mark.CodeLlama_7b_Instruct_hf
    @allure.story('CodeLlama-7b-Instruct-hf')
    def future_test_quantization_CodeLlama_7b_Instruct_hf(self, config):
        quantization(config, 'CodeLlama-7b-Instruct-hf-inner-w4',
                     'CodeLlama-7b-Instruct-hf')

    @pytest.mark.llama_2_7b_chat
    @allure.story('llama-2-7b-chat')
    def test_quantization_llama_2_7b_chat(self, config):
        quantization(config, 'llama-2-7b-chat-inner-w4', 'llama-2-7b-chat',
                     'CUDA_VISIBLE_DEVICES=0')

    @pytest.mark.timeout(900)
    @pytest.mark.internlm_chat_20b
    @allure.story('internlm-chat-20b')
    def test_quantization_internlm_chat_20b(self, config):
        quantization(config, 'internlm-chat-20b-inner-w4', 'internlm-chat-20b',
                     'CUDA_VISIBLE_DEVICES=1')

    @pytest.mark.timeout(900)
    @pytest.mark.internlm2_chat_20b
    @allure.story('internlm2-chat-20b')
    def test_quantization_internlm2_chat_20b(self, config):
        quantization(config, 'internlm2-chat-20b-inner-w4',
                     'internlm2-chat-20b', 'CUDA_VISIBLE_DEVICES=2')

    @pytest.mark.Qwen_14B_Chat
    @allure.story('Qwen-14B-Chat')
    def test_quantization_Qwen_14B_Chat(self, config):
        quantization(config, 'Qwen-14B-Chat-inner-w4', 'Qwen-14B-Chat',
                     'CUDA_VISIBLE_DEVICES=3')

    @pytest.mark.Baichuan2_7B_Chat
    @allure.story('Baichuan2-7B-Chat')
    def test_quantization_Baichuan2_7B_Chat(self, config):
        quantization(config, 'Baichuan2-7B-Chat-inner-w4', 'Baichuan2-7B-Chat',
                     'CUDA_VISIBLE_DEVICES=4')

    @pytest.mark.Baichuan2_13B_Chat
    @allure.story('Baichuan2-13B-Chat')
    def future_test_quantization_Baichuan2_13B_Chat(self, config):
        quantization(config, 'Baichuan2-13B-Chat-inner-w4',
                     'Baichuan2-13B-Chat', 'CUDA_VISIBLE_DEVICES=5')

    @pytest.mark.Qwen_7B_Chat
    @allure.story('Qwen-7B-Chat')
    def test_quantization_Qwen_7B_Chat(self, config):
        quantization(config, 'Qwen-7B-Chat-inner-w4', 'Qwen-7B-Chat',
                     'CUDA_VISIBLE_DEVICES=6')


def quantization(config, w4_model_name, origin_model_name, cuda_prefix):
    model_path = config.get('model_path')
    log_path = config.get('log_path')

    quantization_cmd = get_command_with_extra(
        'lmdeploy lite auto_awq ' + model_path + '/' + origin_model_name +
        ' --work-dir ' + model_path + '/' + w4_model_name +
        ' --calib-dataset ptb', config, origin_model_name, False, cuda_prefix)

    quantization_log = os.path.join(log_path,
                                    'quantization_' + w4_model_name + '.log')

    with open(quantization_log, 'w') as f:
        f.writelines('commondLine quantization_cmd: ' + quantization_cmd +
                     '\n')
        # quantization
        quantizationRes = subprocess.run([quantization_cmd],
                                         stdout=f,
                                         stderr=PIPE,
                                         shell=True,
                                         text=True,
                                         encoding='utf-8')
        f.writelines(quantizationRes.stderr)
        result = quantizationRes.returncode == 0

    allure.attach.file(quantization_log,
                       attachment_type=allure.attachment_type.TEXT)
    assert result, quantizationRes.stderr
