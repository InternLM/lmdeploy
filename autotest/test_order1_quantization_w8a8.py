import os
import subprocess
from subprocess import PIPE

import allure
import pytest
from utils.get_run_config import get_command_with_extra


@pytest.mark.quantization_w8a8
@pytest.mark.timeout(600)
class TestW8a8Quantization:

    @pytest.mark.timeout(900)
    @pytest.mark.internlm2_chat_20b
    @allure.story('internlm2-chat-20b')
    def test_quantization_internlm2_chat_20b(self, config):
        w8a8_quantization(config, 'internlm2-chat-20b-inner-w8a8',
                          'internlm2-chat-20b')


def w8a8_quantization(config, w8a8_model_name, origin_model_name):
    model_path = config.get('model_path')
    log_path = config.get('log_path')

    quantization_cmd = get_command_with_extra(
        'lmdeploy lite smooth_quant ' + model_path + '/' + origin_model_name +
        ' --work-dir ' + model_path + '/' + w8a8_model_name, config,
        origin_model_name, False)

    quantization_log = os.path.join(log_path,
                                    'quantization_' + w8a8_model_name + '.log')

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
