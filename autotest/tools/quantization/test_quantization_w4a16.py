import os

import allure
import pytest
from utils.quantization_utils import quantization

model_list = [('llama-2-7b-chat', 'CUDA_VISIBLE_DEVICES=0'),
              ('internlm-chat-20b', 'CUDA_VISIBLE_DEVICES=1'),
              ('Qwen-7B-Chat', 'CUDA_VISIBLE_DEVICES=3'),
              ('Qwen-14B-Chat', 'CUDA_VISIBLE_DEVICES=4'),
              ('Qwen-VL', 'CUDA_VISIBLE_DEVICES=5'),
              ('internlm2-1_8b', 'CUDA_VISIBLE_DEVICES=6'),
              ('Baichuan2-7B-Chat', 'CUDA_VISIBLE_DEVICES=7')]


@pytest.mark.order(3)
@pytest.mark.quantization_w4a16
@pytest.mark.timeout(900)
@pytest.mark.parametrize('model, prefix', model_list)
def test_quantization_w4a16(config, model, prefix):
    quantization_w4a16(config, model + '-inner-w4a16', model, prefix)


@pytest.mark.order(3)
@pytest.mark.quantization_w4a16
@pytest.mark.pr_test
@pytest.mark.timeout(900)
@pytest.mark.parametrize('model, prefix',
                         [('internlm2-chat-20b', 'CUDA_VISIBLE_DEVICES=5')])
def test_quantization_w4a16_pr(config, model, prefix):
    quantization_w4a16(config, model + '-inner-w4a16', model, prefix)


def quantization_w4a16(config, quantization_model_name, origin_model_name,
                       cuda_prefix):
    quantization_type = 'w4a16'
    result, msg = quantization(config, quantization_model_name,
                               origin_model_name, quantization_type,
                               cuda_prefix)
    log_path = config.get('log_path')
    quantization_log = os.path.join(
        log_path,
        '_'.join(['quantization', quantization_type, quantization_model_name
                  ]) + '.log')

    allure.attach.file(quantization_log,
                       attachment_type=allure.attachment_type.TEXT)
    assert result, msg
