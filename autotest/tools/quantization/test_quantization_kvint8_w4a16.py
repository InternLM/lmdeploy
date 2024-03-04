import os

import allure
import pytest
from utils.quantization_utils import quantization

model_list = [('llama-2-7b-chat-inner-kvint8', 'CUDA_VISIBLE_DEVICES=1'),
              ('internlm-chat-20b-inner-kvint8', 'CUDA_VISIBLE_DEVICES=2'),
              ('internlm2-chat-20b-inner-kvint8', 'CUDA_VISIBLE_DEVICES=3'),
              ('Qwen-7B-Chat-inner-kvint8', 'CUDA_VISIBLE_DEVICES=4'),
              ('Qwen-14B-Chat-inner-kvint8', 'CUDA_VISIBLE_DEVICES=5'),
              ('internlm2-20b-inner-kvint8', 'CUDA_VISIBLE_DEVICES=6'),
              ('Baichuan2-7B-Chat-inner-kvint8', 'CUDA_VISIBLE_DEVICES=7')]


@pytest.mark.order(4)
@pytest.mark.quantization_kvint8_w4a16
@pytest.mark.timeout(900)
@pytest.mark.parametrize('model, prefix', model_list)
def test_quantization_kvint8_w4a16(config, model, prefix):
    quantization_kvint8(config, model + '-w4a16', model, prefix)


def quantization_kvint8(config, quantization_model_name, origin_model_name,
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
