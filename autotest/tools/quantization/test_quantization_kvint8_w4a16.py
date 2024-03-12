import os

import allure
import pytest
from utils.config_utils import get_cuda_prefix_by_workerid
from utils.quantization_utils import quantization

model_list = [
    'meta-llama/Llama-2-7b-chat-inner-kvint8',
    'internlm/internlm-chat-20b-inner-kvint8',
    'internlm/internlm2-chat-20b-inner-kvint8',
    'Qwen/Qwen-7B-Chat-inner-kvint8', 'Qwen/Qwen-14B-Chat-inner-kvint8',
    'internlm/internlm2-20b-inner-kvint8',
    'baichuan-inc/Baichuan2-7B-Chat-inner-kvint8'
]


@pytest.mark.order(4)
@pytest.mark.quantization_kvint8_w4a16
@pytest.mark.timeout(900)
@pytest.mark.parametrize('model', model_list)
def test_quantization_kvint8_w4a16(config, model, worker_id):
    quantization_kvint8(config, model + '-w4a16', model,
                        get_cuda_prefix_by_workerid(worker_id))


def quantization_kvint8(config, quantization_model_name, origin_model_name,
                        cuda_prefix):
    quantization_type = 'w4a16'
    result, msg = quantization(config, quantization_model_name,
                               origin_model_name, quantization_type,
                               cuda_prefix)
    log_path = config.get('log_path')
    quantization_log = os.path.join(
        log_path, '_'.join([
            'quantization', quantization_type,
            quantization_model_name.split('/')[1]
        ]) + '.log')

    allure.attach.file(quantization_log,
                       attachment_type=allure.attachment_type.TEXT)
    assert result, msg
