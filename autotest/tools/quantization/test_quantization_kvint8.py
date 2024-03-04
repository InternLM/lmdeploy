import os

import allure
import pytest
from utils.config_utils import get_cuda_prefix_by_workerid
from utils.quantization_utils import quantization

model_list = [
    'llama-2-7b-chat', 'internlm-chat-20b', 'internlm2-chat-20b',
    'Qwen-7B-Chat', 'Qwen-14B-Chat', 'internlm2-20b', 'Baichuan2-7B-Chat'
]


@pytest.mark.order(1)
@pytest.mark.quantization_kvint8
@pytest.mark.timeout(900)
@pytest.mark.parametrize('model', model_list)
def test_quantization_kvint8(config, model, worker_id):
    quantization_kvint8(config, model + '-inner-kvint8', model,
                        get_cuda_prefix_by_workerid(worker_id))


def quantization_kvint8(config, quantization_model_name, origin_model_name,
                        cuda_prefix):
    quantization_type = 'kvint8'
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
