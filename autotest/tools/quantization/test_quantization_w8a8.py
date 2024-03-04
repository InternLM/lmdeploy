import os

import allure
import pytest
from utils.config_utils import get_cuda_prefix_by_workerid
from utils.quantization_utils import quantization

model_list = [
    'meta-llama/Llama-2-7b-chat', 'internlm/internlm-chat-20b',
    'internlm/internlm2-chat-20b', 'internlm/internlm2-chat-7b',
    '01-ai/Yi-6B-Chat', 'internlm/internlm2-20b'
]


@pytest.mark.order(2)
@pytest.mark.quantization_w8a8
@pytest.mark.timeout(900)
@pytest.mark.parametrize('model', model_list)
def test_quantization_w8a8(config, model, worker_id):
    quantization_w8a8(config, model + '-inner-w8a8', model,
                      get_cuda_prefix_by_workerid(worker_id))


def quantization_w8a8(config, quantization_model_name, origin_model_name,
                      cuda_prefix):
    quantization_type = 'w8a8'
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
