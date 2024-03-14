import os

import allure
import pytest
from utils.config_utils import get_cuda_prefix_by_workerid
from utils.quantization_utils import quantization

model_list = [
    'meta-llama/Llama-2-7b-chat-hf', 'internlm/internlm-chat-20b',
    'Qwen/Qwen-7B-Chat', 'Qwen/Qwen-14B-Chat', 'Qwen/Qwen-VL',
    'internlm/internlm2-chat-20b', 'internlm/internlm2-20b',
    'baichuan-inc/Baichuan2-7B-Chat'
]


@pytest.mark.order(3)
@pytest.mark.quantization_w4a16
@pytest.mark.timeout(900)
@pytest.mark.parametrize('model', model_list)
def test_quantization_w4a16(config, model, worker_id):
    quantization_w4a16(config, model + '-inner-w4a16', model,
                       get_cuda_prefix_by_workerid(worker_id))


@pytest.mark.order(3)
@pytest.mark.quantization_w4a16
@pytest.mark.pr_test
@pytest.mark.flaky(reruns=0)
@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    'model, prefix',
    [('internlm/internlm2-chat-20b', 'CUDA_VISIBLE_DEVICES=5')])
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
        log_path, '_'.join([
            'quantization', quantization_type,
            quantization_model_name.split('/')[1]
        ]) + '.log')

    allure.attach.file(quantization_log,
                       attachment_type=allure.attachment_type.TEXT)
    assert result, msg
