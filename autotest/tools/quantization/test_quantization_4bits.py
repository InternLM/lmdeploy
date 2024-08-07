import os

import allure
import pytest
from utils.config_utils import (get_cuda_prefix_by_workerid,
                                get_quantization_model_list)
from utils.quantization_utils import quantization


@pytest.mark.order(3)
@pytest.mark.quantization_4bits
@pytest.mark.timeout(900)
@pytest.mark.parametrize('model', get_quantization_model_list('4bits'))
def test_quantization_4bits(config, model, worker_id):
    quantization_4bits(config, model + '-inner-4bits', model,
                       get_cuda_prefix_by_workerid(worker_id))


@pytest.mark.order(3)
@pytest.mark.quantization_4bits
@pytest.mark.pr_test
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    'model, prefix',
    [('internlm/internlm2-chat-20b', 'CUDA_VISIBLE_DEVICES=5')])
def test_quantization_4bits_pr(config, model, prefix):
    quantization_4bits(config, model + '-inner-4bits', model, prefix)


def quantization_4bits(config, quantization_model_name, origin_model_name,
                       cuda_prefix):
    quantization_type = '4bits'
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
