import os

import allure
import pytest
from utils.config_utils import (get_cuda_prefix_by_workerid,
                                get_quantization_model_list)
from utils.quantization_utils import quantization


@pytest.mark.order(3)
@pytest.mark.quantization_w4a16
@pytest.mark.timeout(900)
@pytest.mark.parametrize('model', get_quantization_model_list('w4a16'))
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
