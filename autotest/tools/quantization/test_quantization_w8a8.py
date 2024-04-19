import os

import allure
import pytest
from utils.config_utils import (get_cuda_prefix_by_workerid,
                                get_quantization_model_list)
from utils.quantization_utils import quantization


@pytest.mark.order(2)
@pytest.mark.quantization_w8a8
@pytest.mark.timeout(900)
@pytest.mark.parametrize('model', get_quantization_model_list('w8a8'))
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
