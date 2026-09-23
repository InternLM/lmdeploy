import os

import allure
import pytest
from utils.config_utils import (
    SUFFIX_INNER_AWQ,
    SUFFIX_INNER_GPTQ,
    SUFFIX_INNER_W8A8,
    get_cuda_prefix_by_workerid,
    get_quantization_model_list,
)
from utils.pytest_layout_utils import layout_mark
from utils.quantization_utils import quantization

_QUANT_SUFFIX = {
    'awq': SUFFIX_INNER_AWQ,
    'gptq': SUFFIX_INNER_GPTQ,
    'w8a8': SUFFIX_INNER_W8A8,
}


def _quantization_cases():
    """Build (model, type, output_name) cases; skip empty types to avoid
    phantom collects."""
    cases = []
    for quantization_type, suffix in _QUANT_SUFFIX.items():
        for model in get_quantization_model_list(quantization_type):
            cases.append(
                pytest.param(
                    model,
                    quantization_type,
                    model + suffix,
                    id=f'{quantization_type}:{model}',
                ))
    return cases


@pytest.mark.order(3)
@pytest.mark.test_3090
@pytest.mark.timeout(900)
@pytest.mark.parametrize('model,quantization_type,quantization_model_name', _quantization_cases())
def test_quantization(config, model, quantization_type, quantization_model_name, worker_id):
    quantization_all(config, quantization_model_name, model, quantization_type,
                     get_cuda_prefix_by_workerid(worker_id, {'tp': 1}))


@pytest.mark.order(3)
@pytest.mark.pr_test
@layout_mark({'tp': 2})
@pytest.mark.flaky(reruns=0)
@pytest.mark.timeout(900)
@pytest.mark.parametrize('model', ['Qwen/Qwen3-0.6B'])
def test_quantization_awq_pr(config, model):
    quantization_type = 'awq'
    quantization_all(config, model + SUFFIX_INNER_AWQ, model, quantization_type, cuda_prefix='CUDA_VISIBLE_DEVICES=6')


def quantization_all(config, quantization_model_name, origin_model_name, quantization_type, cuda_prefix: str = ''):
    result, msg = quantization(config, quantization_model_name, origin_model_name, quantization_type, cuda_prefix)
    log_path = config.get('log_path')
    quantization_log = os.path.join(
        log_path, '_'.join(['quantization', quantization_type,
                            quantization_model_name.split('/')[1]]) + '.log')

    allure.attach.file(quantization_log, name=quantization_log, attachment_type=allure.attachment_type.TEXT)
    assert result, msg
