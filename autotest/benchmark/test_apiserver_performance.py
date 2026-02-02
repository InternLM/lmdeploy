import pytest
from utils.benchmark_utils import restful_test
from utils.config_utils import get_func_config_list


def get_models(backend, parallel_config):
    return get_func_config_list(backend, parallel_config, func_type='benchmark')


@pytest.mark.turbomind
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(backend='turbomind', parallel_config={'tp': 1}))
def test_turbomind_apiserver_tp1(config, run_config, worker_id):
    result, msg = restful_test(config, run_config, worker_id=worker_id)
    assert result, msg


@pytest.mark.turbomind
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(backend='turbomind', parallel_config={'tp': 2}))
def test_turbomind_apiserver_tp2(config, run_config, worker_id):
    result, msg = restful_test(config, run_config, worker_id=worker_id)
    assert result, msg


@pytest.mark.turbomind
@pytest.mark.gpu_num_4
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(backend='turbomind', parallel_config={'tp': 4}))
def test_turbomind_apiserver_tp4(config, run_config, worker_id):
    result, msg = restful_test(config, run_config, worker_id=worker_id)
    assert result, msg


@pytest.mark.turbomind
@pytest.mark.gpu_num_8
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(backend='turbomind', parallel_config={'tp': 8}))
def test_turbomind_apiserver_tp8(config, run_config, worker_id):
    result, msg = restful_test(config, run_config, worker_id=worker_id)
    assert result, msg


@pytest.mark.pytorch
@pytest.mark.gpu_num_1
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(backend='pytorch', parallel_config={'tp': 1}))
def test_pytorch_apiserver_tp1(config, run_config, worker_id):
    result, msg = restful_test(config, run_config, worker_id=worker_id)
    assert result, msg


@pytest.mark.pytorch
@pytest.mark.gpu_num_2
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(backend='pytorch', parallel_config={'tp': 2}))
def test_pytorch_apiserver_tp2(config, run_config, worker_id):
    result, msg = restful_test(config, run_config, worker_id=worker_id)
    assert result, msg


@pytest.mark.pytorch
@pytest.mark.gpu_num_4
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(backend='pytorch', parallel_config={'tp': 4}))
def test_pytorch_apiserver_tp4(config, run_config, worker_id):
    result, msg = restful_test(config, run_config, worker_id=worker_id)
    assert result, msg


@pytest.mark.pytorch
@pytest.mark.gpu_num_8
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(backend='pytorch', parallel_config={'tp': 8}))
def test_pytorch_apiserver_tp8(config, run_config, worker_id):
    result, msg = restful_test(config, run_config, worker_id=worker_id)
    assert result, msg


@pytest.mark.pytorch
@pytest.mark.gpu_num_16
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(backend='pytorch', parallel_config={'tp': 16}))
def test_pytorch_apiserver_tp16(config, run_config, worker_id):
    result, msg = restful_test(config, run_config, worker_id=worker_id)
    assert result, msg


@pytest.mark.function
@pytest.mark.flaky(reruns=0)
@pytest.mark.gpu_num_2
@pytest.mark.parametrize('run_config', [{
    'model': 'Qwen/Qwen3-30B-A3B',
    'backend': 'pytorch',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {}
}, {
    'model': 'Qwen/Qwen3-30B-A3B',
    'backend': 'turbomind',
    'communicator': 'nccl',
    'quant_policy': 4,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {}
}, {
    'model': 'Qwen/Qwen3-30B-A3B',
    'backend': 'turbomind',
    'communicator': 'cuda-ipc',
    'quant_policy': 8,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {}
}, {
    'model': 'Qwen/Qwen3-VL-32B-Instruct',
    'backend': 'pytorch',
    'communicator': 'nccl',
    'quant_policy': 8,
    'parallel_config': {
        'tp': 2
    },
    'extra_params': {}
}])
def test_restful_func_tp2(config, run_config, worker_id):
    result, msg = restful_test(config, run_config, worker_id=worker_id, is_smoke=True)

    assert result, msg
