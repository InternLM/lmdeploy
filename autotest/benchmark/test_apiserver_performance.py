import pytest
import utils.constant as constant
from utils.benchmark_utils import restful_profile, restful_test
from utils.config_utils import get_func_config_list
from utils.proxy_distributed_utils import ApiServerPerTest, proxy_worker_node_wait


def get_models(backend, parallel_config):
    return get_func_config_list(backend, parallel_config, func_type='benchmark')


def _run_proxy_distributed_benchmark_test(config, run_config, manager=None):
    assert manager is not None, 'Manager instance must be provided'

    api_server = ApiServerPerTest(proxy_manager=manager, config=config, run_config=run_config)
    api_server.start()
    try:
        if manager.is_master:
            api_server.wait_until_ready()
            result, msg = restful_profile(config, run_config, port=constant.PROXY_PORT)
            assert result, msg
        else:
            proxy_worker_node_wait(manager, timeout_minutes=4880)
    finally:
        api_server.cleanup()


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
    'model': 'Qwen/Qwen3-VL-30B-A3B-Instruct',
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


@pytest.mark.pytorch
@pytest.mark.gpu_num_distributed_dpep8
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', get_models(backend='pytorch', parallel_config={'dp': 8, 'ep': 8}))
def test_pytorch_apiserver_distributed_dpep8(shared_proxy_manager, config, run_config, worker_id):
    _run_proxy_distributed_benchmark_test(config=config, run_config=run_config, manager=shared_proxy_manager)
