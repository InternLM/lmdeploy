import pytest
import utils.constant as constant
from utils.benchmark_utils import restful_profile, restful_test
from utils.config_utils import get_func_config_list
from utils.proxy_distributed_utils import ApiServerPerTest, proxy_worker_node_wait
from utils.pytest_layout_utils import (
    DISTRIBUTED_DP_EP_EQUAL_LAYOUTS,
    LOCAL_TP_LAYOUTS,
    build_layout_params,
    build_multi_backend_layout_params,
    layout_mark,
)

TURBOMIND_LAYOUTS = LOCAL_TP_LAYOUTS[:4]
PYTORCH_LAYOUTS = LOCAL_TP_LAYOUTS


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


_APISERVER_PARAMS = build_multi_backend_layout_params(
    (
        ('turbomind', TURBOMIND_LAYOUTS),
        ('pytorch', PYTORCH_LAYOUTS),
    ),
    func_type='benchmark',
    param_marks=[pytest.mark.flaky(reruns=0)],
)

_PROXY_PARAMS = build_layout_params(
    'pytorch',
    DISTRIBUTED_DP_EP_EQUAL_LAYOUTS[:1],
    func_type='benchmark',
    param_marks=[pytest.mark.flaky(reruns=0), pytest.mark.pytorch],
)

_FUNC_SMOKE_CONFIGS = [{
    'model': 'Qwen/Qwen3-30B-A3B',
    'backend': 'pytorch',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {'tp': 2},
    'extra_params': {}
}, {
    'model': 'Qwen/Qwen3-30B-A3B',
    'backend': 'turbomind',
    'communicator': 'nccl',
    'quant_policy': 4,
    'parallel_config': {'tp': 2},
    'extra_params': {}
}, {
    'model': 'Qwen/Qwen3-30B-A3B',
    'backend': 'turbomind',
    'communicator': 'cuda-ipc',
    'quant_policy': 8,
    'parallel_config': {'tp': 2},
    'extra_params': {}
}, {
    'model': 'Qwen/Qwen3-VL-30B-A3B-Instruct',
    'backend': 'pytorch',
    'communicator': 'nccl',
    'quant_policy': 8,
    'parallel_config': {'tp': 2},
    'extra_params': {}
}]


@pytest.mark.parametrize('run_config', _APISERVER_PARAMS)
def test_apiserver_performance(config, run_config, worker_id):
    result, msg = restful_test(config, run_config, worker_id=worker_id)
    assert result, msg


@pytest.mark.distributed
@pytest.mark.parametrize('run_config', _PROXY_PARAMS)
def test_apiserver_performance_proxy(shared_proxy_manager, config, run_config, worker_id):
    del worker_id
    _run_proxy_distributed_benchmark_test(
        config=config, run_config=run_config, manager=shared_proxy_manager,
    )


@pytest.mark.function
@pytest.mark.flaky(reruns=0)
@layout_mark({'tp': 2})
@pytest.mark.parametrize('run_config', _FUNC_SMOKE_CONFIGS)
def test_restful_func_tp2(config, run_config, worker_id):
    result, msg = restful_test(config, run_config, worker_id=worker_id, is_smoke=True)
    assert result, msg
