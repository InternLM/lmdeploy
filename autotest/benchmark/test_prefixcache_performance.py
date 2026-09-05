import pytest
from utils.benchmark_utils import prefixcache_throughput_test
from utils.config_utils import get_case_str_by_config, get_func_config_list
from utils.pytest_layout_utils import LOCAL_TP_LAYOUTS, layout_mark

TURBOMIND_LAYOUTS = LOCAL_TP_LAYOUTS[:4]
PYTORCH_LAYOUTS = LOCAL_TP_LAYOUTS


def get_models(backend, parallel_config):
    return get_func_config_list(backend, parallel_config, func_type='benchmark') + \
        get_func_config_list(backend, parallel_config, func_type='benchmark', extra={'enable-prefix-caching': True})


def _build_prefix_params():
    rows = []
    for backend, layouts in (('turbomind', TURBOMIND_LAYOUTS), ('pytorch', PYTORCH_LAYOUTS)):
        backend_mark = getattr(pytest.mark, backend)
        for layout in layouts:
            configs = get_models(backend, layout)
            if not configs:
                continue
            marks = [layout_mark(layout), backend_mark, pytest.mark.flaky(reruns=0)]
            for run_config in configs:
                rows.append(
                    pytest.param(
                        run_config,
                        marks=marks,
                        id=get_case_str_by_config(run_config),
                    ))
    return rows


_PREFIX_PARAMS = _build_prefix_params()

_FUNC_SMOKE_CONFIGS = [{
    'model': 'Qwen/Qwen3-30B-A3B',
    'backend': 'turbomind',
    'communicator': 'cuda-ipc',
    'quant_policy': 0,
    'parallel_config': {'tp': 2},
    'extra_params': {}
}, {
    'model': 'Qwen/Qwen3-VL-30B-A3B-Instruct',
    'backend': 'pytorch',
    'communicator': 'nccl',
    'quant_policy': 8,
    'parallel_config': {'tp': 2},
    'extra_params': {}
}, {
    'model': 'Qwen/Qwen3-30B-A3B',
    'backend': 'turbomind',
    'communicator': 'cuda-ipc',
    'quant_policy': 0,
    'parallel_config': {'tp': 2},
    'extra_params': {'enable-prefix-caching': True}
}, {
    'model': 'Qwen/Qwen3-VL-30B-A3B-Instruct',
    'backend': 'pytorch',
    'communicator': 'nccl',
    'quant_policy': 8,
    'parallel_config': {'tp': 2},
    'extra_params': {'enable-prefix-caching': True}
}]


@pytest.mark.parametrize('run_config', _PREFIX_PARAMS)
def test_prefixcache_performance(config, run_config, worker_id):
    result, msg = prefixcache_throughput_test(config, run_config, worker_id=worker_id)
    assert result, msg


@pytest.mark.flaky(reruns=0)
@pytest.mark.function
@layout_mark({'tp': 2})
@pytest.mark.parametrize('run_config', _FUNC_SMOKE_CONFIGS)
def test_prefixcache_func_tp2(config, run_config, worker_id):
    result, msg = prefixcache_throughput_test(config, run_config, worker_id=worker_id, is_smoke=True)
    assert result, msg
