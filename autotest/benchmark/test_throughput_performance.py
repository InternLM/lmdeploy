import pytest
from utils.benchmark_utils import throughput_test
from utils.config_utils import get_case_str_by_config, get_func_config_list, get_workerid
from utils.pytest_layout_utils import LOCAL_TP_LAYOUTS, layout_mark

TURBOMIND_LAYOUTS = LOCAL_TP_LAYOUTS[:4]
PYTORCH_LAYOUTS = LOCAL_TP_LAYOUTS


def get_models(backend, parallel_config):
    run_configs = get_func_config_list(backend, parallel_config, func_type='benchmark')
    return [item for item in run_configs
            if 'gpt' not in item['model']]


def _filtered_multi_backend_params():
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


_THROUGHPUT_PARAMS = _filtered_multi_backend_params()

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
}]

_PR_SMOKE_CONFIGS = [{
    'model': 'meta-llama/Meta-Llama-3-1-8B-Instruct',
    'backend': 'turbomind',
    'communicator': 'nccl',
    'quant_policy': 0,
    'parallel_config': {'tp': 1},
    'extra_params': {}
}, {
    'model': 'Qwen/Qwen3-VL-8B-Instruct',
    'backend': 'pytorch',
    'communicator': 'nccl',
    'quant_policy': 8,
    'parallel_config': {'tp': 1},
    'extra_params': {}
}]


@pytest.mark.parametrize('run_config', _THROUGHPUT_PARAMS)
def test_throughput_performance(config, run_config, worker_id):
    result, msg = throughput_test(config, run_config, worker_id=worker_id)
    assert result, msg


@pytest.mark.function
@pytest.mark.flaky(reruns=0)
@layout_mark({'tp': 2})
@pytest.mark.parametrize('run_config', _FUNC_SMOKE_CONFIGS)
def test_throughput_func_tp2(config, run_config, worker_id):
    result, msg = throughput_test(config, run_config, worker_id=worker_id, is_smoke=True)
    assert result, msg


@pytest.mark.flaky(reruns=0)
@pytest.mark.pr_test
@layout_mark({'tp': 1})
@pytest.mark.parametrize('run_config', _PR_SMOKE_CONFIGS)
def test_throughput_prtest_tp1(config, run_config, worker_id):
    worker_id = 'gw' + str(6 + get_workerid(worker_id))
    result, msg = throughput_test(config, run_config, worker_id=worker_id, is_smoke=True)
    assert result, msg
