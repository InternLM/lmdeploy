import pytest
from utils.benchmark_utils import restful_test
from utils.config_utils import get_case_str_by_config, get_func_config_list
from utils.pytest_layout_utils import LOCAL_TP_LAYOUTS, layout_mark

TURBOMIND_LAYOUTS = LOCAL_TP_LAYOUTS[:4]
PYTORCH_LAYOUTS = LOCAL_TP_LAYOUTS


def get_models(backend, parallel_config):
    return get_func_config_list(
        backend, parallel_config, model_type='vl_model', func_type='mllm_evaluate',
    )


def _build_mllm_apiserver_params():
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


_MLLM_APISERVER_PARAMS = _build_mllm_apiserver_params()


@pytest.mark.parametrize('run_config', _MLLM_APISERVER_PARAMS)
def test_mllm_apiserver_performance(config, run_config, worker_id):
    result, msg = restful_test(config, run_config, worker_id=worker_id, is_mllm=True)
    assert result, msg
