"""Interface REST coverage for LLM (self-start api_server, GPU xdist)."""

import pytest
from utils.config_utils import get_interface_backend_list, get_interface_run_config_list
from utils.pytest_layout_utils import (
    DISTRIBUTED_DP_EP_EQUAL_LAYOUTS,
    DISTRIBUTED_DP_EP_LAYOUTS,
    LOCAL_TP_LAYOUTS,
    layout_mark,
)
from utils.run_interface_restful import (
    run_interface_restful_proxy_distributed_test,
    run_interface_restful_ray_distributed_test,
    run_interface_restful_test,
)


def _iface_configs(parallel_config: dict):
    rows = []
    for backend in get_interface_backend_list():
        rows.extend(
            get_interface_run_config_list(
                backend, parallel_config, model_types=('chat', 'base'),
            ),
        )
    return rows


def _iface_id(run_config):
    layout = '-'.join(
        f'{k}{v}' for k, v in sorted(run_config['parallel_config'].items())
    )
    return f"{run_config['backend']}-{run_config['model']}-{layout}"


def _iface_layout_marks(layout: dict[str, int]):
    marks = [pytest.mark.test_ascend]
    if layout == {'tp': 1}:
        marks.append(pytest.mark.test_3090)
    return marks


def _build_iface_params(layouts):
    rows = []
    for layout in layouts:
        marks = [layout_mark(layout), pytest.mark.flaky(reruns=0), *_iface_layout_marks(layout)]
        for run_config in _iface_configs(layout):
            rows.append(pytest.param(run_config, marks=marks, id=_iface_id(run_config)))
    return rows


_LOCAL_IFACE_PARAMS = _build_iface_params(LOCAL_TP_LAYOUTS)
_RAY_IFACE_PARAMS = _build_iface_params(({'tp': 16},))
_PROXY_IFACE_PARAMS = _build_iface_params(
    DISTRIBUTED_DP_EP_EQUAL_LAYOUTS[:1] + DISTRIBUTED_DP_EP_LAYOUTS,
)


@pytest.mark.parametrize('run_config', _LOCAL_IFACE_PARAMS)
def test_restful_interface_local(config, run_config, worker_id):
    run_interface_restful_test(config, run_config, worker_id)


@pytest.mark.distributed
@pytest.mark.parametrize('run_config', _RAY_IFACE_PARAMS)
def test_restful_interface_ray(shared_ray_manager, config, run_config, worker_id):
    del worker_id
    run_interface_restful_ray_distributed_test(config, run_config, shared_ray_manager)


@pytest.mark.distributed
@pytest.mark.parametrize('run_config', _PROXY_IFACE_PARAMS)
def test_restful_interface_proxy(shared_proxy_manager, config, run_config, worker_id):
    del worker_id
    run_interface_restful_proxy_distributed_test(
        config, run_config, shared_proxy_manager,
    )
