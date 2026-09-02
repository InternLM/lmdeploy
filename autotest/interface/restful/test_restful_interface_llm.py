"""Interface REST coverage for LLM (self-start api_server, GPU xdist)."""

import pytest
from utils.config_utils import get_interface_backend_list, get_interface_run_config_list
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


@pytest.mark.gpu_num_1
@pytest.mark.test_3090
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', _iface_configs({'tp': 1}), ids=_iface_id)
def test_restful_interface_tp1(config, run_config, worker_id):
    run_interface_restful_test(config, run_config, worker_id)


@pytest.mark.gpu_num_2
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', _iface_configs({'tp': 2}), ids=_iface_id)
def test_restful_interface_tp2(config, run_config, worker_id):
    run_interface_restful_test(config, run_config, worker_id)


@pytest.mark.gpu_num_4
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', _iface_configs({'tp': 4}), ids=_iface_id)
def test_restful_interface_tp4(config, run_config, worker_id):
    run_interface_restful_test(config, run_config, worker_id)


@pytest.mark.gpu_num_8
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', _iface_configs({'tp': 8}), ids=_iface_id)
def test_restful_interface_tp8(config, run_config, worker_id):
    run_interface_restful_test(config, run_config, worker_id)


@pytest.mark.gpu_num_16
@pytest.mark.test_ascend
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', _iface_configs({'tp': 16}), ids=_iface_id)
def test_restful_interface_tp16(config, run_config, worker_id):
    run_interface_restful_test(config, run_config, worker_id)


@pytest.mark.gpu_num_distributed_tp16
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize('run_config', _iface_configs({'tp': 16}), ids=_iface_id)
def test_restful_interface_distributed_tp16(
        shared_ray_manager, config, run_config, worker_id):
    del worker_id
    run_interface_restful_ray_distributed_test(config, run_config, shared_ray_manager)


@pytest.mark.gpu_num_distributed_dpep16
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize(
    'run_config', _iface_configs({'dp': 16, 'ep': 16}), ids=_iface_id,
)
def test_restful_interface_distributed_dpep16(
        shared_proxy_manager, config, run_config, worker_id):
    del worker_id
    run_interface_restful_proxy_distributed_test(
        config, run_config, shared_proxy_manager,
    )


@pytest.mark.gpu_num_distributed_tp2dp4ep8
@pytest.mark.flaky(reruns=0)
@pytest.mark.parametrize(
    'run_config', _iface_configs({'tp': 2, 'dp': 4, 'ep': 8}), ids=_iface_id,
)
def test_restful_interface_distributed_tp2dp4ep8(
        shared_proxy_manager, config, run_config, worker_id):
    del worker_id
    run_interface_restful_proxy_distributed_test(
        config, run_config, shared_proxy_manager,
    )
