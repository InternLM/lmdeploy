"""Interface REST coverage for LLM (self-start api_server, GPU xdist)."""

import pytest
from utils.config_utils import get_interface_run_config_list
from utils.constant import BACKEND_LIST
from utils.run_interface_restful import run_interface_restful_test


def _iface_configs(tp: int):
    rows = []
    for backend in BACKEND_LIST:
        rows.extend(
            get_interface_run_config_list(backend, {'tp': tp}, model_types=('chat', 'base')),
        )
    return rows


def _iface_id(run_config):
    return f"{run_config['backend']}-{run_config['model']}-tp{run_config['parallel_config'].get('tp', 1)}"


@pytest.mark.gpu_num_1
@pytest.mark.test_3090
@pytest.mark.test_ascend
@pytest.mark.parametrize('run_config', _iface_configs(1), ids=_iface_id)
def test_restful_interface_tp1(config, run_config, worker_id):
    run_interface_restful_test(config, run_config, worker_id)


@pytest.mark.gpu_num_2
@pytest.mark.test_ascend
@pytest.mark.parametrize('run_config', _iface_configs(2), ids=_iface_id)
def test_restful_interface_tp2(config, run_config, worker_id):
    run_interface_restful_test(config, run_config, worker_id)


@pytest.mark.gpu_num_4
@pytest.mark.test_ascend
@pytest.mark.parametrize('run_config', _iface_configs(4), ids=_iface_id)
def test_restful_interface_tp4(config, run_config, worker_id):
    run_interface_restful_test(config, run_config, worker_id)


@pytest.mark.gpu_num_8
@pytest.mark.test_ascend
@pytest.mark.parametrize('run_config', _iface_configs(8), ids=_iface_id)
def test_restful_interface_tp8(config, run_config, worker_id):
    run_interface_restful_test(config, run_config, worker_id)
