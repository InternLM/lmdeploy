import os

import pytest
import yaml
from utils.proxy_distributed_utils import ProxyDistributedManager
from utils.ray_distributed_utils import RayLMDeployManager

cli_prompt_case_file = 'autotest/chat_prompt_case.yaml'
common_prompt_case_file = 'autotest/prompt_case.yaml'
config_file = 'autotest/config.yaml'

PROXY_PORT = 8000


@pytest.fixture(scope='session')
def config():
    # Use device-specific config file if DEVICE environment variable is set
    device = os.environ.get('DEVICE', '')
    if device:
        device_config_path = f'autotest/config-{device}.yaml'
        if os.path.exists(device_config_path):
            config_path = device_config_path
        else:
            config_path = config_file
    else:
        config_path = config_file

    with open(config_path) as f:
        env_config = yaml.load(f.read(), Loader=yaml.SafeLoader)

    return env_config


@pytest.fixture(scope='session')
def cli_case_config():
    case_path = os.path.join(cli_prompt_case_file)
    with open(case_path) as f:
        case_config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return case_config


@pytest.fixture(scope='class', autouse=True)
def common_case_config():
    case_path = os.path.join(common_prompt_case_file)
    with open(case_path) as f:
        case_config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return case_config


@pytest.fixture(scope='session')
def shared_ray_manager():
    master_addr = os.getenv('MASTER_ADDR', 'localhost')
    device = os.environ.get('DEVICE', '')
    if device:
        device_config_path = f'autotest/config-{device}.yaml'
        if os.path.exists(device_config_path):
            config_path = device_config_path
        else:
            config_path = config_file
    else:
        config_path = config_file

    with open(config_path) as f:
        env_config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    log_dir = env_config.get('log_path', '/tmp/lmdeploy_test')

    manager = RayLMDeployManager(master_addr=master_addr, api_port=PROXY_PORT, log_dir=log_dir, health_check=True)

    manager.start_ray_cluster()

    if manager.is_master:
        print('🎯 Master node: Ray cluster started, waiting for worker nodes to join...')

    yield manager

    print(f'\n[Final Cleanup] Node {manager.node_rank} performing final resource cleanup...')
    manager.cleanup(force=True)


@pytest.fixture(scope='session')
def shared_proxy_manager():
    master_addr = os.getenv('MASTER_ADDR', 'localhost')

    manager = ProxyDistributedManager()

    if manager.is_master:
        manager.start()
        print(f'🎯 Master node: LMDeploy Proxy started on {master_addr}:{manager.proxy_port}')
        print('⏳ Waiting for worker nodes to connect...')

    yield manager

    print(f'\n[Final Cleanup] Node {manager.node_rank} performing final resource cleanup...')
    manager.cleanup()


def pytest_addoption(parser):
    parser.addoption('--run_id', action='store', default='', help='github run_id')
    parser.addoption('--device', action='store', default='', help='device config suffix')


def pytest_configure(config):
    # Set DEVICE environment variable before test execution
    device = config.getoption('--device')
    if device:
        os.environ['DEVICE'] = device


@pytest.fixture(scope='session')
def run_id(request):
    return request.config.getoption('--run_id')


@pytest.fixture(scope='session')
def device(request):
    return request.config.getoption('--device')
