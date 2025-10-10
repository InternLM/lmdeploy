import os

import pytest
import yaml

cli_prompt_case_file = 'autotest/chat_prompt_case.yaml'
common_prompt_case_file = 'autotest/prompt_case.yaml'
config_file = 'autotest/config.yaml'


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
