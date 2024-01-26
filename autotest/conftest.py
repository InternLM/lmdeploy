import os

import pytest
import yaml


@pytest.fixture(scope='session')
def config(request):
    config_path = os.path.join(request.config.rootdir, 'config.yaml')
    print(config_path)
    with open(config_path) as f:
        env_config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return env_config


cli_prompt_case_file = 'autotest/chat_prompt_case.yaml'
common_prompt_case_file = 'autotest/prompt_case.yaml'


@pytest.fixture(scope='session')
def cli_case_config():
    case_path = os.path.join(cli_prompt_case_file)
    print(case_path)
    with open(case_path) as f:
        case_config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return case_config


@pytest.fixture(scope='class', autouse=True)
def common_case_config():
    case_path = os.path.join(common_prompt_case_file)
    print(case_path)
    with open(case_path) as f:
        case_config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return case_config


def _init_cli_case_list():
    case_path = os.path.join(cli_prompt_case_file)
    with open(case_path) as f:
        case_config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    global global_cli_case_List
    global_cli_case_List = list(case_config.keys())


def _init_common_case_list():
    case_path = os.path.join(common_prompt_case_file)
    with open(case_path) as f:
        case_config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    global global_common_case_List
    global_common_case_List = list(case_config.keys())
