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


@pytest.fixture(scope='session')
def case_config(request):
    case_path = os.path.join(request.config.rootdir, 'chat_prompt_case.yaml')
    print(case_path)
    with open(case_path) as f:
        case_config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return case_config


@pytest.fixture(scope='class', autouse=True)
def restful_case_config(request):
    case_path = os.path.join(request.config.rootdir,
                             'restful_prompt_case.yaml')
    print(case_path)
    with open(case_path) as f:
        case_config = yaml.load(f.read(), Loader=yaml.SafeLoader)
        del case_config['session_len_error']
    return case_config


def _init_case_list():
    case_path = os.path.join('autotest/chat_prompt_case.yaml')
    with open(case_path) as f:
        case_config = yaml.load(f.read(), Loader=yaml.SafeLoader)

    global global_case_List
    global_case_List = list(case_config.keys())


def _init_restful_case_list():
    case_path = os.path.join('autotest/chat_prompt_case.yaml')
    with open(case_path) as f:
        case_config = yaml.load(f.read(), Loader=yaml.SafeLoader)
        del case_config['session_len_error']

    global global_restful_case_List
    global_restful_case_List = list(case_config.keys())
