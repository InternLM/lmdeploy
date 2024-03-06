import os

import yaml


def get_turbomind_model_list():
    config_path = os.path.join('autotest/config.yaml')
    print(config_path)
    with open(config_path) as f:
        config = yaml.load(f.read(), Loader=yaml.SafeLoader)

    case_list = config.get('turbomind_model')
    quatization_case_config = config.get('quatization_case_config')
    for key in quatization_case_config.get('w4a16'):
        case_list.append(key + '-inner-w4a16')
    for key in quatization_case_config.get('kvint8'):
        case_list.append(key + '-inner-kvint8')
    for key in quatization_case_config.get('kvint8_w4a16'):
        case_list.append(key + '-inner-kvint8-w4a16')

    return case_list


def get_torch_model_list():
    config_path = os.path.join('autotest/config.yaml')
    print(config_path)
    with open(config_path) as f:
        config = yaml.load(f.read(), Loader=yaml.SafeLoader)

    case_list = config.get('pytorch_model')
    quatization_case_config = config.get('quatization_case_config')
    for key in quatization_case_config.get('w8a8'):
        case_list.append(key + '-inner-w8a8')

    return case_list
