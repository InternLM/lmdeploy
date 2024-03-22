import os

import yaml
from utils.get_run_config import get_tp_num


def get_turbomind_model_list(tp_num: int = None):
    config_path = os.path.join('autotest/config.yaml')
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

    if tp_num is not None:
        return [
            item for item in case_list if get_tp_num(config, item) == tp_num
        ]
    else:
        return case_list


def get_torch_model_list(tp_num: int = None):
    config_path = os.path.join('autotest/config.yaml')
    with open(config_path) as f:
        config = yaml.load(f.read(), Loader=yaml.SafeLoader)

    case_list = config.get('pytorch_model')
    quatization_case_config = config.get('quatization_case_config')
    for key in quatization_case_config.get('w8a8'):
        case_list.append(key + '-inner-w8a8')

    if tp_num is not None:
        return [
            item for item in case_list if get_tp_num(config, item) == tp_num
        ]
    else:
        return case_list


def get_all_model_list(tp_num: int = None):
    config_path = os.path.join('autotest/config.yaml')
    with open(config_path) as f:
        config = yaml.load(f.read(), Loader=yaml.SafeLoader)

    case_list = config.get('turbomind_model')
    for key in config.get('pytorch_model'):
        if key not in case_list:
            case_list.append(key)
    quatization_case_config = config.get('quatization_case_config')
    for key in quatization_case_config.get('w4a16'):
        case_list.append(key + '-inner-w4a16')
    for key in quatization_case_config.get('kvint8'):
        case_list.append(key + '-inner-kvint8')
    for key in quatization_case_config.get('kvint8_w4a16'):
        case_list.append(key + '-inner-kvint8-w4a16')

    if tp_num is not None:
        return [
            item for item in case_list if get_tp_num(config, item) == tp_num
        ]
    else:
        return case_list


def get_vl_model_list(tp_num: int = None):
    config_path = os.path.join('autotest/config.yaml')
    with open(config_path) as f:
        config = yaml.load(f.read(), Loader=yaml.SafeLoader)

    case_list = config.get('vl_model')

    if tp_num is not None:
        return [
            item for item in case_list if get_tp_num(config, item) == tp_num
        ]
    else:
        return case_list


def get_cuda_prefix_by_workerid(worker_id, tp_num: int = 1):
    if worker_id is None or 'gw' not in worker_id:
        return None
    else:
        if tp_num == 1:
            return 'CUDA_VISIBLE_DEVICES=' + worker_id.replace('gw', '')
        elif tp_num == 2:
            cuda_num = int(worker_id.replace('gw', '')) * 2
            return 'CUDA_VISIBLE_DEVICES=' + ','.join(
                [str(cuda_num), str(cuda_num + 1)])


def get_cuda_id_by_workerid(worker_id, tp_num: int = 1):
    if worker_id is None or 'gw' not in worker_id:
        return None
    else:
        if tp_num == 1:
            return worker_id.replace('gw', '')
        elif tp_num == 2:
            cuda_num = int(worker_id.replace('gw', '')) * 2
            return ','.join([str(cuda_num), str(cuda_num + 1)])


def get_workerid(worker_id):
    if worker_id is None or 'gw' not in worker_id:
        return None
    else:
        return int(worker_id.replace('gw', ''))
