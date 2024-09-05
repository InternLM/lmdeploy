import os

import yaml
from utils.get_run_config import get_tp_num


def get_turbomind_model_list(tp_num: int = None,
                             model_type: str = 'chat_model'):
    config = get_config()

    case_list = config.get('turbomind_' + model_type)
    quatization_case_config = config.get('turbomind_quatization')
    for key in quatization_case_config.get('awq'):
        if key in case_list:
            case_list.append(key + '-inner-4bits')
    for key in quatization_case_config.get('gptq'):
        if key in case_list:
            case_list.append(key + '-inner-gptq')

    if tp_num is not None:
        return [
            item for item in case_list if get_tp_num(config, item) == tp_num
        ]
    else:
        return case_list


def get_torch_model_list(tp_num: int = None, model_type: str = 'chat_model'):
    config = get_config()

    case_list = config.get('pytorch_' + model_type)
    quatization_case_config = config.get('pytorch_quatization')
    for key in quatization_case_config.get('w8a8'):
        if key in case_list:
            case_list.append(key + '-inner-w8a8')
    for key in quatization_case_config.get('awq'):
        if key in case_list:
            case_list.append(key + '-inner-4bits')

    if tp_num is not None:
        return [
            item for item in case_list if get_tp_num(config, item) == tp_num
        ]
    else:
        return case_list


def get_all_model_list(tp_num: int = None, model_type: str = 'chat_model'):
    config = get_config()

    case_list = config.get('turbomind_' + model_type)
    for key in config.get('pytorch_' + model_type):
        if key not in case_list:
            case_list.append(key)
    turbomind_quantization_config = config.get('turbomind_quatization')
    pytorch_quantization_config = config.get('pytorch_quatization')
    for key in turbomind_quantization_config.get(
            'awq') + pytorch_quantization_config.get(
                'awq') + turbomind_quantization_config.get('gptq'):
        if key in case_list and key + '-inner-4bits' not in case_list:
            case_list.append(key + '-inner-4bits')

    if tp_num is not None:
        return [
            item for item in case_list if get_tp_num(config, item) == tp_num
        ]
    else:
        return case_list


def get_kvint_model_list(tp_num: int = None, model_type: str = 'chat_model'):
    config = get_config()

    case_list_base = config.get('turbomind_' + model_type)
    for key in config.get('pytorch_' + model_type):
        if key not in case_list_base:
            case_list_base.append(key)

    case_list = []
    for key in config.get('turbomind_quatization').get('kvint'):
        if key in case_list_base:
            case_list.append(key)

    for key in config.get('turbomind_quatization').get('awq'):
        if key in case_list_base and key in case_list:
            case_list.append(key + '-inner-4bits')
    for key in config.get('turbomind_quatization').get('gptq'):
        if key in case_list_base and key in case_list:
            case_list.append(key + '-inner-gptq')

    if tp_num is not None:
        return [
            item for item in case_list if get_tp_num(config, item) == tp_num
        ]
    else:
        return case_list


def get_quantization_model_list(type):
    config = get_config()
    if type == 'awq':
        case_list = config.get('turbomind_quatization').get('awq')
        for key in config.get('pytorch_quatization').get('awq'):
            if key not in case_list:
                case_list.append(key)
        return case_list
    if type == 'kvint':
        return config.get('turbomind_quatization').get(type)
    if type == 'gptq':
        return config.get('turbomind_quatization').get(type)
    if type == 'w8a8':
        return config.get('pytorch_quatization').get(type)
    return []


def get_vl_model_list(tp_num: int = None):
    config = get_config()

    case_list = config.get('vl_model')

    for key in config.get('turbomind_quatization').get('awq'):
        if key in case_list:
            case_list.append(key + '-inner-4bits')

    if tp_num is not None:
        return [
            item for item in case_list if get_tp_num(config, item) == tp_num
        ]
    else:
        return case_list


def get_cuda_prefix_by_workerid(worker_id, tp_num: int = 1):
    cuda_id = get_cuda_id_by_workerid(worker_id, tp_num)
    if cuda_id is None or 'gw' not in worker_id:
        return None
    else:
        return 'CUDA_VISIBLE_DEVICES=' + cuda_id


def get_cuda_id_by_workerid(worker_id, tp_num: int = 1):
    if worker_id is None or 'gw' not in worker_id:
        return None
    else:
        if tp_num == 1:
            return worker_id.replace('gw', '')
        elif tp_num == 2:
            cuda_num = int(worker_id.replace('gw', '')) * 2
            return ','.join([str(cuda_num), str(cuda_num + 1)])
        elif tp_num == 4:
            cuda_num = int(worker_id.replace('gw', '')) * 4
            return ','.join([
                str(cuda_num),
                str(cuda_num + 1),
                str(cuda_num + 2),
                str(cuda_num + 3)
            ])


def get_config():
    config_path = os.path.join('autotest/config.yaml')
    with open(config_path) as f:
        config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return config


def get_benchmark_model_list(tp_num,
                             is_longtext: bool = False,
                             kvint_list: list = []):
    config = get_config()
    if is_longtext:
        case_list = [item for item in config.get('longtext_model')]
    else:
        case_list = config.get('benchmark_model')
    quatization_case_config = config.get('turbomind_quatization')
    for key in quatization_case_config.get('awq'):
        if key in case_list:
            case_list.append(key + '-inner-4bits')

    model_list = [
        item for item in case_list if get_tp_num(config, item) == tp_num
    ]

    result = []
    if len(model_list) > 0:
        result += [{
            'model': item,
            'backend': 'turbomind',
            'quant_policy': 0,
            'tp_num': tp_num
        } for item in model_list if item.replace('-inner-4bits', '') in
                   config.get('turbomind_chat_model') or tp_num == 4]
        result += [{
            'model': item,
            'backend': 'pytorch',
            'tp_num': tp_num
        } for item in model_list if '4bits' not in item and (
            item in config.get('pytorch_chat_model') or tp_num == 4)]
        for kvint in kvint_list:
            result += [{
                'model': item,
                'backend': 'turbomind',
                'quant_policy': kvint,
                'tp_num': tp_num
            } for item in model_list if item.replace('-inner-4bits', '') in
                       config.get('turbomind_quatization').get('kvint')]
    return result


def get_workerid(worker_id):
    if worker_id is None or 'gw' not in worker_id:
        return None
    else:
        return int(worker_id.replace('gw', ''))
