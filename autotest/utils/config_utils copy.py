import copy
import os

import yaml


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
        case_list_base = [item for item in config.get('longtext_model')]
    else:
        case_list_base = config.get('benchmark_model')
    quatization_case_config = config.get('turbomind_quatization')

    case_list = copy.deepcopy(case_list_base)
    for key in case_list_base:
        if key in config.get('turbomind_chat_model'
                             ) and key not in quatization_case_config.get(
                                 'no_awq') and not is_quantization_model(key):
            case_list.append(key + '-inner-4bits')

    model_list = [item for item in case_list]

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
                       config.get('turbomind_chat_model')]
    return result


def get_workerid(worker_id):
    if worker_id is None or 'gw' not in worker_id:
        return None
    else:
        return int(worker_id.replace('gw', ''))


def is_quantization_model(name):
    return 'awq' in name.lower() or '4bits' in name.lower(
    ) or 'w4' in name.lower() or 'int4' in name.lower()


if __name__ == '__main__':
    print(get_benchmark_model_list)
