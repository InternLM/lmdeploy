import copy
import os
from collections import OrderedDict

import yaml
from utils.get_run_config import get_tp_num

from lmdeploy.utils import is_bf16_supported


def get_turbomind_model_list(tp_num: int = None, model_type: str = 'chat_model', quant_policy: int = None):
    config = get_config()

    if quant_policy is None:
        case_list = copy.deepcopy(config.get('turbomind_' + model_type))
    else:
        case_list = [
            x for x in config.get('turbomind_' + model_type)
            if x not in config.get('turbomind_quatization').get('no_kvint' + str(quant_policy))
        ]

    quatization_case_config = config.get('turbomind_quatization')
    for key in config.get('turbomind_' + model_type):
        if key in case_list and key not in quatization_case_config.get('no_awq') and not is_quantization_model(key):
            case_list.append(key + '-inner-4bits')
    for key in quatization_case_config.get('gptq'):
        if key in case_list:
            case_list.append(key + '-inner-gptq')

    if tp_num is not None:
        return [item for item in case_list if get_tp_num(config, item) == tp_num]
    else:
        return case_list


def get_torch_model_list(tp_num: int = None,
                         model_type: str = 'chat_model',
                         exclude_dup: bool = False,
                         quant_policy: int = None):
    config = get_config()
    exclude_dup = False

    if exclude_dup:
        if quant_policy is None:
            case_list = [x for x in config.get('pytorch_' + model_type) if x in config.get('turbomind_' + model_type)]
        else:
            case_list = [
                x for x in config.get('pytorch_' + model_type)
                if x in config.get('turbomind_' +
                                   model_type) and x not in config.get('pytorch_quatization').get('no_kvint' +
                                                                                                  str(quant_policy))
            ]
    else:
        if quant_policy is None:
            case_list = config.get('pytorch_' + model_type)
        else:
            case_list = [
                x for x in config.get('pytorch_' + model_type)
                if x not in config.get('pytorch_quatization').get('no_kvint' + str(quant_policy))
            ]

    quatization_case_config = config.get('pytorch_quatization')
    for key in quatization_case_config.get('w8a8'):
        if key in case_list:
            case_list.append(key + '-inner-w8a8')
    for key in quatization_case_config.get('awq'):
        if key in case_list:
            case_list.append(key + '-inner-4bits')

    if tp_num is not None:
        return [item for item in case_list if get_tp_num(config, item) == tp_num]
    else:
        return case_list


def get_all_model_list(tp_num: int = None, quant_policy: int = None, model_type: str = 'chat_model'):

    case_list = get_turbomind_model_list(tp_num=tp_num, model_type=model_type, quant_policy=quant_policy)
    if _is_bf16_supported_by_device():
        for case in get_torch_model_list(tp_num=tp_num, quant_policy=quant_policy, model_type=model_type):
            if case not in case_list:
                case_list.append(case)
    return case_list


def get_communicator_list(tp_num: int = None):
    if tp_num != 1 and _is_bf16_supported_by_device():
        return ['cuda-ipc', 'nccl']
    return ['nccl']


def get_quantization_model_list(type):
    config = get_config()
    if type == 'awq':
        case_list = [
            x
            for x in list(OrderedDict.fromkeys(config.get('turbomind_chat_model') + config.get('turbomind_base_model')))
            if x not in config.get('turbomind_quatization').get('no_awq') and not is_quantization_model(x)
        ]
        for key in config.get('pytorch_quatization').get('awq'):
            if key not in case_list:
                case_list.append(key)
        return case_list
    if type == 'gptq':
        return config.get('turbomind_quatization').get(type)
    if type == 'w8a8':
        return config.get('pytorch_quatization').get(type)
    return []


def get_vl_model_list(tp_num: int = None, quant_policy: int = None):
    config = get_config()
    if quant_policy is None:
        case_list = copy.deepcopy(config.get('vl_model'))
    else:
        case_list = [
            x for x in config.get('vl_model')
            if (x in config.get('turbomind_chat_model') and x not in config.get('turbomind_quatization').get(
                'no_kvint' + str(quant_policy))) or (x in config.get('pytorch_chat_model') and x not in config.get(
                    'pytorch_quatization').get('no_kvint' + str(quant_policy)))
        ]

    for key in config.get('vl_model'):
        if key in config.get('turbomind_chat_model') and key not in config.get('turbomind_quatization').get(
                'no_awq') and not is_quantization_model(key) and key + '-inner-4bits' not in case_list and (
                    quant_policy is not None
                    and key not in config.get('turbomind_quatization').get('no_kvint' + str(quant_policy))):
            case_list.append(key + '-inner-4bits')
        if key in config.get('pytorch_chat_model') and key in config.get('pytorch_quatization').get(
                'awq') and not is_quantization_model(key) and key + '-inner-4bits' not in case_list and (
                    quant_policy is not None
                    and key not in config.get('pytorch_quatization').get('no_kvint' + str(quant_policy))):
            case_list.append(key + '-inner-4bits')
    if tp_num is not None:
        return [item for item in case_list if get_tp_num(config, item) == tp_num]
    else:
        return case_list


def get_cuda_prefix_by_workerid(worker_id, tp_num: int = 1):
    cuda_id = get_cuda_id_by_workerid(worker_id, tp_num)
    if cuda_id is None or 'gw' not in worker_id:
        return None
    else:
        device_type = os.environ.get('DEVICE', 'cuda')
        if device_type == 'ascend':
            return 'ASCEND_RT_VISIBLE_DEVICES=' + cuda_id
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
            return ','.join([str(cuda_num), str(cuda_num + 1), str(cuda_num + 2), str(cuda_num + 3)])


def get_config():
    # Determine config file based on DEVICE environment variable
    device = os.environ.get('DEVICE', '')
    if device:
        config_path = f'autotest/config-{device}.yaml'
        # Fallback to default config if device-specific config doesn't exist
        if not os.path.exists(config_path):
            config_path = 'autotest/config.yaml'
    else:
        config_path = 'autotest/config.yaml'

    with open(config_path) as f:
        config = yaml.load(f.read(), Loader=yaml.SafeLoader)

    config_copy = copy.deepcopy(config)
    github_run_id = os.environ.get('GITHUB_RUN_ID', 'local_run')
    if 'log_path' in config_copy:
        config_copy['log_path'] = os.path.join(config_copy['log_path'], str(github_run_id))
        os.makedirs(config_copy['log_path'], exist_ok=True)

    return config_copy


def get_benchmark_model_list(tp_num, is_longtext: bool = False, kvint_list: list = []):
    config = get_config()
    if is_longtext:
        case_list_base = [item for item in config.get('longtext_model')]
    else:
        case_list_base = config.get('benchmark_model')
    quatization_case_config = config.get('turbomind_quatization')
    pytorch_quatization_case_config = config.get('pytorch_quatization')

    case_list = copy.deepcopy(case_list_base)
    for key in case_list_base:
        if key in config.get('turbomind_chat_model') and key not in quatization_case_config.get(
                'no_awq') and not is_quantization_model(key):
            case_list.append(key + '-inner-4bits')

    for key in case_list_base:
        if key in config.get('pytorch_chat_model') and key in pytorch_quatization_case_config.get(
                'w8a8') and not is_quantization_model(key):
            case_list.append(key + '-inner-w8a8')

    model_list = [item for item in case_list if get_tp_num(config, item) == tp_num]

    result = []
    if len(model_list) > 0:
        result += [{
            'model': item,
            'backend': 'turbomind',
            'quant_policy': 0,
            'tp_num': tp_num
        } for item in model_list
                   if item.replace('-inner-4bits', '') in config.get('turbomind_chat_model') or tp_num == 4]
        result += [{
            'model': item,
            'backend': 'pytorch',
            'tp_num': tp_num
        } for item in model_list if '4bits' not in item and (
            item.replace('-inner-w8a8', '') in config.get('pytorch_chat_model') or tp_num == 4)]
        for kvint in kvint_list:
            result += [{
                'model': item,
                'backend': 'turbomind',
                'quant_policy': kvint,
                'tp_num': tp_num
            } for item in model_list if item.replace('-inner-4bits', '') in config.get('turbomind_chat_model')
                       and item.replace('-inner-4bits', '') not in quatization_case_config.get('no_kvint' + str(kvint))]
    return result


def get_evaluate_turbomind_model_list(tp_num, is_longtext: bool = False, kvint_list: list = []):
    config = get_config()

    if is_longtext:
        case_list_base = [item for item in config.get('longtext_model', [])]
    else:
        case_list_base = config.get('evaluate_model', config.get('benchmark_model', []))
    quatization_case_config = config.get('turbomind_quatization')

    case_list = copy.deepcopy(case_list_base)
    for key in case_list_base:
        if key in config.get('turbomind_chat_model') and key not in quatization_case_config.get(
                'no_awq') and not is_quantization_model(key):
            case_list.append(key + '-inner-4bits')

    model_list = [item for item in case_list if get_tp_num(config, item) == tp_num]

    result = []
    if len(model_list) > 0:

        if tp_num > 1:
            communicators = ['cuda-ipc', 'nccl']
        else:
            communicators = ['cuda-ipc']
        for communicator in communicators:
            for item in model_list:
                if item.replace('-inner-4bits', '') in config.get('turbomind_chat_model') or item.replace(
                        '-inner-4bits', '') in config.get('turbomind_base_model'):
                    model_config = {
                        'model': item,
                        'backend': 'turbomind',
                        'communicator': communicator,
                        'quant_policy': 0,
                        'tp_num': tp_num,
                        'extra': f'--communicator {communicator} '
                    }
                    result.append(model_config)

        for kvint in kvint_list:
            for item in model_list:
                if item.replace('-inner-4bits', '') in config.get('turbomind_chat_model') and item.replace(
                        '-inner-4bits', '') not in quatization_case_config.get('no_kvint' + str(kvint)):
                    model_config = {
                        'model': item,
                        'backend': 'turbomind',
                        'quant_policy': kvint,
                        'tp_num': tp_num,
                        'extra': ''
                    }
                    result.append(model_config)
    return result


def get_evaluate_pytorch_model_list(tp_num, is_longtext: bool = False, kvint_list: list = []):
    config = get_config()

    if is_longtext:
        case_list_base = [item for item in config.get('longtext_model', [])]
    else:
        case_list_base = config.get('evaluate_model', config.get('benchmark_model', []))
    pytorch_quatization_case_config = config.get('pytorch_quatization')

    case_list = copy.deepcopy(case_list_base)

    for key in case_list_base:
        if key in config.get('pytorch_chat_model') and key in pytorch_quatization_case_config.get(
                'w8a8') and not is_quantization_model(key):
            case_list.append(key + '-inner-w8a8')

    model_list = [item for item in case_list if get_tp_num(config, item) == tp_num]

    result = []
    if len(model_list) > 0:
        for item in model_list:
            if '4bits' not in item and (item.replace('-inner-w8a8', '') in config.get('pytorch_chat_model')
                                        or item.replace('-inner-w8a8', '') in config.get('pytorch_base_model')):
                model_config = {'model': item, 'backend': 'pytorch', 'tp_num': tp_num, 'extra': ''}
                result.append(model_config)
    return result


def get_workerid(worker_id):
    if worker_id is None or 'gw' not in worker_id:
        return None
    else:
        return int(worker_id.replace('gw', ''))


def is_quantization_model(name):
    return 'awq' in name.lower() or '4bits' in name.lower() or 'w4' in name.lower() or 'int4' in name.lower()


def _is_bf16_supported_by_device():
    """Check if bf16 is supported based on the current device."""
    device = os.environ.get('DEVICE', 'cuda')
    if device == 'ascend':
        # For Ascend, bf16 support check would be different
        # Placeholder implementation
        return True
    else:
        # For CUDA and default, use the existing check
        return is_bf16_supported()


def set_device_env_variable(worker_id, tp_num: int = 1):
    """Set device environment variable based on the device type."""
    device = os.environ.get('DEVICE', 'cuda')  # Default to cuda

    if device == 'ascend':
        device_id = get_cuda_id_by_workerid(worker_id, tp_num)
        if device_id is not None:
            os.environ['ASCEND_RT_VISIBLE_DEVICES'] = device_id
    else:  # Default to cuda
        cuda_id = get_cuda_id_by_workerid(worker_id, tp_num)
        if cuda_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = cuda_id


def unset_device_env_variable():
    device_type = os.environ.get('DEVICE', 'cuda')
    if device_type == 'ascend':
        if 'ASCEND_RT_VISIBLE_DEVICES' in os.environ:
            del os.environ['ASCEND_RT_VISIBLE_DEVICES']
    else:
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']
