import copy
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union

import yaml

from lmdeploy.utils import is_bf16_supported


def get_parallel_config(config: Dict, model_name: str) -> Dict[str, int]:
    result = {}
    base_model = model_name.split('/')[-1]

    for key in config.keys():
        if key.endswith('_config'):
            if base_model in config.get(key, {}):
                config_value = config[key][base_model]
                if isinstance(config_value, dict):

                    result.update(config_value)
                elif isinstance(config_value, int):

                    config_type = key.replace('_config', '')
                    result[config_type] = config_value

    return result


def _extract_models_from_config(config_value: Any) -> List[str]:

    models = []

    if isinstance(config_value, dict):

        for parallel_type, model_list in config_value.items():
            if isinstance(model_list, list):
                models.extend(model_list)
    elif isinstance(config_value, list):

        models.extend(config_value)

    return models


def get_turbomind_model_list(parallel_config: Optional[Union[int, Dict[str, int]]] = None,
                             model_type: str = 'chat_model',
                             quant_policy: int = None):

    config = get_config()

    key = f'turbomind_{model_type}'
    all_models = []

    if key in config:
        config_value = config[key]
        all_models = _extract_models_from_config(config_value)

    all_models = list(OrderedDict.fromkeys(all_models))

    if parallel_config is not None:
        filtered_models = []

        for model in all_models:
            if is_model_in_list(config, parallel_config, model):
                filtered_models.append(model)

        all_models = filtered_models

    if quant_policy is None:
        case_list = copy.deepcopy(all_models)
    else:
        case_list = [
            x for x in all_models
            if x not in config.get('turbomind_quantization', {}).get(f'no_kvint{quant_policy}', [])
        ]

    quantization_config = config.get('turbomind_quantization', {})
    for key in all_models:
        if key in case_list and key not in quantization_config.get('no_awq', []) and not is_quantization_model(key):
            case_list.append(key + '-inner-4bits')

    for key in quantization_config.get('gptq', []):
        if key in case_list:
            case_list.append(key + '-inner-gptq')

    return case_list


def get_torch_model_list(parallel_config: Optional[Union[int, Dict[str, int]]] = None,
                         model_type: str = 'chat_model',
                         exclude_dup: bool = False,
                         quant_policy: int = None):

    config = get_config()

    key = f'pytorch_{model_type}'
    all_models = []

    if key in config:
        config_value = config[key]
        all_models = _extract_models_from_config(config_value)

    all_models = list(OrderedDict.fromkeys(all_models))

    if parallel_config is not None:
        filtered_models = []

        target_config = {}
        if isinstance(parallel_config, int):
            target_config = {'tp': parallel_config}
        elif isinstance(parallel_config, dict):
            target_config = parallel_config

        for model in all_models:

            model_config = get_parallel_config(config, model)
            if is_model_in_list(config, parallel_config, model):
                filtered_models.append(model)

        all_models = filtered_models

    if exclude_dup:

        turbomind_key = f'turbomind_{model_type}'
        turbomind_models = []
        if turbomind_key in config:
            turbomind_config = config[turbomind_key]
            turbomind_models = _extract_models_from_config(turbomind_config)

        if parallel_config is not None:
            filtered_turbomind = []
            for model in turbomind_models:
                model_config = get_parallel_config(config, model)
                if model_config:
                    match = True
                    for key, target_value in target_config.items():
                        if key not in model_config or model_config[key] != target_value:
                            match = False
                            break
                    if match:
                        filtered_turbomind.append(model)
                elif not target_config or (len(target_config) == 1 and 'tp' in target_config
                                           and target_config['tp'] == 1):

                    filtered_turbomind.append(model)
            turbomind_models = filtered_turbomind

        turbomind_set = set(turbomind_models)

        if quant_policy is None:
            case_list = [x for x in all_models if x in turbomind_set]
        else:
            case_list = [
                x for x in all_models if x in turbomind_set
                and x not in config.get('pytorch_quantization', {}).get(f'no_kvint{quant_policy}', [])
            ]
    else:
        if quant_policy is None:
            case_list = all_models.copy()
        else:
            case_list = [
                x for x in all_models
                if x not in config.get('pytorch_quantization', {}).get(f'no_kvint{quant_policy}', [])
            ]

    quantization_config = config.get('pytorch_quantization', {})
    for key in quantization_config.get('w8a8', []):
        if key in case_list:
            case_list.append(key + '-inner-w8a8')

    for key in quantization_config.get('awq', []):
        if key in case_list:
            case_list.append(key + '-inner-4bits')

    return case_list


def get_all_model_list(parallel_config: Optional[Union[int, Dict[str, int]]] = None,
                       quant_policy: int = None,
                       model_type: str = 'chat_model'):

    case_list = get_turbomind_model_list(parallel_config=parallel_config,
                                         model_type=model_type,
                                         quant_policy=quant_policy)

    if _is_bf16_supported_by_device():
        pytorch_models = get_torch_model_list(parallel_config=parallel_config,
                                              model_type=model_type,
                                              exclude_dup=False,
                                              quant_policy=quant_policy)
        for case in pytorch_models:
            if case not in case_list:
                case_list.append(case)

    return case_list


def get_quantization_model_list(type: str) -> List[str]:
    config = get_config()

    if type == 'awq':

        turbomind_chat_models = []
        if 'turbomind_chat_model' in config:
            turbomind_chat_models = _extract_models_from_config(config['turbomind_chat_model'])

        turbomind_base_models = []
        if 'turbomind_base_model' in config:
            turbomind_base_models = _extract_models_from_config(config['turbomind_base_model'])

        all_models = list(OrderedDict.fromkeys(turbomind_chat_models + turbomind_base_models))

        case_list = [
            x for x in all_models
            if x not in config.get('turbomind_quantization', {}).get('no_awq', []) and not is_quantization_model(x)
        ]

        for key in config.get('pytorch_quantization', {}).get('awq', []):
            if key not in case_list:
                case_list.append(key)

        return case_list

    if type == 'gptq':
        return config.get('turbomind_quantization', {}).get(type, [])

    if type == 'w8a8':
        return config.get('pytorch_quantization', {}).get(type, [])

    return []


def get_vl_model_list(parallel_config: Optional[Union[int, Dict[str, int]]] = None, quant_policy: int = None):
    config = get_config()

    vl_models = config.get('vl_model', [])

    if parallel_config is not None:
        filtered_models = []

        for model in vl_models:
            if is_model_in_list(config, parallel_config, model):
                filtered_models.append(model)

        vl_models = filtered_models

    if quant_policy is None:
        case_list = copy.deepcopy(vl_models)
    else:
        case_list = []
        for model in vl_models:

            in_turbomind = False
            in_pytorch = False

            if 'turbomind_chat_model' in config:
                turbomind_models = _extract_models_from_config(config['turbomind_chat_model'])
                if model in turbomind_models:
                    if model not in config.get('turbomind_quantization', {}).get(f'no_kvint{quant_policy}', []):
                        in_turbomind = True

            if 'pytorch_chat_model' in config:
                pytorch_models = _extract_models_from_config(config['pytorch_chat_model'])
                if model in pytorch_models:
                    if model not in config.get('pytorch_quantization', {}).get(f'no_kvint{quant_policy}', []):
                        in_pytorch = True

            if in_turbomind or in_pytorch:
                case_list.append(model)

    quantization_config = config.get('turbomind_quantization', {})
    pytorch_quantization_config = config.get('pytorch_quantization', {})

    for key in vl_models:

        in_turbomind = False
        if 'turbomind_chat_model' in config:
            turbomind_models = _extract_models_from_config(config['turbomind_chat_model'])
            in_turbomind = key in turbomind_models

        if in_turbomind:
            if (key not in quantization_config.get('no_awq', []) and not is_quantization_model(key)
                    and key + '-inner-4bits' not in case_list):
                if (quant_policy is None or key not in quantization_config.get(f'no_kvint{quant_policy}', [])):
                    case_list.append(key + '-inner-4bits')

        in_pytorch = False
        if 'pytorch_chat_model' in config:
            pytorch_models = _extract_models_from_config(config['pytorch_chat_model'])
            in_pytorch = key in pytorch_models

        if in_pytorch:
            if (key in pytorch_quantization_config.get('awq', []) and not is_quantization_model(key)
                    and key + '-inner-4bits' not in case_list):
                if (quant_policy is None or key not in pytorch_quantization_config.get(f'no_kvint{quant_policy}', [])):
                    case_list.append(key + '-inner-4bits')

    return case_list


def get_evaluate_turbomind_model_list(parallel_config: Optional[Union[int, Dict[str, int]]] = None,
                                      is_longtext: bool = False,
                                      is_mllm: bool = False,
                                      kvint_list: list = []):

    config = get_config()

    if is_longtext:
        case_list_base = [item for item in config.get('longtext_model', [])]
    elif is_mllm:
        case_list_base = config.get('mllm_evaluate_model', [])
    else:
        case_list_base = config.get('evaluate_model', [])

    if parallel_config is not None:
        filtered_models = []

        for model in case_list_base:
            if is_model_in_list(config, parallel_config, model):
                filtered_models.append(model)

        case_list_base = filtered_models

    quantization_config = config.get('turbomind_quantization', {})
    case_list = copy.deepcopy(case_list_base)

    if quantization_config:
        for key in case_list_base:

            in_turbomind = False
            if 'turbomind_chat_model' in config:
                turbomind_models = _extract_models_from_config(config['turbomind_chat_model'])
                in_turbomind = key in turbomind_models

            if in_turbomind:
                if (key not in quantization_config.get('no_awq', []) and not is_quantization_model(key)):
                    case_list.append(key + '-inner-4bits')

    result = []
    if case_list:

        parallel_info = {}
        if parallel_config is not None:
            if isinstance(parallel_config, int):
                parallel_info = {'tp': parallel_config}
            elif isinstance(parallel_config, dict):
                parallel_info = parallel_config

        tp_num = parallel_info.get('tp', 1)
        if tp_num > 1:
            communicators = ['cuda-ipc', 'nccl']
        else:
            communicators = ['cuda-ipc']

        for communicator in communicators:
            for item in case_list:
                base_name = item.replace('-inner-4bits', '')

                in_turbomind_chat = False
                in_turbomind_base = False

                if 'turbomind_chat_model' in config:
                    turbomind_chat_models = _extract_models_from_config(config['turbomind_chat_model'])
                    in_turbomind_chat = base_name in turbomind_chat_models

                if 'turbomind_base_model' in config:
                    turbomind_base_models = _extract_models_from_config(config['turbomind_base_model'])
                    in_turbomind_base = base_name in turbomind_base_models

                if in_turbomind_chat or in_turbomind_base:
                    model_config = {
                        'model': item,
                        'backend': 'turbomind',
                        'communicator': communicator,
                        'quant_policy': 0,
                        'parallel_config': parallel_info.copy() if parallel_info else {},
                        'tp_num': tp_num,
                        'extra': f'--communicator {communicator} '
                    }
                    result.append(model_config)

        if quantization_config:
            for kvint in kvint_list:
                for item in case_list:
                    base_name = item.replace('-inner-4bits', '')

                    in_turbomind = False
                    if 'turbomind_chat_model' in config:
                        turbomind_models = _extract_models_from_config(config['turbomind_chat_model'])
                        in_turbomind = base_name in turbomind_models

                    if in_turbomind:
                        if base_name not in quantization_config.get(f'no_kvint{kvint}', []):
                            model_config = {
                                'model': item,
                                'backend': 'turbomind',
                                'quant_policy': kvint,
                                'parallel_config': parallel_info.copy() if parallel_info else {},
                                'tp_num': tp_num,
                                'extra': ''
                            }
                            result.append(model_config)

    return result


def get_evaluate_pytorch_model_list(parallel_config: Optional[Union[int, Dict[str, int]]] = None,
                                    is_longtext: bool = False,
                                    is_mllm: bool = False,
                                    kvint_list: list = []):

    config = get_config()

    if is_longtext:
        case_list_base = [item for item in config.get('longtext_model', [])]
    elif is_mllm:
        case_list_base = config.get('mllm_evaluate_model', [])
    else:
        case_list_base = config.get('evaluate_model', [])

    if parallel_config is not None:
        filtered_models = []

        for model in case_list_base:
            if is_model_in_list(config, parallel_config, model):
                filtered_models.append(model)

        case_list_base = filtered_models

    pytorch_quantization_config = config.get('pytorch_quantization', {})
    case_list = copy.deepcopy(case_list_base)

    if pytorch_quantization_config:
        for key in case_list_base:

            in_pytorch = False
            if 'pytorch_chat_model' in config:
                pytorch_models = _extract_models_from_config(config['pytorch_chat_model'])
                in_pytorch = key in pytorch_models

            if in_pytorch:
                if (key in pytorch_quantization_config.get('w8a8', []) and not is_quantization_model(key)):
                    case_list.append(key + '-inner-w8a8')

    result = []
    if case_list:

        parallel_info = {}
        if parallel_config is not None:
            if isinstance(parallel_config, int):
                parallel_info = {'tp': parallel_config}
            elif isinstance(parallel_config, dict):
                parallel_info = parallel_config

        for item in case_list:
            base_name = item.replace('-inner-w8a8', '')

            in_pytorch_chat = False
            in_pytorch_base = False

            if 'pytorch_chat_model' in config:
                pytorch_chat_models = _extract_models_from_config(config['pytorch_chat_model'])
                in_pytorch_chat = base_name in pytorch_chat_models

            if 'pytorch_base_model' in config:
                pytorch_base_models = _extract_models_from_config(config['pytorch_base_model'])
                in_pytorch_base = base_name in pytorch_base_models

            if in_pytorch_chat or in_pytorch_base:
                tp_num = parallel_info.get('tp', 1)
                model_config = {
                    'model': item,
                    'backend': 'pytorch',
                    'parallel_config': parallel_info.copy() if parallel_info else {},
                    'tp_num': tp_num,
                    'extra': ''
                }
                result.append(model_config)

    return result


def get_benchmark_model_list(parallel_config: Optional[Union[int, Dict[str, int]]] = None,
                             is_longtext: bool = False,
                             kvint_list: list = []):

    config = get_config()

    if is_longtext:
        case_list_base = [item for item in config.get('longtext_model', [])]
    else:
        case_list_base = config.get('benchmark_model', [])

    if parallel_config is not None:
        filtered_models = []

        for model in case_list_base:

            if is_model_in_list(config, parallel_config, model):
                filtered_models.append(model)

        case_list_base = filtered_models

    quantization_config = config.get('turbomind_quantization', {})
    pytorch_quantization_config = config.get('pytorch_quantization', {})

    case_list = copy.deepcopy(case_list_base)

    for key in case_list_base:

        in_turbomind = False
        if 'turbomind_chat_model' in config:
            turbomind_models = _extract_models_from_config(config['turbomind_chat_model'])
            in_turbomind = key in turbomind_models

        in_pytorch = False
        if 'pytorch_chat_model' in config:
            pytorch_models = _extract_models_from_config(config['pytorch_chat_model'])
            in_pytorch = key in pytorch_models

        if in_turbomind:
            if (key not in quantization_config.get('no_awq', []) and not is_quantization_model(key)):
                case_list.append(key + '-inner-4bits')

        if in_pytorch:
            if (key in pytorch_quantization_config.get('w8a8', []) and not is_quantization_model(key)):
                case_list.append(key + '-inner-w8a8')

    result = []

    parallel_info = {}
    if parallel_config is not None:
        if isinstance(parallel_config, int):
            parallel_info = {'tp': parallel_config}
        elif isinstance(parallel_config, dict):
            parallel_info = parallel_config

    for item in case_list:
        base_name = item.replace('-inner-4bits', '').replace('-inner-w8a8', '')

        in_turbomind = False
        if 'turbomind_chat_model' in config:
            turbomind_models = _extract_models_from_config(config['turbomind_chat_model'])
            in_turbomind = base_name in turbomind_models

        in_pytorch = False
        if 'pytorch_chat_model' in config:
            pytorch_models = _extract_models_from_config(config['pytorch_chat_model'])
            in_pytorch = base_name in pytorch_models

        if in_turbomind:
            tp_num = parallel_info.get('tp', 1)
            result.append({
                'model': item,
                'backend': 'turbomind',
                'quant_policy': 0,
                'parallel_config': parallel_info.copy() if parallel_info else {},
                'tp_num': tp_num
            })

        if '4bits' not in item and in_pytorch:
            tp_num = parallel_info.get('tp', 1)
            result.append({
                'model': item,
                'backend': 'pytorch',
                'parallel_config': parallel_info.copy() if parallel_info else {},
                'tp_num': tp_num
            })

    for kvint in kvint_list:
        for item in case_list:
            base_name = item.replace('-inner-4bits', '').replace('-inner-w8a8', '')
            if base_name not in quantization_config.get(f'no_kvint{kvint}', []):
                in_turbomind = False
                if 'turbomind_chat_model' in config:
                    turbomind_models = _extract_models_from_config(config['turbomind_chat_model'])
                    in_turbomind = base_name in turbomind_models

                if in_turbomind:
                    tp_num = parallel_info.get('tp', 1)
                    result.append({
                        'model': item,
                        'backend': 'turbomind',
                        'quant_policy': kvint,
                        'parallel_config': parallel_info.copy() if parallel_info else {},
                        'tp_num': tp_num
                    })

    return result


def get_communicator_list(parallel_config: Optional[Union[int, Dict[str, int]]] = None):

    tp_num = 1
    if parallel_config is not None:
        if isinstance(parallel_config, int):
            tp_num = parallel_config
        elif isinstance(parallel_config, dict):
            tp_num = parallel_config.get('tp', 1)

    if tp_num > 1 and _is_bf16_supported_by_device():
        return ['cuda-ipc', 'nccl']
    return ['nccl']


def get_config():

    device = os.environ.get('DEVICE', '')
    if device:
        config_path = f'autotest/config-{device}.yaml'
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


def get_cuda_prefix_by_workerid(worker_id, parallel_config: Optional[Union[int, Dict[str, int]]] = None):

    tp_num = 1
    if parallel_config is not None:
        if isinstance(parallel_config, int):
            tp_num = parallel_config
        elif isinstance(parallel_config, dict):
            tp_num = parallel_config.get('tp', 1)

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
        base_id = int(worker_id.replace('gw', ''))
        cuda_num = base_id * tp_num
        return ','.join([str(cuda_num + i) for i in range(tp_num)])


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
        return True
    else:
        return is_bf16_supported()


def set_device_env_variable(worker_id, parallel_config: Optional[Union[int, Dict[str, int]]] = None):
    """Set device environment variable based on the device type."""
    device = os.environ.get('DEVICE', 'cuda')

    tp_num = 1
    if parallel_config is not None:
        if isinstance(parallel_config, int):
            tp_num = parallel_config
        elif isinstance(parallel_config, dict):
            tp_num = parallel_config.get('tp', 1)

    if device == 'ascend':
        device_id = get_cuda_id_by_workerid(worker_id, tp_num)
        if device_id is not None:
            os.environ['ASCEND_RT_VISIBLE_DEVICES'] = device_id
    else:
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


def is_model_in_list(config, parallel_config, model):
    model_config = get_parallel_config(config, model)

    target_config = {}
    if isinstance(parallel_config, int):
        target_config = {'tp': parallel_config}
    elif isinstance(parallel_config, dict):
        target_config = parallel_config

    if not model_config:
        if not target_config or (len(target_config) == 1 and 'tp' in target_config and target_config['tp'] == 1):
            return True

    match = True
    for key, target_value in target_config.items():
        if key not in model_config or model_config[key] != target_value:
            match = False
            break

    return match
