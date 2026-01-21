import copy
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import yaml

from lmdeploy.utils import is_bf16_supported

SUFFIX_INNER_AWQ = '-inner-4bits'
SUFFIX_INNER_GPTQ = '-inner-gptq'
SUFFIX_INNER_W8A8 = '-inner-w8a8'


def get_func_config_list(backend: str,
                         parallel_config: Dict[str, int],
                         model_type: str = 'chat_model',
                         func_type: str = 'func',
                         extra: Optional[Dict[str, Any]] = None) -> List[Dict]:
    """Generate all valid running config combinations (communicator + quant
    policy + model).

    Args:
        backend: Backend type (turbomind/pytorch)
        parallel_config: Parallel config for tensor parallel
        model_type: Model type, default: chat_model
        func_type: Test func type filter, default: func
        extra: extra config to update in each run config dict
    Returns:
        List[Dict]: All valid run config dicts
    """
    config = get_config()
    device = config.get('device', 'cuda')
    base_case_list = get_model_list(config, backend, parallel_config, model_type, func_type)

    if extra is None:
        extra = {}

    run_configs = []
    dtype = 'float16' if not is_bf16_supported(device) else None

    for communicator in _get_communicator_list(config, backend, parallel_config):
        for model in base_case_list:
            for quant_policy in [0, 4, 8]:
                # temp remove testcase because of issue 3434
                if 'turbomind' == backend and communicator == 'cuda-ipc' and parallel_config.get(
                        'tp', 1) > 1 and ('InternVL3' in model or 'InternVL2_5' in model or 'MiniCPM-V-2_6' in model
                                          or 'InternVL2-Llama3' in model):  # noqa
                    continue
                if 'turbomind' == backend and parallel_config.get(
                        'tp', 1
                ) > 1 and model_type == 'vl_model' and func_type == 'mllm_evaluate':  # mllm eval with bug when tp > 2
                    continue
                # [TM][FATAL] models/llama/LlamaBatch.cc(362): Check failed: r->session.start_flag Mrope doesn't support interactive chat # noqa
                if ('Qwen2.5-VL' in model or 'Qwen2-VL' in model) and 'turbomind' == backend:
                    continue
                # AssertionError: prompts should be a list
                if 'phi' in model.lower() and model_type == 'vl_model':
                    continue
                if not _is_kvint_model(config, backend, model, quant_policy):
                    continue
                run_config = {
                    'model': model,
                    'backend': backend,
                    'communicator': communicator,
                    'quant_policy': quant_policy,
                    'parallel_config': parallel_config,
                    'extra_params': copy.copy(extra)
                }
                if dtype and backend == 'pytorch':
                    run_config['extra_params']['dtype'] = dtype
                if device != 'cuda':
                    run_config['extra_params']['device'] = device
                run_configs.append(run_config)

    for run_config in run_configs:
        if 'Qwen3-235B-A22B-Thinking-2507' in run_config['model']:
            run_config['extra_params']['cache-max-entry-count'] = 0.9
            run_config['extra_params']['max-batch-size'] = 1024

        if config.get('env_tag', '') in ['3090', '5080']:
            run_config['extra_params']['cache-max-entry-count'] = 0.5

        if 'sdar' in run_config['model'].lower():
            run_config['extra_params']['dllm-block-length'] = 4
            run_config['extra_params']['dllm-denoising-steps'] = 4
            run_config['extra_params']['dllm-confidence-threshold'] = 0.9

    return run_configs


def get_cli_common_param(run_config: Dict[str, Any]) -> str:
    """Generate cli common params string by run config dict."""
    backend = run_config.get('backend')
    model = run_config.get('model')
    communicator = run_config.get('communicator')
    quant_policy = run_config.get('quant_policy')
    extra_params = run_config.get('extra_params', {})
    parallel_config = run_config.get('parallel_config', {})

    cli_params = [f'--backend {backend}', f'--communicator {communicator}']
    # Optional params
    if quant_policy != 0:
        cli_params.append(f'--quant-policy {quant_policy}')

    # quant format
    model_lower = model.lower()
    if 'w4' in model_lower or '4bits' in model_lower or 'awq' in model_lower:
        cli_params.append('--model-format awq')
    if 'gptq' in model_lower:
        cli_params.append('--model-format gptq')

    # Parallel config
    for para_key in ('dp', 'ep', 'cp'):
        if para_key in parallel_config and parallel_config[para_key] > 1:
            cli_params.append(f'--{para_key} {parallel_config[para_key]}')
    if 'tp' in parallel_config and parallel_config['tp'] > 1:
        tp_num = parallel_config['tp']
        cli_params.append(f'--tp {tp_num}')  # noqa

    # Extra params
    for key, value in extra_params.items():
        cli_params.append(f'--{key} {value}' if value else f'--{key}')

    return ' '.join(cli_params).replace('_', '-')


def get_cli_str(config: Dict[str, Any]) -> str:
    cli_str = []
    # Extra params
    for key, value in config.items():
        key = key.replace('_', '-')
        cli_str.append(f'--{key} {value}' if value else f'--{key}')

    return ' '.join(cli_str)


def get_parallel_config(config: Dict, model_name: str) -> Dict[str, int]:
    """Get matched parallel config dict by model name, default tp:1 if no
    match."""
    result = {}
    base_model = _base_model_name(model_name)
    parallel_configs = config.get('config', {})

    for conf_key, model_map in parallel_configs.items():
        if model_map is None:
            continue
        if base_model in model_map:
            conf_value = model_map[base_model]
            if isinstance(conf_value, dict):
                result.update(conf_value)
            elif isinstance(conf_value, int):
                result[conf_key] = conf_value

    return result if result else {'tp': 1}


def _extract_models_from_config(config_value: Any) -> List[str]:
    """Extract flat model name list from config value (dict/list supported)"""
    models = []
    if isinstance(config_value, Dict):
        for model_list in config_value.values():
            if isinstance(model_list, List):
                models.extend([m for m in model_list if isinstance(m, str)])
    elif isinstance(config_value, List):
        models.extend([m for m in config_value if isinstance(m, str)])
    return models


def get_model_list(config: Dict,
                   backend: str,
                   parallel_config: Dict[str, int] = None,
                   model_type: str = 'chat_model',
                   func_type: str = 'func') -> List[str]:
    """Get filtered model list with quantization extended models by
    backend/parallel config/model type/func type.

    Args:
        config: Global system config dict
        backend: Backend type (turbomind/pytorch)
        parallel_config: Parallel filter config
        model_type: Model type, default: chat_model
        func_type: Test func type filter, default: func
    Returns:
        List[str]: Base models + quantization extended models
    """
    model_config_key = f'{backend}_{model_type}'
    all_models = []

    if model_config_key in config:
        all_models = _extract_models_from_config(config[model_config_key])

    all_models = _filter_by_test_func_type(config, all_models, func_type)
    all_models = list(OrderedDict.fromkeys(all_models))  # Deduplicate, keep order
    all_models = [model for model in all_models if is_model_in_list(config, parallel_config, model)]

    extended_models = list(all_models)
    quantization_config = config.get(f'{backend}_quantization', {})

    # Append quantization models by backend
    if backend == 'turbomind':
        _extend_turbomind_quant_models(quantization_config, all_models, extended_models)
    elif backend == 'pytorch':
        _extend_pytorch_quant_models(quantization_config, all_models, extended_models)

    return extended_models


def _filter_by_test_func_type(config: Dict, model_list: List[str], func_type: str) -> List[str]:
    """Filter model list by test function type, return intersection of two
    model sets."""
    if func_type == 'func':
        return model_list

    filtered_models = []
    model_config_key = f'{func_type}_model'
    if model_config_key in config:
        filtered_models = _extract_models_from_config(config[model_config_key])

    return list(set(filtered_models) & set(model_list))


def _extend_turbomind_quant_models(quant_config: dict, base_models: list, target_list: list) -> None:
    """Append turbomind quantization models to target list (AWQ 4bits +
    GPTQ)"""
    no_awq_models = quant_config.get('no_awq', [])
    # Append AWQ 4bits quantization models
    for model_name in base_models:
        if model_name in target_list and model_name not in no_awq_models and not is_quantization_model(model_name):
            target_list.append(model_name + SUFFIX_INNER_AWQ)
    # Append GPTQ quantization models
    for model_name in quant_config.get('gptq', []):
        if model_name in target_list:
            target_list.append(model_name + SUFFIX_INNER_GPTQ)


def _extend_pytorch_quant_models(quant_config: dict, base_models: list, target_list: list) -> None:
    """Append pytorch quantization models to target list (AWQ 4bits + W8A8)"""
    # Append AWQ quantization models
    for model_name in quant_config.get('awq', []):
        if model_name in target_list:
            target_list.append(model_name + SUFFIX_INNER_AWQ)
    # Append W8A8 quantization models
    for model_name in quant_config.get('w8a8', []):
        if model_name in target_list:
            target_list.append(model_name + SUFFIX_INNER_W8A8)


def _is_kvint_model(config: Dict, backend: str, model: str, quant_policy: int) -> bool:
    """Check if model supports the kv quantization policy, quant_policy=0
    always return True."""
    if quant_policy == 0:
        return True
    no_kvint_black_list = config.get(f'{backend}_quantization', {}).get(f'no_kvint{quant_policy}', [])

    return _base_model_name(model) not in no_kvint_black_list


def _base_model_name(model: str) -> str:
    """Simplify model name by removing quantization suffix for config
    matching."""
    return model.replace('-inner-4bits', '').replace('-inner-w8a8', '').replace('-inner-gptq', '')


def get_quantization_model_list(type: str) -> List[str]:
    """Get quantization model list by specified quant type(awq/gptq/w8a8)"""
    config = get_config()
    quant_model_list = []

    if type == 'awq':
        # Get all turbomind chat/base models & deduplicate
        turbo_chat = _extract_models_from_config(
            config['turbomind_chat_model']) if 'turbomind_chat_model' in config else []
        turbo_base = _extract_models_from_config(
            config['turbomind_base_model']) if 'turbomind_base_model' in config else []
        all_turbo_models = list(OrderedDict.fromkeys(turbo_chat + turbo_base))

        # Filter turbomind valid awq models
        no_awq = config.get('turbomind_quantization', {}).get('no_awq', [])
        quant_model_list = [m for m in all_turbo_models if m not in no_awq and not is_quantization_model(m)]

        # Append pytorch awq models
        torch_awq = config.get('pytorch_quantization', {}).get('awq', [])
        for model in torch_awq:
            if model not in quant_model_list:
                quant_model_list.append(model)

    elif type == 'gptq':
        quant_model_list = config.get('turbomind_quantization', {}).get(type, [])

    elif type == 'w8a8':
        quant_model_list = config.get('pytorch_quantization', {}).get(type, [])

    return quant_model_list


def get_config() -> Dict[str, Any]:
    """Load & get yaml config file, auto adapt device env & update log path."""
    # Get device env & match config file path
    env_tag = os.environ.get('TEST_ENV')
    config_path = f'autotest/config-{env_tag}.yaml' if env_tag else 'autotest/config.yaml'

    # Fallback to default config if device-specific config not exist
    if env_tag and not os.path.exists(config_path):
        config_path = 'autotest/config.yaml'

    # Load yaml config file safely
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.SafeLoader)

    # Deep copy config to avoid modify raw data, update log path with github run id
    config_copy = copy.deepcopy(config)
    run_id = os.environ.get('RUN_ID', 'local_run')
    config_copy['log_path'] = os.path.join(config_copy['log_path'], str(run_id).replace('/', '_'))
    config_copy['eval_path'] = os.path.join(config_copy['eval_path'], str(run_id).replace('/', '_'))
    config_copy['mllm_eval_path'] = os.path.join(config_copy['mllm_eval_path'], str(run_id).replace('/', '_'))
    config_copy['benchmark_path'] = os.path.join(config_copy['benchmark_path'], str(run_id).replace('/', '_'))
    config_copy['server_log_path'] = os.path.join(config_copy['server_log_path'], str(run_id).replace('/', '_'))
    os.makedirs(config_copy['log_path'], exist_ok=True)
    os.makedirs(config_copy['eval_path'], exist_ok=True)
    os.makedirs(config_copy['mllm_eval_path'], exist_ok=True)
    os.makedirs(config_copy['benchmark_path'], exist_ok=True)
    os.makedirs(config_copy['server_log_path'], exist_ok=True)

    return config_copy


def get_cuda_prefix_by_workerid(worker_id: Optional[str], parallel_config: Dict[str, int] = None) -> Optional[str]:
    """Get cuda/ascend visible devices env prefix by worker id & parallel
    config."""
    para_conf = parallel_config or {}
    device_type = para_conf.get('device', 'cuda')

    tp_num = para_conf.get('tp')
    if not tp_num:
        return ''

    cuda_id = get_cuda_id_by_workerid(worker_id, tp_num)
    if not cuda_id:
        return ''

    return f'ASCEND_RT_VISIBLE_DEVICES={cuda_id}' if device_type == 'ascend' else f'CUDA_VISIBLE_DEVICES={cuda_id}'


def get_cuda_id_by_workerid(worker_id: Optional[str], tp_num: int = 1) -> Optional[str]:
    """Get cuda id str by worker id and tp num, return None if invalid worker
    id."""
    if worker_id is None or 'gw' not in worker_id:
        return None

    base_id = int(worker_id.replace('gw', ''))
    cuda_num = base_id * tp_num
    return ','.join([str(cuda_num + i) for i in range(tp_num)])


def get_workerid(worker_id: Optional[str]) -> int:
    """Parse numeric worker id from worker id str, return 0 if invalid worker
    id."""
    if worker_id is None or 'gw' not in worker_id:
        return 0

    return int(worker_id.replace('gw', ''))


def is_quantization_model(model: str) -> bool:
    """Check if model name contains quantization related keywords."""
    lower_name = model.lower()
    return any(key in lower_name for key in ('awq', '4bits', 'w4', 'int4'))


def _get_communicator_list(config: Dict, backend: str, parallel_config: Dict[str, int] = None) -> List[str]:
    """Get available communicator list by device and parallel config."""
    device = config.get('device', None)

    if device == 'ascend':
        return ['hccl']
    if backend == 'pytorch':
        return ['nccl']
    if ('cp' in parallel_config or 'dp' in parallel_config or 'ep' in parallel_config):
        return ['nccl']
    if 'tp' in parallel_config and parallel_config['tp'] == 1:
        return ['nccl']

    return ['nccl', 'cuda-ipc']


def set_device_env_variable(worker_id, parallel_config: Dict[str, int] = None):
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


def is_model_in_list(config: Dict, parallel_config: Dict[str, int], model: str) -> bool:
    """Check if model matches the target parallel config."""
    model_config = get_parallel_config(config, model)
    return model_config == parallel_config


def get_case_str_by_config(run_config: Dict[str, Any], is_simple: bool = True) -> str:
    """Generate case name string by run config dict."""
    model_name = run_config['model']
    backend_type = run_config['backend']
    communicator = run_config.get('communicator', 'nccl')
    quant_policy = run_config.get('quant_policy', 0)
    parallel_config = run_config.get('parallel_config', {'tp': 1})
    extra_params = run_config.get('extra_params', {})

    # Sorted parallel config to fixed string format
    sorted_items = sorted(parallel_config.items())
    parallel_str = '_'.join(f'{k}{v}' for k, v in sorted_items)
    # Get last section of model name, compatible with model name contains '/'
    pure_model_name = model_name.split('/')[-1].replace('_', '-')
    extra_params_case = ''
    if not is_simple:
        for k, v in extra_params.items():
            if len(v) > 10:
                extra_params_case += f'_{k}'.replace('_', '-').replace('/', '-').replace('.', '-')
            else:
                extra_params_case += f'_{k}{v}'.replace('_', '-').replace('/', '-').replace('.', '-')

    return f'{backend_type}_{pure_model_name}_{communicator}_{parallel_str}_{quant_policy}{extra_params_case}'


def parse_config_by_case(case_str: str) -> Dict[str, Any]:
    """Parse run config dict from case name string (fix split & type convert
    bug)"""
    case_parts = case_str.split('_')
    # Parse fixed field & reassemble dynamic parallel config
    backend = case_parts[0]
    model = case_parts[1]
    communicator = case_parts[2]
    quant_policy = int(case_parts[-1])
    parallel_parts = case_parts[3:-1]

    # Convert parallel str to dict, e.g: ['tp1','pp2'] -> {'tp':1, 'pp':2}
    parallel_config = {}
    for part in parallel_parts:
        for idx, char in enumerate(part):
            if char.isdigit():
                k = part[:idx]
                v = int(part[idx:])
                parallel_config[k] = v
                break

    return {
        'backend': backend,
        'model': model,
        'communicator': communicator,
        'parallel_config': parallel_config,
        'quant_policy': quant_policy
    }


def test_config():
    os.environ['DEVICE'] = 'test'
    config = get_config()
    assert 'model_path' in config.keys()
    assert 'resource_path' in config.keys()
    assert 'log_path' in config.keys()
    assert 'server_log_path' in config.keys()
    assert 'eval_path' in config.keys()
    assert 'mllm_eval_path' in config.keys()
    assert 'benchmark_path' in config.keys()
    assert 'dataset_path' in config.keys()
    assert 'prefix_dataset_path' in config.keys()
    assert 'env_tag' in config.keys()
    assert 'config' in config.keys()
    assert 'tp' in config.get('config')

    assert is_model_in_list(config, parallel_config={'tp': 1}, model='test/test_tp1')
    assert is_model_in_list(config, parallel_config={'tp': 2}, model='test/test_tp1') is False
    assert is_model_in_list(config, parallel_config={'ep': 1},
                            model='test/test_tp1') is False, is_model_in_list(config,
                                                                              parallel_config={'ep': 1},
                                                                              model='test/test_tp1')
    assert is_model_in_list(config, parallel_config={'tp': 2}, model='test/test_tp2-inner-4bits')
    assert is_model_in_list(config, parallel_config={'tp': 2}, model='test/test_tp2-inner-w8a8')
    assert is_model_in_list(config, parallel_config={'tp': 8}, model='test/test_tp8-inner-gptq')
    assert is_model_in_list(config, parallel_config={'tp': 8}, model='test/test_cp2tp8') is False
    assert is_model_in_list(config, parallel_config={'tp': 8, 'cp': 2}, model='test/test_cp2tp8')
    assert is_model_in_list(config, parallel_config={'cp': 2, 'tp': 8}, model='test/test_cp2tp8')
    assert is_model_in_list(config, parallel_config={'cp': 4, 'tp': 8}, model='test/test_cp2tp8') is False
    assert is_model_in_list(config, parallel_config={'dp': 8, 'ep': 8}, model='test/test_dpep8')
    assert is_model_in_list(config, parallel_config={'dp': 4, 'ep': 8}, model='test/test_dpep8') is False
    assert is_model_in_list(config, parallel_config={'ep': 4, 'dp': 8}, model='test/test_dpep8') is False

    assert _is_kvint_model(config, 'turbomind', 'test/test_tp1-inner-4bits', 8) is False
    assert _is_kvint_model(config, 'turbomind', 'test/test_tp1-inner-4bits', 4)
    assert _is_kvint_model(config, 'turbomind', 'any', 0)
    assert _is_kvint_model(config, 'pytorch', 'test/test_tp1-inner-gptq', 8) is False
    assert _is_kvint_model(config, 'pytorch', 'test/test_tp1-inner-gptq', 4)
    assert _is_kvint_model(config, 'pytorch', 'test/test_vl_tp1-inner-gptq', 8) is False
    assert _is_kvint_model(config, 'pytorch', 'test/test_cp2tp8-inner-w8a8', 4) is False
    os.unsetenv('DEVICE')


def test_get_case_str_by_config():
    run_config = {
        'model': 'test/test_dpep16',
        'backend': 'turbomind',
        'communicator': 'nccl',
        'quant_policy': 8,
        'parallel_config': {
            'dp': 16,
            'ep': 16
        }
    }
    case_str = get_case_str_by_config(run_config)
    assert case_str == 'turbomind_test-dpep16_nccl_dp16_ep16_8', case_str
    run_config_parsed = parse_config_by_case(case_str)
    assert run_config_parsed['model'] == 'test-dpep16'
    assert run_config_parsed['backend'] == 'turbomind'
    assert run_config_parsed['communicator'] == 'nccl'
    assert run_config_parsed['quant_policy'] == 8
    assert run_config_parsed['parallel_config']['dp'] == 16
    assert run_config_parsed['parallel_config']['ep'] == 16


def test_cli_common_param():
    run_config = {
        'model': 'test/test_dpep16-inner-4bits',
        'backend': 'turbomind',
        'communicator': 'nccl',
        'quant_policy': 8,
        'parallel_config': {
            'dp': 16,
            'ep': 16
        },
        'extra_params': {
            'dtype': 'bfloat16',
            'device': 'ascend',
            'enable_prefix_caching': None,
            'max_batch_size': 2048,
            'session_len': 8192,
            'cache_max_entry_count': 0.75
        }
    }

    cli_params = get_cli_common_param(run_config)
    assert cli_params == '--backend turbomind --communicator nccl --quant-policy 8 --model-format awq --dp 16 --ep 16 --dtype bfloat16 --device ascend --enable-prefix-caching --max-batch-size 2048 --session-len 8192 --cache-max-entry-count 0.75', cli_params  # noqa
    run_config = {
        'model': 'test/test_dpep16-inner-4bits',
        'backend': 'pytorch',
        'communicator': 'hccl',
        'quant_policy': 0,
        'parallel_config': {
            'tp': 8
        }
    }

    cli_params = get_cli_common_param(run_config)
    assert cli_params == '--backend pytorch --communicator hccl --model-format awq --tp 8', cli_params
    os.unsetenv('TEST_ENV')


def test_return_info_turbomind():
    os.environ['TEST_ENV'] = 'test'
    backend = 'turbomind'
    func_chat_tp1 = get_func_config_list(backend, parallel_config={'tp': 1}, model_type='chat_model', func_type='func')
    assert len(func_chat_tp1) == 12, len(func_chat_tp1)
    func_chat_tp2 = get_func_config_list(backend, parallel_config={'tp': 2}, model_type='chat_model', func_type='func')
    assert len(func_chat_tp2) == 32, len(func_chat_tp2)
    func_chat_tp8 = get_func_config_list(backend, parallel_config={'tp': 8}, model_type='chat_model', func_type='func')
    assert len(func_chat_tp8) == 24, len(func_chat_tp8)
    func_chat_cptp = get_func_config_list(backend,
                                          parallel_config={
                                              'cp': 2,
                                              'tp': 8
                                          },
                                          model_type='chat_model',
                                          func_type='func')
    assert len(func_chat_cptp) == 8, len(func_chat_cptp)
    func_chat_dpep8 = get_func_config_list(backend,
                                           parallel_config={
                                               'dp': 8,
                                               'ep': 8
                                           },
                                           model_type='chat_model',
                                           func_type='func')
    assert len(func_chat_dpep8) == 0, len(func_chat_dpep8)
    func_chat_dpep16 = get_func_config_list(backend,
                                            parallel_config={
                                                'dp': 16,
                                                'ep': 16
                                            },
                                            model_type='chat_model',
                                            func_type='func')
    assert len(func_chat_dpep16) == 0, len(func_chat_dpep16)
    func_base_tp1 = get_func_config_list(backend, parallel_config={'tp': 1}, model_type='base_model', func_type='func')
    assert len(func_base_tp1) == 6, len(func_base_tp1)
    func_base_tp2 = get_func_config_list(backend, parallel_config={'tp': 2}, model_type='base_model', func_type='func')
    assert len(func_base_tp2) == 4, len(func_base_tp2)

    evaluate_tp1 = get_func_config_list(backend,
                                        parallel_config={'tp': 1},
                                        model_type='chat_model',
                                        func_type='evaluate')
    assert len(evaluate_tp1) == 6, len(evaluate_tp1)
    benchmark_tp2 = get_func_config_list(backend,
                                         parallel_config={'tp': 2},
                                         model_type='chat_model',
                                         func_type='benchmark')
    assert len(benchmark_tp2) == 4, len(benchmark_tp2)
    longtext_tp8 = get_func_config_list(backend,
                                        parallel_config={'tp': 8},
                                        model_type='chat_model',
                                        func_type='longtext')
    assert len(longtext_tp8) == 12, len(longtext_tp8)
    evaluate_cptp = get_func_config_list(backend,
                                         parallel_config={
                                             'cp': 2,
                                             'tp': 8
                                         },
                                         model_type='chat_model',
                                         func_type='evaluate')
    assert len(evaluate_cptp) == 4, len(evaluate_cptp)
    benchmark_dpep8 = get_func_config_list(backend,
                                           parallel_config={
                                               'dp': 8,
                                               'ep': 8
                                           },
                                           model_type='chat_model',
                                           func_type='benchmark')
    assert len(benchmark_dpep8) == 0, len(benchmark_dpep8)

    mllm_benchmark_tp1 = get_func_config_list(backend,
                                              parallel_config={'tp': 1},
                                              model_type='chat_model',
                                              func_type='mllm_benchmark')
    assert len(mllm_benchmark_tp1) == 6, len(mllm_benchmark_tp1)
    mllm_longtext_tp2 = get_func_config_list(backend,
                                             parallel_config={'tp': 2},
                                             model_type='chat_model',
                                             func_type='mllm_longtext')
    assert len(mllm_longtext_tp2) == 0, len(mllm_longtext_tp2)
    mllm_evaluate_tp8 = get_func_config_list(backend,
                                             parallel_config={'tp': 8},
                                             model_type='chat_model',
                                             func_type='mllm_evaluate')
    assert len(mllm_evaluate_tp8) == 12, len(mllm_evaluate_tp8)
    mllm_evaluate_dpep16 = get_func_config_list(backend,
                                                parallel_config={
                                                    'dp': 16,
                                                    'ep': 16
                                                },
                                                model_type='chat_model',
                                                func_type='evaluate')
    assert len(mllm_evaluate_dpep16) == 0, len(mllm_evaluate_dpep16)
    mllm_benchmark_cptp = get_func_config_list(backend,
                                               parallel_config={
                                                   'cp': 2,
                                                   'tp': 8
                                               },
                                               model_type='chat_model',
                                               func_type='benchmark')
    assert len(mllm_benchmark_cptp) == 4, len(mllm_benchmark_cptp)
    os.unsetenv('TEST_ENV')


def test_return_info_pytorch():
    os.environ['TEST_ENV'] = 'test'
    backend = 'pytorch'
    func_chat_tp1 = get_func_config_list(backend, parallel_config={'tp': 1}, model_type='chat_model', func_type='func')
    assert len(func_chat_tp1) == 12, len(func_chat_tp1)
    func_chat_tp2 = get_func_config_list(backend, parallel_config={'tp': 2}, model_type='chat_model', func_type='func')
    assert len(func_chat_tp2) == 19, len(func_chat_tp2)
    func_chat_tp8 = get_func_config_list(backend, parallel_config={'tp': 8}, model_type='chat_model', func_type='func')
    assert len(func_chat_tp8) == 6, len(func_chat_tp8)
    func_chat_cptp = get_func_config_list(backend,
                                          parallel_config={
                                              'cp': 2,
                                              'tp': 8
                                          },
                                          model_type='chat_model',
                                          func_type='func')
    assert len(func_chat_cptp) == 4, len(func_chat_cptp)
    func_chat_dpep8 = get_func_config_list(backend,
                                           parallel_config={
                                               'dp': 8,
                                               'ep': 8
                                           },
                                           model_type='chat_model',
                                           func_type='func')
    assert len(func_chat_dpep8) == 5, len(func_chat_dpep8)
    func_chat_dpep16 = get_func_config_list(backend,
                                            parallel_config={
                                                'dp': 16,
                                                'ep': 16
                                            },
                                            model_type='chat_model',
                                            func_type='func')
    assert len(func_chat_dpep16) == 6, len(func_chat_dpep16)
    func_base_tp1 = get_func_config_list(backend, parallel_config={'tp': 1}, model_type='base_model', func_type='func')
    assert len(func_base_tp1) == 7, len(func_base_tp1)
    func_base_tp2 = get_func_config_list(backend, parallel_config={'tp': 2}, model_type='base_model', func_type='func')
    assert len(func_base_tp2) == 4, len(func_base_tp2)

    evaluate_tp1 = get_func_config_list(backend,
                                        parallel_config={'tp': 1},
                                        model_type='chat_model',
                                        func_type='evaluate')
    assert len(evaluate_tp1) == 7, len(evaluate_tp1)
    benchmark_tp2 = get_func_config_list(backend,
                                         parallel_config={'tp': 2},
                                         model_type='chat_model',
                                         func_type='benchmark')
    assert len(benchmark_tp2) == 3, len(benchmark_tp2)
    longtext_tp8 = get_func_config_list(backend,
                                        parallel_config={'tp': 8},
                                        model_type='chat_model',
                                        func_type='longtext')
    assert len(longtext_tp8) == 3, len(longtext_tp8)
    evaluate_cptp = get_func_config_list(backend,
                                         parallel_config={
                                             'cp': 2,
                                             'tp': 8
                                         },
                                         model_type='chat_model',
                                         func_type='evaluate')
    assert len(evaluate_cptp) == 2, len(evaluate_cptp)
    benchmark_dpep8 = get_func_config_list(backend,
                                           parallel_config={
                                               'dp': 8,
                                               'ep': 8
                                           },
                                           model_type='chat_model',
                                           func_type='benchmark')
    assert len(benchmark_dpep8) == 2, len(benchmark_dpep8)

    mllm_benchmark_tp1 = get_func_config_list(backend,
                                              parallel_config={'tp': 1},
                                              model_type='chat_model',
                                              func_type='mllm_benchmark')
    assert len(mllm_benchmark_tp1) == 5, len(mllm_benchmark_tp1)
    mllm_longtext_tp2 = get_func_config_list(backend,
                                             parallel_config={'tp': 2},
                                             model_type='chat_model',
                                             func_type='mllm_longtext')
    assert len(mllm_longtext_tp2) == 0, len(mllm_longtext_tp2)
    mllm_evaluate_tp8 = get_func_config_list(backend,
                                             parallel_config={'tp': 8},
                                             model_type='chat_model',
                                             func_type='mllm_evaluate')
    assert len(mllm_evaluate_tp8) == 3, len(mllm_evaluate_tp8)
    mllm_evaluate_dpep16 = get_func_config_list(backend,
                                                parallel_config={
                                                    'dp': 16,
                                                    'ep': 16
                                                },
                                                model_type='chat_model',
                                                func_type='evaluate')
    assert len(mllm_evaluate_dpep16) == 3, len(mllm_evaluate_dpep16)
    mllm_benchmark_cptp = get_func_config_list(backend,
                                               parallel_config={
                                                   'cp': 2,
                                                   'tp': 8
                                               },
                                               model_type='chat_model',
                                               func_type='benchmark')
    assert len(mllm_benchmark_cptp) == 2, len(mllm_benchmark_cptp)
    os.unsetenv('TEST_ENV')


def test_run_config():
    os.environ['TEST_ENV'] = 'test'
    backend = 'turbomind'
    run_config1 = get_func_config_list(backend, parallel_config={'tp': 1}, model_type='chat_model', func_type='func')[0]
    assert run_config1['model'] == 'test/test_tp1'
    assert run_config1['backend'] == 'turbomind'
    assert run_config1['communicator'] == 'nccl'
    assert run_config1['quant_policy'] == 0
    assert run_config1['parallel_config'] == {'tp': 1}
    os.environ['TEST_ENV'] = 'testascend'
    backend = 'pytorch'
    run_config2 = get_func_config_list(backend, parallel_config={'tp': 1}, model_type='chat_model', func_type='func')[0]
    assert run_config2['model'] == 'test/test_tp1'
    assert run_config2['backend'] == 'pytorch'
    assert run_config2['communicator'] == 'hccl'
    assert run_config2['quant_policy'] == 0
    assert run_config2['parallel_config'] == {'tp': 1}
    run_config3 = get_func_config_list(backend,
                                       parallel_config={'tp': 1},
                                       model_type='chat_model',
                                       func_type='func',
                                       extra={
                                           'speculative_algorithm': 'eagle',
                                           'session_len': 1024
                                       })[0]
    assert run_config3['model'] == 'test/test_tp1'
    assert run_config3['backend'] == 'pytorch'
    assert run_config3['communicator'] == 'hccl'
    assert run_config3['quant_policy'] == 0
    assert run_config3['parallel_config'] == {'tp': 1}
    assert run_config3['extra_params']['speculative_algorithm'] == 'eagle'
    assert run_config3['extra_params']['session_len'] == 1024
    os.unsetenv('TEST_ENV')


if __name__ == '__main__':
    test_cli_common_param()
    test_run_config()
    test_get_case_str_by_config()
    test_return_info_pytorch()
    test_config()
    test_return_info_turbomind()
