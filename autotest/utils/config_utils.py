import copy
import os
from collections import OrderedDict
from typing import Any

import yaml

from lmdeploy.utils import is_bf16_supported

SUFFIX_INNER_AWQ = '-inner-4bits'
SUFFIX_INNER_GPTQ = '-inner-gptq'
SUFFIX_INNER_W8A8 = '-inner-w8a8'


def resolve_extra_params(extra_params: dict[str, Any], model_base_path: str) -> None:
    """Resolve relative model paths in extra_params to absolute paths.

    Centralised helper so that every call-site does not need its own
    ``if key in extra_params …`` guard – adding a new key here is enough.
    """
    # Keys in extra_params whose string values are relative model paths
    model_path_keys = ['speculative-draft-model']

    # Flat string-valued keys
    for key in model_path_keys:
        if key in extra_params:
            value = extra_params[key]
            if value and isinstance(value, str) and not os.path.isabs(value):
                extra_params[key] = os.path.join(model_base_path, value)

    # Nested speculative_config (pipeline usage)
    spec_cfg = extra_params.get('speculative_config')
    if isinstance(spec_cfg, dict) and 'model' in spec_cfg:
        model = spec_cfg['model']
        if model and isinstance(model, str) and not os.path.isabs(model):
            spec_cfg['model'] = os.path.join(model_base_path, model)


_MODEL_RUN_PARAMS_PATH = os.path.join(os.path.dirname(__file__), 'model_run_params.yml')
_model_run_params_rules: list[dict[str, Any]] | None = None


def _load_model_run_params_rules() -> list[dict[str, Any]]:
    global _model_run_params_rules
    if _model_run_params_rules is None:
        with open(_MODEL_RUN_PARAMS_PATH, encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        _model_run_params_rules = data.get('rules', [])
    return _model_run_params_rules


def _parallel_rule_matches(para_match: dict[str, Any], parallel_config: dict[str, Any]) -> bool:
    for key in ('dp', 'ep', 'tp'):
        if key in para_match and parallel_config.get(key, 0) != para_match[key]:
            return False
    return True


def _model_subcondition_matches(cond: dict[str, Any], model: str) -> bool:
    if 'model_contains' in cond:
        return cond['model_contains'] in model
    if 'model_contains_ignore_case' in cond:
        return cond['model_contains_ignore_case'].lower() in model.lower()
    if 'model_equals' in cond:
        return model == cond['model_equals']
    if 'model_not_contains' in cond:
        return cond['model_not_contains'] not in model
    return False


def _rule_match_matches(match: dict[str, Any], *, config: dict[str, Any], extra: dict[str, Any],
                        func_type: str, backend: str, run_config: dict[str, Any]) -> bool:
    if not match:
        return True
    model = run_config['model']
    if 'model_contains' in match and match['model_contains'] not in model:
        return False
    if 'model_contains_ignore_case' in match:
        if match['model_contains_ignore_case'].lower() not in model.lower():
            return False
    if 'model_equals' in match and model != match['model_equals']:
        return False
    if 'model_not_contains' in match and match['model_not_contains'] in model:
        return False
    if 'model_any' in match:
        if not any(_model_subcondition_matches(c, model) for c in match['model_any']):
            return False
    if 'func_type' in match and func_type != match['func_type']:
        return False
    if 'func_type_in' in match and func_type not in match['func_type_in']:
        return False
    if 'backend' in match and backend != match['backend']:
        return False
    if 'env_tag_in' in match:
        env_tag = str(config.get('env_tag', ''))
        if env_tag not in {str(tag) for tag in match['env_tag_in']}:
            return False
    if 'extra_missing_keys' in match:
        for key in match['extra_missing_keys']:
            if key in extra:
                return False
    return True


def apply_model_run_params(run_config: dict[str, Any], config: dict[str, Any], extra: dict[str, Any],
                           func_type: str, backend: str) -> None:
    """Apply ordered rules from model_run_params.yml to run_config
    extra_params."""
    parallel_config = run_config.get('parallel_config', {})
    for rule in _load_model_run_params_rules():
        if not _rule_match_matches(rule.get('match', {}), config=config, extra=extra, func_type=func_type,
                                   backend=backend, run_config=run_config):
            continue
        for model_rule in rule.get('model_rules', []):
            if _model_subcondition_matches(model_rule.get('match', {}), run_config['model']):
                run_config['extra_params'].update(model_rule.get('extra_params', {}))
        if 'extra_params' in rule:
            run_config['extra_params'].update(rule['extra_params'])
        for para_rule in rule.get('parallel_rules', []):
            if _parallel_rule_matches(para_rule.get('match', {}), parallel_config):
                run_config['extra_params'].update(para_rule.get('extra_params', {}))


def get_func_config_list(backend: str,
                         parallel_config: dict[str, int],
                         model_type: str = 'chat_model',
                         func_type: str = 'func',
                         extra: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Generate all valid running config combinations (communicator + quant
    policy + model).

    Args:
        backend: Backend type (turbomind/pytorch)
        parallel_config: Parallel config for tensor parallel
        model_type: Model type, default: chat_model
        func_type: Test func type filter, default: func
        extra: extra config merged into each run config's extra_params.
    Returns:
        list[dict]: All valid run config dicts
    """
    config = get_config()
    device = config.get('device', 'cuda')
    base_case_list = get_model_list(config, backend, parallel_config, model_type, func_type)

    if extra is None:
        extra = {}

    run_configs = []
    dtype = 'float16' if not is_bf16_supported(device) else None

    quantization_config = config.get(f'{backend}_quantization', {})
    fp8_model_list = quantization_config.get('fp8', [])

    def get_model_extra_params(model: str) -> dict:
        if model in fp8_model_list:
            return {'model-format': 'fp8'}
        return {}

    for communicator in _get_communicator_list(config, backend, parallel_config):
        for model in base_case_list:
            for quant_policy in [0, 4, 8, 42]:
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
                # Prefix caching is unsupported when linear attention is present
                if 'enable-prefix-caching' in extra and 'Qwen3.5' in model:
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

                model_extra_params = get_model_extra_params(model)
                if model_extra_params and quant_policy == 0:
                    run_config_with_format = copy.deepcopy(run_config)
                    run_config_with_format['extra_params'].update(model_extra_params)
                    run_configs.append(run_config_with_format)

                run_configs.append(run_config)

    for run_config in run_configs:
        apply_model_run_params(run_config, config, extra, func_type, backend)

    return run_configs


def get_cli_common_param(run_config: dict[str, Any]) -> str:
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
    if len(extra_params) > 0:
        cli_params.append(get_cli_str(extra_params))
    cli_params.append('--trust-remote-code')

    return ' '.join(cli_params).strip()


def get_cli_str(config: dict[str, Any]) -> str:
    cli_str = []
    # Extra params
    for key, value in config.items():
        key = key.replace('_', '-')
        if value is None:
            cli_str.append(f'--{key}')
        elif isinstance(value, list):
            tmp_cli = ' '.join(map(str, value))
            cli_str.append(f'--{key} {tmp_cli}')
        elif isinstance(value, dict):
            tmp_cli = ' '.join([f'{k}={v}' for k, v in value.items()])
            cli_str.append(f'--{key} {tmp_cli}')
        else:
            cli_str.append(f'--{key} {value}' if value else f'--{key}')
    return ' '.join(cli_str)


def get_parallel_config(config: dict[str, Any], model_name: str) -> list[dict[str, int]]:
    """Get matched parallel config dict by model name, default tp:1 if no
    match."""
    result = []
    base_model = _base_model_name(model_name)
    parallel_configs = config.get('config', {})

    for conf_key, model_map in parallel_configs.items():
        if model_map is None:
            continue
        if base_model in model_map:
            conf_value = model_map[base_model]
            if isinstance(conf_value, dict):
                result.append(conf_value.copy())
            elif isinstance(conf_value, int):
                result.append({conf_key: conf_value})

    return result if result else [{'tp': 1}]


def _extract_models_from_config(config_value: Any) -> list[str]:
    """Extract flat model name list from config value (dict/list supported)"""
    models = []
    if isinstance(config_value, dict):
        for model_list in config_value.values():
            if isinstance(model_list, list):
                models.extend([m for m in model_list if isinstance(m, str)])
    elif isinstance(config_value, list):
        models.extend([m for m in config_value if isinstance(m, str)])
    return models


def get_model_list(config: dict[str, Any],
                   backend: str,
                   parallel_config: dict[str, int] | None = None,
                   model_type: str = 'chat_model',
                   func_type: str = 'func') -> list[str]:
    """Get filtered model list with quantization extended models by
    backend/parallel config/model type/func type.

    Args:
        config: Global system config dict
        backend: Backend type (turbomind/pytorch)
        parallel_config: Parallel filter config
        model_type: Model type, default: chat_model
        func_type: Test func type filter, default: func
    Returns:
        list[str]: Base models + quantization extended models
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


def _filter_by_test_func_type(config: dict[str, Any], model_list: list[str], func_type: str) -> list[str]:
    """Filter model list by test function type, return intersection of two
    model sets."""
    if func_type == 'func':
        return model_list

    filtered_models = []
    model_config_key = f'{func_type}_model'
    if model_config_key in config:
        filtered_models = _extract_models_from_config(config[model_config_key])

    return list(set(filtered_models) & set(model_list))


def _extend_turbomind_quant_models(quant_config: dict[str, Any], base_models: list[str],
                                   target_list: list[str]) -> None:
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


def _extend_pytorch_quant_models(quant_config: dict[str, Any], base_models: list[str], target_list: list[str]) -> None:
    """Append pytorch quantization models to target list (AWQ 4bits + W8A8)"""
    # Append AWQ quantization models
    for model_name in quant_config.get('awq', []):
        if model_name in target_list:
            target_list.append(model_name + SUFFIX_INNER_AWQ)
    # Append W8A8 quantization models
    for model_name in quant_config.get('w8a8', []):
        if model_name in target_list:
            target_list.append(model_name + SUFFIX_INNER_W8A8)


def _is_kvint_model(config: dict[str, Any], backend: str, model: str, quant_policy: int) -> bool:
    """Check if model supports the kv quantization policy, quant_policy=0
    always return True."""
    if quant_policy == 0:
        return True
    if quant_policy in [4, 8]:
        no_kvint_black_list = config.get(f'{backend}_quantization', {}).get(f'no_kvint{quant_policy}', [])
        return _base_model_name(model) not in no_kvint_black_list

    if quant_policy == 42:
        kv42_list = config.get(f'{backend}_quantization', {}).get('kvint42', [])
        return _base_model_name(model) in kv42_list
    return False

def _base_model_name(model: str) -> str:
    """Simplify model name by removing quantization suffix for config
    matching."""
    return model.replace('-inner-4bits', '').replace('-inner-w8a8', '').replace('-inner-gptq', '')


def get_quantization_model_list(type: str) -> list[str]:
    """Get quantization model list by specified quant type(awq/gptq/w8a8)"""
    config = get_config()
    quant_model_list = []

    # Get all chat/base models & deduplicate
    turbomind_chat = _extract_models_from_config(
        config['turbomind_chat_model']) if 'turbomind_chat_model' in config else []
    turbomind_base = _extract_models_from_config(
        config['turbomind_base_model']) if 'turbomind_base_model' in config else []
    all_turbomind_models = list(OrderedDict.fromkeys(turbomind_chat + turbomind_base))

    pytorch_chat = _extract_models_from_config(config['pytorch_chat_model']) if 'pytorch_chat_model' in config else []
    pytorch_base = _extract_models_from_config(config['pytorch_base_model']) if 'pytorch_base_model' in config else []
    all_pytorch_models = list(OrderedDict.fromkeys(pytorch_chat + pytorch_base))

    if type == 'awq':
        # Filter turbomind valid awq models
        no_awq = config.get('turbomind_quantization', {}).get('no_awq', [])
        quant_model_list = [m for m in all_turbomind_models if m not in no_awq and not is_quantization_model(m)]

        # Append pytorch awq models
        torch_awq = config.get('pytorch_quantization', {}).get('awq', [])
        for model in torch_awq:
            if model not in quant_model_list:
                quant_model_list.append(model)

    elif type == 'gptq':
        gptq_model_list = config.get('turbomind_quantization', {}).get(type, [])
        for model in gptq_model_list:
            if model in all_turbomind_models:
                quant_model_list.append(model)
    elif type == 'w8a8':
        w8a8_model_list = config.get('pytorch_quantization', {}).get(type, [])
        for model in w8a8_model_list:
            if model in all_pytorch_models:
                quant_model_list.append(model)

    return quant_model_list


def get_config() -> dict[str, Any]:
    """Load & get yaml config file, auto adapt device env & update log path."""
    # Get device env & match config file path
    env_tag = os.environ.get('TEST_ENV')
    config_path = f'autotest/config_{env_tag}.yml' if env_tag else 'autotest/config.yml'

    # Fallback to default config if device-specific config not exist
    if env_tag and not os.path.exists(config_path):
        config_path = 'autotest/config.yml'
    # Load yaml config file safely
    with open(config_path, encoding='utf-8') as f:
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


def get_cuda_prefix_by_workerid(worker_id: str | None, parallel_config: dict[str, int] | None = None) -> str | None:
    """Get cuda/ascend visible devices env prefix by worker id & parallel
    config."""
    para_conf = parallel_config or {}
    device_type = os.environ.get('DEVICE', 'cuda')

    tp_num = para_conf.get('tp')
    if not tp_num:
        return ''

    cuda_id = get_cuda_id_by_workerid(worker_id, tp_num)
    if not cuda_id:
        return ''

    return f'ASCEND_RT_VISIBLE_DEVICES={cuda_id}' if device_type == 'ascend' else f'CUDA_VISIBLE_DEVICES={cuda_id}'


def get_cuda_id_by_workerid(worker_id: str | None, tp_num: int = 1) -> str | None:
    """Get cuda id str by worker id and tp num, return None if invalid worker
    id."""
    if worker_id is None or 'gw' not in worker_id:
        return None

    base_id = int(worker_id.replace('gw', ''))
    cuda_num = base_id * tp_num
    return ','.join([str(cuda_num + i) for i in range(tp_num)])


def get_workerid(worker_id: str | None) -> int:
    """Parse numeric worker id from worker id str, return 0 if invalid worker
    id."""
    if worker_id is None or 'gw' not in worker_id:
        return 0

    return int(worker_id.replace('gw', ''))


def is_quantization_model(model: str) -> bool:
    """Check if model name contains quantization related keywords."""
    lower_name = model.lower()
    return any(key in lower_name for key in ('awq', '4bits', 'w4', 'int4'))


def _get_communicator_list(config: dict[str, Any],
                           backend: str,
                           parallel_config: dict[str, int] | None = None) -> list[str]:
    """Get available communicator list by device and parallel config."""
    device = config.get('device', None)

    if device == 'ascend':
        return ['nccl']
    if backend == 'pytorch':
        return ['nccl']
    if ('cp' in parallel_config or 'dp' in parallel_config or 'ep' in parallel_config):
        return ['nccl']
    if 'tp' in parallel_config and parallel_config['tp'] == 1:
        return ['nccl']

    return ['nccl', 'cuda-ipc']


def set_device_env_variable(worker_id: str | None, parallel_config: dict[str, int] | None = None) -> None:
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


def is_model_in_list(config: dict[str, Any], parallel_config: dict[str, int], model: str) -> bool:
    """Check if model matches the target parallel config."""
    model_config = get_parallel_config(config, model)
    return parallel_config in model_config


_MODEL_EVAL_CONFIG_RULES = (
    ('gpt', 'gpt'),
    ('sdar', 'sdar'),
    ('intern-s1-pro', 'intern-s1-pro'),
    ('qwen3.5', 'qwen3.5'),
)

def _resolve_base_eval_config_name(run_config: dict[str, Any], rules: tuple[tuple[str, str], ...]) -> str:
    model = run_config['model'].lower()
    for needle, resolved in rules:
        if needle in model:
            return resolved
    return 'default'


def _apply_eval_config_env_suffix(config: dict[str, Any], name: str) -> str:
    env_tag = str(config['env_tag'])
    if env_tag == 'a100':
        return f'{name}-32k'
    if env_tag == 'ascend':
        return f'{name}-2batch'
    return name


def resolve_eval_config_name(config: dict[str, Any],
                             run_config: dict[str, Any],
                             eval_config_name: str = 'default',
                             *,
                             only_if_default: bool = True) -> str:
    """Resolve eval preset key (EVAL_CONFIGS / MLLM_EVAL_CONFIGS) from model
    and env_tag."""
    if only_if_default and eval_config_name != 'default':
        return eval_config_name

    if eval_config_name == 'default':
        name = _resolve_base_eval_config_name(run_config, _MODEL_EVAL_CONFIG_RULES)
    else:
        name = eval_config_name

    return _apply_eval_config_env_suffix(config, name)


def get_case_str_by_config(run_config: dict[str, Any], is_simple: bool = True) -> str:
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
    model_format = extra_params.get('model-format')
    if model_format:
        extra_params_case += f'_{model_format}'
    if not is_simple:
        for k, v in extra_params.items():
            if len(v) > 10:
                extra_params_case += f'_{k}'.replace('_', '-').replace('/', '-').replace('.', '-')
            else:
                extra_params_case += f'_{k}{v}'.replace('_', '-').replace('/', '-').replace('.', '-')

    return f'{backend_type}_{pure_model_name}_{communicator}_{parallel_str}_{quant_policy}{extra_params_case}'


def parse_config_by_case(case_str: str) -> dict[str, Any]:
    """Parse run config dict from case name string (fix split & type convert
    bug)"""
    case_parts = case_str.split('_')
    if len(case_parts) < 4:
        raise ValueError(f'Invalid case string: {case_str}')

    backend = case_parts[0]
    model = case_parts[1]
    communicator = case_parts[2]

    quant_idx = None
    for i in range(len(case_parts) - 1, 2, -1):
        if case_parts[i].isdigit():
            quant_idx = i
            break
    if quant_idx is None:
        raise ValueError(f'No numeric quant policy found in case string: {case_str}')

    quant_policy = int(case_parts[quant_idx])
    parallel_parts = case_parts[3:quant_idx]

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
            'cache_max_entry_count': 0.75,
            'adapters': {
                'a': 'lora/2024-01-25_self_dup',
                'b': 'lora/2024-01-25_self'
            }
        }
    }

    cli_params = get_cli_common_param(run_config)
    assert cli_params == '--backend turbomind --communicator nccl --quant-policy 8 --model-format awq --dp 16 --ep 16 --dtype bfloat16 --device ascend --enable-prefix-caching --max-batch-size 2048 --session-len 8192 --cache-max-entry-count 0.75 --adapters a=lora/2024-01-25_self_dup b=lora/2024-01-25_self --trust-remote-code', cli_params  # noqa
    run_config = {
        'model': 'test/test_dpep16-inner-4bits',
        'backend': 'pytorch',
        'communicator': 'nccl',
        'quant_policy': 0,
        'parallel_config': {
            'tp': 8
        }
    }

    cli_params = get_cli_common_param(run_config)
    assert cli_params == '--backend pytorch --communicator nccl --model-format awq --tp 8 --trust-remote-code', cli_params # noqa
    os.unsetenv('TEST_ENV')


def test_return_info_turbomind():
    os.environ['TEST_ENV'] = 'test'
    backend = 'turbomind'
    func_chat_tp1 = get_func_config_list(backend, parallel_config={'tp': 1}, model_type='chat_model', func_type='func')
    assert len(func_chat_tp1) == 12, len(func_chat_tp1)
    func_chat_tp2 = get_func_config_list(backend, parallel_config={'tp': 2}, model_type='chat_model', func_type='func')
    assert len(func_chat_tp2) == 32, len(func_chat_tp2)
    func_chat_tp8 = get_func_config_list(backend, parallel_config={'tp': 8}, model_type='chat_model', func_type='func')
    assert len(func_chat_tp8) == 36, len(func_chat_tp8)
    func_chat_cptp = get_func_config_list(backend,
                                          parallel_config={
                                              'cp': 2,
                                              'tp': 8
                                          },
                                          model_type='chat_model',
                                          func_type='func')
    assert len(func_chat_cptp) == 14, len(func_chat_cptp)
    func_chat_dpep8 = get_func_config_list(backend,
                                           parallel_config={
                                               'dp': 8,
                                               'ep': 8
                                           },
                                           model_type='chat_model',
                                           func_type='func')
    assert len(func_chat_dpep8) == 6, len(func_chat_dpep8)
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
    assert len(func_chat_tp8) == 9, len(func_chat_tp8)
    func_chat_cptp = get_func_config_list(backend,
                                          parallel_config={
                                              'cp': 2,
                                              'tp': 8
                                          },
                                          model_type='chat_model',
                                          func_type='func')
    assert len(func_chat_cptp) == 7, len(func_chat_cptp)
    func_chat_dpep8 = get_func_config_list(backend,
                                           parallel_config={
                                               'dp': 8,
                                               'ep': 8
                                           },
                                           model_type='chat_model',
                                           func_type='func')
    assert len(func_chat_dpep8) == 8, len(func_chat_dpep8)
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
    assert run_config2['communicator'] == 'nccl'
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
    assert run_config3['communicator'] == 'nccl'
    assert run_config3['quant_policy'] == 0
    assert run_config3['parallel_config'] == {'tp': 1}
    assert run_config3['extra_params']['speculative_algorithm'] == 'eagle'
    assert run_config3['extra_params']['session_len'] == 1024
    os.unsetenv('TEST_ENV')


def test_resolve_eval_config_name():
    run_config = {'model': 'openai/gpt-oss-120b'}
    assert resolve_eval_config_name({}, run_config) == 'gpt'
    assert resolve_eval_config_name({'env_tag': 'a100'}, run_config) == 'gpt-32k'
    assert resolve_eval_config_name({'env_tag': 'ascend'}, run_config) == 'gpt-2batch'
    assert resolve_eval_config_name({}, run_config, 'longtext-512k') == 'longtext-512k'
    assert resolve_eval_config_name({'env_tag': 'ascend'}, run_config, 'longtext-512k') == 'longtext-512k'

    sdar_config = {'model': 'inclusionAI/SDAR-30B-A3B'}
    assert resolve_eval_config_name({}, sdar_config) == 'sdar'
    qwen_config = {'model': 'Qwen/Qwen3.5-397B-A17B'}
    assert resolve_eval_config_name({}, qwen_config) == 'qwen3.5'
    intern_config = {'model': 'internlm/Intern-S1-Pro-FP8'}
    assert resolve_eval_config_name({}, intern_config) == 'intern-s1-pro'
    assert resolve_eval_config_name({}, {'model': 'meta/llama'}) == 'default'
    mllm_config = {'model': 'Qwen/Qwen3.5-VL-7B'}
    assert resolve_eval_config_name({}, mllm_config) == 'qwen3.5'
    assert resolve_eval_config_name({'env_tag': 'ascend'}, mllm_config) == 'qwen3.5-2batch'


def test_get_parallel_config():
    test = get_parallel_config({}, 'empty')
    assert test == [{'tp': 1}]
    test = get_parallel_config(
        {
            'config': {
                'tp': {
                    'empty': 1
                },
                'dp_ep': {
                    'empty': {
                        'dp': 1,
                        'ep': 8
                    }
                },
                'cp_tp': {
                    'empty': {
                        'cp': 8,
                        'tp': 8
                    }
                }
            }
        }, 'empty')
    assert test == [{'tp': 1}, {'dp': 1, 'ep': 8}, {'cp': 8, 'tp': 8}]


def _apply_model_run_params_for_test(model: str,
                                     env_tag: str = '',
                                     func_type: str = 'func',
                                     backend: str = 'pytorch',
                                     extra: dict[str, Any] | None = None,
                                     parallel_config: dict[str, int] | None = None) -> dict[str, Any]:
    extra = extra or {}
    run_config = {
        'model': model,
        'extra_params': copy.copy(extra),
        'parallel_config': parallel_config or {'dp': 1, 'ep': 1, 'tp': 1},
    }
    apply_model_run_params(run_config, {'env_tag': env_tag}, extra, func_type, backend)
    return run_config['extra_params']


def test_apply_model_run_params():
    rules = _load_model_run_params_rules()
    assert len(rules) == 12
    assert rules[0]['name'] == 'qwen3-235b-thinking-2507'

    params = _apply_model_run_params_for_test('x/Qwen3-235B-A22B-Thinking-2507', func_type='benchmark')
    assert params['cache-max-entry-count'] == 0.9
    assert params['max-batch-size'] == 1024

    params = _apply_model_run_params_for_test('x/Qwen3-235B-A22B-Thinking-2507',
                                              parallel_config={'dp': 8, 'ep': 8, 'tp': 1})
    assert params['max-batch-size'] == 256

    params = _apply_model_run_params_for_test('org/GLM-5-FP8')
    assert params['cache-max-entry-count'] == 0.9
    assert params['max-batch-size'] == 128

    params = _apply_model_run_params_for_test('some/model', func_type='evaluate')
    assert params['session_len'] == 65536

    params = _apply_model_run_params_for_test('THUDM/cogvlm-chat-hf', func_type='evaluate')
    assert params['session-len'] == 32568

    params = _apply_model_run_params_for_test('THUDM/cogvlm-chat-hf', func_type='func')
    assert params['session-len'] == 32568

    params = _apply_model_run_params_for_test('some/model', func_type='evaluate', extra={'session_len': 4096})
    assert params['session_len'] == 4096

    params = _apply_model_run_params_for_test('Qwen3.5-7B', func_type='evaluate')
    assert 'session_len' not in params
    assert 'session-len' not in params

    params = _apply_model_run_params_for_test('x/Qwen3-235B-A22B-Thinking-2507', env_tag='3090')
    assert params['cache-max-entry-count'] == 0.5

    params = _apply_model_run_params_for_test('x/Qwen3-235B-A22B-Thinking-2507', env_tag='5080')
    assert params['cache-max-entry-count'] == 0.5

    params = _apply_model_run_params_for_test('x/Qwen3-235B-A22B', env_tag='a100')
    assert params['cache-max-entry-count'] == 0.6

    params = _apply_model_run_params_for_test('internlm/Intern-S1', env_tag='a100')
    assert params['cache-max-entry-count'] == 0.6

    params = _apply_model_run_params_for_test('x/Qwen3-235B-A22B-Thinking-2507', env_tag='a100')
    assert params['cache-max-entry-count'] == 0.6

    params = _apply_model_run_params_for_test('My-SDAR-Model')
    assert params['dllm-block-length'] == 4
    assert params['dllm-denoising-steps'] == 4
    assert params['dllm-confidence-threshold'] == 0.9

    params = _apply_model_run_params_for_test('kimi-k2', parallel_config={'dp': 16, 'ep': 16, 'tp': 1})
    assert params['max-batch-size'] == 256

    params = _apply_model_run_params_for_test('kimi-k2', parallel_config={'dp': 8, 'ep': 8, 'tp': 1})
    assert 'max-batch-size' not in params

    params = _apply_model_run_params_for_test('Intern-S1-Pro-FP8', parallel_config={'dp': 16, 'ep': 16, 'tp': 1})
    assert params['model-format'] == 'fp8'
    assert params['max-prefill-token-num'] == 1024
    assert params['max-batch-size'] == 128

    params = _apply_model_run_params_for_test('Intern-S1-Pro-BF16', parallel_config={'dp': 16, 'ep': 16, 'tp': 1})
    assert 'model-format' not in params
    assert params['max-prefill-token-num'] == 1024

    params = _apply_model_run_params_for_test('openai/gpt-oss-20b',
                                              func_type='benchmark',
                                              backend='turbomind')
    assert params['model-format'] == 'mxfp4'

    params = _apply_model_run_params_for_test('openai/gpt-oss-20b',
                                              func_type='benchmark',
                                              backend='pytorch')
    assert 'model-format' not in params

    params = _apply_model_run_params_for_test('Qwen3.5-7B', func_type='mtp_evaluate')
    assert params['reasoning-parser'] == 'qwen-qwq'
    assert params['speculative-algorithm'] == 'qwen3_5_mtp'
    assert params['speculative-num-draft-tokens'] == 4
    assert params['max-batch-size'] == 256


if __name__ == '__main__':
    test_apply_model_run_params()
    test_resolve_eval_config_name()
    test_get_parallel_config()
    test_cli_common_param()
    test_run_config()
    test_get_case_str_by_config()
    test_return_info_pytorch()
    test_config()
    test_return_info_turbomind()
