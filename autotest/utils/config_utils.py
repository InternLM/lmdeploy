import copy
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import utils.constant as constant
import yaml

from lmdeploy.utils import is_bf16_supported

DepsProfileSelector = str | dict[str, str]

SUFFIX_INNER_AWQ = '-inner-4bits'
SUFFIX_INNER_GPTQ = '-inner-gptq'
SUFFIX_INNER_W8A8 = '-inner-w8a8'

_AUTOTEST_ROOT = os.path.join(os.path.dirname(__file__), '..')
CONFIGS_DIR = os.path.join(_AUTOTEST_ROOT, 'configs')
ENV_PATHS_YML = os.path.join(_AUTOTEST_ROOT, 'env_paths.yml')
PATHS_YML = ENV_PATHS_YML  # alias for error messages / imports
PARALLEL_LAYOUT_KEYS = ('tp', 'dp', 'ep', 'cp')
ENGINE_CONFIG_KEY = 'engine_config'
TEST_COVERAGE_KEY = 'test_coverage'


def _entry_engine_config(entry: dict[str, Any]) -> dict[str, Any]:
    """Per-model yaml engine / parallel block (``engine_config``; legacy
    ``parallel``)."""
    return entry.get(ENGINE_CONFIG_KEY) or entry.get('parallel') or {}
PROFILE_TO_MODEL_TYPE_KEY = {
    'chat': 'chat_model',
    'vl': 'vl_model',
    'base': 'base_model',
}
# Filter model yaml rows by entry ``deps`` in per-model yaml only.
# Unset or empty ``DEPS_PROFILE``: only rows with no entry-level deps pins.
# Selector: pip-style ``pkg==ver`` (multi-key: space or ``;`` between tokens).
# ``all``: disable filtering (tests / debug).
DEPS_PROFILE_ENV = 'DEPS_PROFILE'
EMPTY_DEPS_SELECTOR = '__empty__'


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


_paths_doc_cache: dict[str, Any] | None = None


def _matrix_env_key(env_key: str) -> str:
    """Top-level key in per-model yaml (``*_legacy`` flat sources are merged
    under the base env)."""
    if not env_key:
        return 'a100'
    if env_key == 'legacy':
        return 'a100'
    if env_key.endswith('_legacy'):
        return env_key[: -len('_legacy')]
    return env_key


def _normalize_dep_spec_value(value: str) -> str | None:
    if value.lower() in ('null', 'none', ''):
        return None
    return value.strip()


def _dep_spec_values_equal(expected: str, actual: Any) -> bool:
    exp = _normalize_dep_spec_value(expected)
    if exp is None:
        return actual is None
    return str(actual).strip() == exp


def _parse_deps_kv_chunk(chunk: str) -> tuple[str, str]:
    chunk = chunk.strip()
    if '==' in chunk:
        key, value = chunk.split('==', 1)
    else:
        raise ValueError(f'invalid deps profile chunk: {chunk!r}')
    return key.strip(), value.strip()


def _split_deps_profile_chunks(text: str) -> list[str]:
    text = text.strip()
    if ';' in text:
        return [c.strip() for c in text.split(';') if c.strip()]
    if '==' in text and ' ' in text:
        return [c for c in text.split() if c.strip()]
    return [text]


def format_deps_profile_env(selector: dict[str, str]) -> str:
    """Canonical ``DEPS_PROFILE`` / ``pip install`` line (``pkg==ver``
    tokens)."""
    return ' '.join(f'{key}=={value}' for key, value in selector.items())


def parse_deps_profile_selector(raw: str) -> DepsProfileSelector:
    """Parse non-empty ``DEPS_PROFILE`` (``pkg==ver`` tokens or ``all``)."""
    text = raw.strip()
    if not text:
        return EMPTY_DEPS_SELECTOR
    if text == 'all':
        return 'all'
    if text.startswith('profile='):
        return text.split('=', 1)[1].strip()
    if text.startswith('profile:'):
        return text.split(':', 1)[1].strip()
    if '==' in text:
        selector: dict[str, str] = {}
        for chunk in _split_deps_profile_chunks(text):
            key, value = _parse_deps_kv_chunk(chunk)
            selector[key] = value
        return selector
    return text


def deps_profile_to_pip_specs(raw: str) -> str:
    """Space-separated pip requirements for ``pip install`` (empty when unset /
    non-dict selector)."""
    text = (raw or '').strip()
    if not text:
        return ''
    parsed = parse_deps_profile_selector(text)
    if isinstance(parsed, dict):
        return format_deps_profile_env(parsed)
    return ''


def get_deps_profile_selector() -> DepsProfileSelector:
    """Active deps selector (env: ``DEPS_PROFILE``).

    Empty/unset → :data:`EMPTY_DEPS_SELECTOR`.
    """
    explicit = (os.environ.get(DEPS_PROFILE_ENV) or '').strip()
    if not explicit:
        return EMPTY_DEPS_SELECTOR
    return parse_deps_profile_selector(explicit)


def get_deps_profile() -> DepsProfileSelector:
    """Alias of :func:`get_deps_profile_selector`."""
    return get_deps_profile_selector()


def _entry_has_empty_deps(entry: dict[str, Any]) -> bool:
    """Entry-level ``deps`` absent or only null placeholders (no ``profile`` /
    pins)."""
    deps = entry.get('deps')
    if deps is None:
        return True
    if not isinstance(deps, dict) or not deps:
        return True
    for key, value in deps.items():
        if key == 'profile' and value:
            return False
        if value is not None:
            return False
    return True


def _model_matrix_env_key(config: dict[str, Any]) -> str:
    """Env key for ``configs/<org>/<model>.yml`` list items (``TEST_ENV`` wins
    over ``env_tag``)."""
    test_env = os.environ.get('TEST_ENV')
    if test_env:
        return _matrix_env_key(test_env)
    return _matrix_env_key(str(config.get('env_tag', 'a100')))


def _per_model_configs_available() -> bool:
    return os.path.isdir(CONFIGS_DIR) and os.path.isfile(PATHS_YML)


def _load_paths_doc() -> dict[str, Any]:
    global _paths_doc_cache
    if _paths_doc_cache is None:
        _paths_doc_cache = _load_yaml(PATHS_YML) if os.path.isfile(PATHS_YML) else {}
    return _paths_doc_cache


def _resolve_paths_env_key(test_env: str | None) -> str:
    """Map ``TEST_ENV`` to a block in ``autotest/env_paths.yml``."""
    if not test_env:
        return 'a100'
    if test_env == 'legacy':
        return 'a100_legacy'
    doc = _load_paths_doc()
    if test_env in doc and isinstance(doc[test_env], dict):
        return test_env
    base = _matrix_env_key(test_env)
    if base in doc and isinstance(doc[base], dict):
        return base
    return test_env


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _load_paths_for_env(env_key: str) -> dict[str, Any]:
    paths_doc = _load_yaml(PATHS_YML)
    block = paths_doc.get(env_key) or paths_doc.get(str(env_key)) or {}
    config: dict[str, Any] = {
        'env_tag': block.get('env_tag', env_key),
        'device': block.get('device', 'cuda'),
    }
    config.update(block.get('paths') or {})
    return config


def _apply_run_id_paths(config: dict[str, Any]) -> None:
    if os.environ.get('CONFIG_COMPARE_SKIP_MKDIRS'):
        return
    run_id = os.environ.get('RUN_ID', 'local_run')
    run_suffix = str(run_id).replace('/', '_')
    for key in ('log_path', 'eval_path', 'mllm_eval_path', 'benchmark_path', 'server_log_path'):
        if key in config:
            config[key] = os.path.join(config[key], run_suffix)
            os.makedirs(config[key], exist_ok=True)


def _model_id_from_config_path(path: str) -> str:
    rel = os.path.relpath(path, CONFIGS_DIR)
    return rel.replace(os.sep, '/').removesuffix('.yml')


def _iter_model_config_paths() -> list[str]:
    paths = []
    for path in sorted(Path(CONFIGS_DIR).rglob('*.yml')):
        if 'environments' in path.parts:
            continue
        paths.append(str(path))
    return paths


def _normalize_profiles(model_type_field) -> list[str]:
    if isinstance(model_type_field, list):
        return list(model_type_field)
    return [model_type_field]


def _parallel_layout(parallel: dict[str, Any]) -> dict[str, int]:
    layout: dict[str, int] = {}
    for key in PARALLEL_LAYOUT_KEYS:
        if key in parallel:
            layout[key] = int(parallel[key])
    return layout or {'tp': 1}


def _parallel_launch_extra(engine_config: dict[str, Any]) -> dict[str, Any]:
    extra = engine_config.get('extra')
    return copy.deepcopy(extra) if isinstance(extra, dict) else {}


def _entry_launch_extra_sig(entry: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    launch_extra = _parallel_launch_extra(_entry_engine_config(entry))
    return tuple(sorted(launch_extra.items()))


def _parallel_dicts_equal(a: dict[str, int], b: dict[str, int]) -> bool:
    return a == b


def _normalize_entry_backends(
    entry: dict[str, Any],
    config: dict[str, Any],
    parallel_config: dict[str, int] | None = None,
) -> dict[str, list[str]]:
    """Normalize ``entry['backends']`` to ``{backend: [communicators...]}``.

    Supported yaml forms:
    - legacy: ``backends: [turbomind, pytorch]``
    - redundant: ``backends: [{name: turbomind, communicators: [nccl, cuda-ipc]}]``
    """
    normalized: dict[str, list[str]] = {}
    backends = entry.get('backends') or []
    for item in backends:
        backend_name = None
        communicators: list[str] | None = None
        if isinstance(item, str):
            backend_name = item
        elif isinstance(item, dict):
            backend_name = item.get('name') or item.get('backend') or item.get('type')
            comm_value = item.get('communicators', item.get('communicator'))
            if isinstance(comm_value, str):
                communicators = [comm_value]
            elif isinstance(comm_value, list):
                communicators = [str(c) for c in comm_value if c]
        if not backend_name:
            continue
        if not communicators:
            communicators = _get_communicator_list(config, backend_name, parallel_config)
        deduped = list(OrderedDict.fromkeys(communicators))
        normalized[backend_name] = deduped or _get_communicator_list(config, backend_name, parallel_config)
    return normalized


def _entry_deps_dict(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Pinned deps from the model yaml entry only (no global ``deps.yml``)."""
    deps = entry.get('deps')
    if not isinstance(deps, dict) or not deps:
        return None
    merged = {key: value for key, value in deps.items() if key != 'profile' and value is not None}
    return merged or None


def _entry_matches_deps_profile(entry: dict[str, Any], env_key: str, selector: DepsProfileSelector) -> bool:
    del env_key  # kept for call-site stability
    if selector == EMPTY_DEPS_SELECTOR:
        return _entry_has_empty_deps(entry)
    if selector == 'all':
        return True
    if isinstance(selector, dict):
        pinned = _entry_deps_dict(entry) or {}
        return all(_dep_spec_values_equal(exp, pinned.get(key)) for key, exp in selector.items())
    return False


def _entry_matches_func(entry: dict[str, Any], func_type: str, extra: dict[str, Any] | None) -> bool:
    funcs = set(entry.get(TEST_COVERAGE_KEY) or [])
    extra = extra or {}
    if extra.get('enable-prefix-caching') is not None or extra.get('enable_prefix_caching') is not None:
        return 'prefix_cache' in funcs
    if func_type == 'benchmark' and funcs == {'prefix_cache'}:
        return False
    if func_type == 'func':
        return 'func' in funcs
    return func_type in funcs


def _entry_matches_profile(entry: dict[str, Any], model_type: str) -> bool:
    profile_name = model_type.replace('_model', '')
    return profile_name in _normalize_profiles(entry.get('model_type', 'chat'))


def _iter_per_model_entries(env_key: str, deps_profile: DepsProfileSelector | None = None):
    active_profile = deps_profile if deps_profile is not None else get_deps_profile_selector()
    for path in _iter_model_config_paths():
        model_id = _model_id_from_config_path(path)
        doc = _load_yaml(path)
        entries = doc.get(env_key) or doc.get(str(env_key))
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if not _entry_matches_deps_profile(entry, env_key, active_profile):
                continue
            yield model_id, entry


def _quant_cfg_for_entry(entry: dict[str, Any]) -> dict[str, list[str]]:
    return entry.get('quantization') or {}


def _is_kvint_enabled_in_entry(
    backend: str,
    base_model: str,
    quant_policy: int,
    quant_cfg: dict[str, list[str]],
) -> bool:
    if quant_policy == 0:
        return True
    enabled = set(quant_cfg.get(backend) or [])
    if quant_policy in (4, 8):
        return f'kvint{quant_policy}' in enabled
    if quant_policy == 42:
        return 'kvint42' in enabled
    return False


def _extend_quant_models_from_entry(
    backend: str,
    base_models: list[str],
    quant_cfg: dict[str, list[str]],
    target: list[str],
) -> None:
    enabled = set(quant_cfg.get(backend) or [])
    for model_name in base_models:
        if model_name not in target:
            continue
        if 'awq' in enabled and not is_quantization_model(model_name):
            target.append(model_name + SUFFIX_INNER_AWQ)
        if backend == 'turbomind' and 'gptq' in enabled:
            target.append(model_name + SUFFIX_INNER_GPTQ)
        if backend == 'pytorch' and 'w8a8' in enabled:
            target.append(model_name + SUFFIX_INNER_W8A8)


def _build_run_config_entry(
    model_id: str,
    entry: dict[str, Any],
    backend: str,
    communicator: str,
    parallel_config: dict[str, int],
    quant_policy: int,
    config: dict[str, Any],
    func_type: str,
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    launch_extra = _parallel_launch_extra(_entry_engine_config(entry))
    merged_extra = copy.deepcopy(launch_extra)
    if extra:
        merged_extra.update(extra)
    if extra and extra.get('enable-prefix-caching') is not None:
        if 'prefix_cache' in (entry.get(TEST_COVERAGE_KEY) or []):
            merged_extra['enable-prefix-caching'] = None

    device = config.get('device', 'cuda')
    dtype = 'float16' if not is_bf16_supported(device) else None

    run_config: dict[str, Any] = {
        'model': model_id,
        'backend': backend,
        'communicator': communicator,
        'quant_policy': quant_policy,
        'parallel_config': copy.deepcopy(parallel_config),
        'extra_params': merged_extra,
    }
    if dtype and backend == 'pytorch':
        run_config['extra_params']['dtype'] = dtype
    if device != 'cuda':
        run_config['extra_params']['device'] = device
    if entry.get('gen_config'):
        run_config['gen_config'] = copy.deepcopy(entry['gen_config'])
    deps = _entry_deps_dict(entry)
    if deps:
        run_config['deps'] = deps
    return run_config


def _get_func_config_list_per_model(
    config: dict[str, Any],
    backend: str,
    parallel_config: dict[str, int],
    model_type: str,
    func_type: str,
    extra: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Expand run configs from autotest/configs/<org>/<model>.yml entries."""
    extra = extra or {}
    env_key = _model_matrix_env_key(config)
    deps_profile = get_deps_profile_selector()
    run_configs: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    base_case_list = get_model_list(
        config, backend, parallel_config, model_type, func_type, extra=extra,
    )

    for model_id, entry in _iter_per_model_entries(env_key, deps_profile):
        layout = _parallel_layout(_entry_engine_config(entry))
        if not _parallel_dicts_equal(layout, parallel_config):
            continue
        backend_map = _normalize_entry_backends(entry, config, layout)
        if backend not in backend_map:
            continue
        if not _entry_matches_profile(entry, model_type):
            continue
        if not _entry_matches_func(entry, func_type, extra):
            continue

        quant_cfg = _quant_cfg_for_entry(entry)
        base_model = model_id
        models_for_quant = [base_model]
        if 'quantization' in (entry.get(TEST_COVERAGE_KEY) or []):
            _extend_quant_models_from_entry(backend, [base_model], quant_cfg, models_for_quant)
        models_for_quant = [m for m in models_for_quant if m in base_case_list]
        launch_extra_sig = _entry_launch_extra_sig(entry)

        for model in models_for_quant:
            qcfg = quant_cfg
            for quant_policy in [0, 4, 8, 42]:
                if not _is_kvint_enabled_in_entry(backend, _base_model_name(model), quant_policy, qcfg):
                    continue
                for communicator in backend_map[backend]:
                    sig = (model, communicator, quant_policy, launch_extra_sig)
                    if sig in seen:
                        continue
                    seen.add(sig)
                    run_configs.append(
                        _build_run_config_entry(
                            model,
                            entry,
                            backend,
                            communicator,
                            parallel_config,
                            quant_policy,
                            config,
                            func_type,
                            extra,
                        ))
    return run_configs


def get_func_config_list(backend: str,
                         parallel_config: dict[str, int],
                         model_type: str = 'chat_model',
                         func_type: str = 'func',
                         extra: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Generate all valid running config combinations (communicator + quant
    policy + model).

    Per-model YAML (``autotest/configs/``): ``engine_config.extra`` = launch params;
    ``gen_config`` = request/eval sampling params on each run_config.
    """
    config = get_config()
    return _get_func_config_list_per_model(config, backend, parallel_config, model_type, func_type, extra)


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
    env_key = _model_matrix_env_key(config)
    deps_profile = get_deps_profile_selector()
    base_model = _base_model_name(model_name)
    layouts: list[dict[str, int]] = []
    seen: set[tuple] = set()
    for mid, entry in _iter_per_model_entries(env_key, deps_profile):
        if _base_model_name(mid) != base_model:
            continue
        funcs = entry.get(TEST_COVERAGE_KEY) or []
        if funcs == ['prefix_cache']:
            continue
        layout = _parallel_layout(_entry_engine_config(entry))
        key = tuple(sorted(layout.items()))
        if key not in seen:
            seen.add(key)
            layouts.append(layout)
    return layouts if layouts else [{'tp': 1}]


def _model_ids_for_entries(
    config: dict[str, Any],
    backend: str,
    parallel_config: dict[str, int],
    model_type: str,
    func_type: str,
    extra: dict[str, Any] | None,
) -> list[str]:
    """Model ids from yaml entries matching backend / profile / parallel /
    function.

    Always ignores rows with entry-level ``deps`` pins (see
    :func:`get_model_list`); use :func:`get_func_config_list` for
    ``DEPS_PROFILE``-scoped runs.
    """
    env_key = _model_matrix_env_key(config)
    deps_profile = EMPTY_DEPS_SELECTOR
    models: list[str] = []
    extended: list[str] = []
    for model_id, entry in _iter_per_model_entries(env_key, deps_profile):
        if not _entry_matches_profile(entry, model_type):
            continue
        if not _entry_matches_func(entry, func_type, extra):
            continue
        layout = _parallel_layout(_entry_engine_config(entry))
        if not _parallel_dicts_equal(layout, parallel_config):
            continue
        if backend not in _normalize_entry_backends(entry, config, layout):
            continue
        if model_id not in models:
            models.append(model_id)
            extended.append(model_id)
        if 'quantization' in (entry.get(TEST_COVERAGE_KEY) or []):
            _extend_quant_models_from_entry(
                backend, [model_id], _quant_cfg_for_entry(entry), extended,
            )
    return list(OrderedDict.fromkeys(extended))


def get_model_list(config: dict[str, Any],
                   backend: str,
                   parallel_config: dict[str, int] | None = None,
                   model_type: str = 'chat_model',
                   func_type: str = 'func',
                   extra: dict[str, Any] | None = None) -> list[str]:
    """Get filtered model list (same rules as legacy flat yaml).

    Non-``func`` types use ``pytorch/turbomind_{profile}`` ∩ ``{func_type}_model`` semantics:
    the model must appear under ``func`` for the same slice **and** under the target function.

    Rows with entry-level ``deps`` are never included (regardless of ``DEPS_PROFILE``).
    """
    parallel_config = parallel_config or {'tp': 1}
    if extra and (extra.get('enable-prefix-caching') is not None or extra.get('enable_prefix_caching') is not None):
        return _model_ids_for_entries(config, backend, parallel_config, model_type, func_type, extra)
    if func_type == 'func':
        return _model_ids_for_entries(config, backend, parallel_config, model_type, 'func', extra)

    chat_models = _model_ids_for_entries(config, backend, parallel_config, model_type, 'func', None)
    typed_models = _model_ids_for_entries(config, backend, parallel_config, model_type, func_type, extra)
    chat_bases = {_base_model_name(m) for m in chat_models}
    return [m for m in typed_models if _base_model_name(m) in chat_bases]


def _is_kvint_model(config: dict[str, Any], backend: str, model: str, quant_policy: int) -> bool:
    """Check KV quant policy support via per-model ``quantization`` blocks."""
    if quant_policy == 0:
        return True
    env_key = _model_matrix_env_key(config)
    deps_profile = get_deps_profile_selector()
    base = _base_model_name(model)
    for mid, entry in _iter_per_model_entries(env_key, deps_profile):
        if _base_model_name(mid) != base:
            continue
        layout = _parallel_layout(_entry_engine_config(entry))
        if backend not in _normalize_entry_backends(entry, config, layout):
            continue
        return _is_kvint_enabled_in_entry(backend, base, quant_policy, _quant_cfg_for_entry(entry))
    return False

def _base_model_name(model: str) -> str:
    """Simplify model name by removing quantization suffix for config
    matching."""
    return model.replace('-inner-4bits', '').replace('-inner-w8a8', '').replace('-inner-gptq', '')


def get_quantization_model_list(type: str) -> list[str]:
    """Get quantization model list by specified quant type(awq/gptq/w8a8)"""
    config = get_config()
    env_key = _model_matrix_env_key(config)
    deps_profile = get_deps_profile_selector()
    quant_model_list: list[str] = []
    for model_id, entry in _iter_per_model_entries(env_key, deps_profile):
        if 'quantization' not in (entry.get(TEST_COVERAGE_KEY) or []):
            continue
        layout = _parallel_layout(_entry_engine_config(entry))
        backend_map = _normalize_entry_backends(entry, config, layout)
        quant_cfg = _quant_cfg_for_entry(entry)
        for backend in ('turbomind', 'pytorch'):
            if backend not in backend_map:
                continue
            enabled = set(quant_cfg.get(backend) or [])
            if type == 'awq' and 'awq' in enabled and not is_quantization_model(model_id):
                quant_model_list.append(model_id)
            elif type == 'gptq' and 'gptq' in enabled and backend == 'turbomind':
                quant_model_list.append(model_id)
            elif type == 'w8a8' and 'w8a8' in enabled and backend == 'pytorch':
                quant_model_list.append(model_id)
    return list(OrderedDict.fromkeys(quant_model_list))


def get_config() -> dict[str, Any]:
    """Load global paths from ``autotest/env_paths.yml``; model matrices from
    ``configs/**``."""
    if not _per_model_configs_available():
        raise FileNotFoundError(
            f'Per-model autotest configs required: missing {PATHS_YML} or {CONFIGS_DIR}',
        )
    paths_key = _resolve_paths_env_key(os.environ.get('TEST_ENV'))
    config_copy = _load_paths_for_env(paths_key)
    _apply_run_id_paths(config_copy)
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


def is_pre_quantized_hf_model(model: str) -> bool:
    """HF weights are already quantized (AWQ/GPTQ/Int4); skip runtime weight-
    quant tests."""
    lower_name = model.lower()
    if 'gptq' in lower_name:
        return True
    return is_quantization_model(model)


def _get_communicator_list(config: dict[str, Any],
                           backend: str,
                           parallel_config: dict[str, int] | None = None) -> list[str]:
    """Get available communicator list by device and parallel config."""
    parallel_config = parallel_config or {}
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
    env_tag = str(config.get('env_tag') or _matrix_env_key(os.environ.get('TEST_ENV') or 'a100'))
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


_EVAL_OC_SCALAR_KEYS = frozenset({
    'query_per_second',
    'max_out_len',
    'max_seq_len',
    'batch_size',
    'temperature',
})


def _snake_key(key: str) -> str:
    return key.replace('-', '_')


def _gen_config_to_opencompass_kwargs(gen: dict[str, Any]) -> dict[str, Any]:
    """Map per-model yaml ``gen_config`` (kebab-case) to OpenCompass
    ``OpenAISDK`` keys."""
    result: dict[str, Any] = {}
    oai: dict[str, Any] = {}
    body: dict[str, Any] = {}
    for key, value in gen.items():
        snake = _snake_key(key)
        if snake == 'temperature':
            result['temperature'] = value
        elif snake in ('reasoning_effort', 'top_p'):
            oai[snake] = value
        elif snake in ('top_k', 'min_p', 'repetition_penalty', 'chat_template_kwargs'):
            body[snake] = value
        else:
            body[snake] = value
    if oai:
        result['openai_extra_kwargs'] = oai
    if body:
        result['extra_body'] = body
    return result


def _eval_table_scalar_params(preset: dict[str, Any]) -> dict[str, Any]:
    return {key: preset[key] for key in _EVAL_OC_SCALAR_KEYS if key in preset}


def get_eval_preset_config(
    config: dict[str, Any],
    run_config: dict[str, Any],
    eval_config_name: str = 'default',
    *,
    mllm: bool = False,
) -> dict[str, Any]:
    """Build kwargs for OpenCompass / VLMEvalKit from table preset + per-model
    yaml.

    Per-model ``gen_config`` overrides sampling fields; OpenCompass throughput /
    length limits (``query_per_second``, ``max_out_len``, …) still come from
    ``EVAL_CONFIGS`` keyed by :func:`resolve_eval_config_name`.
    """
    name = resolve_eval_config_name(config, run_config, eval_config_name)
    table = constant.MLLM_EVAL_CONFIGS if mllm else constant.EVAL_CONFIGS
    if mllm and name == 'default' and 'internvl' in run_config.get('model', '').lower():
        preset = table.get('internvl', {}) or table.get('default', {})
    else:
        preset = table.get(name, {})

    if mllm:
        merged = copy.deepcopy(preset)
        if run_config.get('gen_config'):
            merged.update(copy.deepcopy(run_config['gen_config']))
        return merged

    if run_config.get('gen_config'):
        result = _eval_table_scalar_params(preset)
        result.update(_gen_config_to_opencompass_kwargs(run_config['gen_config']))
        return result

    return copy.deepcopy(preset)


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
    spec_algo = extra_params.get('speculative-algorithm')
    if spec_algo:
        extra_params_case += f'_{spec_algo}'.replace('_', '-')
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

    # Convert parallel str to dict, e.g: ['tp1','dp2'] -> {'tp':1, 'dp':2}
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
        'quant_policy': quant_policy,
    }
