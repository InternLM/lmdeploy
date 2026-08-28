import copy
import json
import os
import re
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
INTERFACE_KEY = 'interface'
INTERFACE_SUITES = frozenset({'base', 'logprob', 'experts', 'anthropic', 'toolcall', 'reasoning'})
GENERATE_SUITES = frozenset({'base', 'logprob', 'experts'})
INTERFACE_SUITE_ORDER = ('base', 'logprob', 'experts', 'anthropic', 'toolcall', 'reasoning')
INTERFACE_BACKENDS_ENV = 'INTERFACE_BACKENDS'


def get_interface_backend_list(backends: list[str] | None = None) -> list[str]:
    """Backends for interface REST collection.

    Priority:
    1. explicit ``backends`` argument
    2. env ``INTERFACE_BACKENDS`` (``pytorch``, ``turbomind``, or comma/JSON list)
    3. ``BACKEND_LIST``
    """
    if backends is not None:
        return list(backends)
    raw = os.environ.get(INTERFACE_BACKENDS_ENV, '').strip()
    if not raw:
        return list(constant.BACKEND_LIST)
    if raw.startswith('['):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = json.loads(raw.replace("'", '"'))
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            raise ValueError(f'{INTERFACE_BACKENDS_ENV} must be a string list, got {raw!r}')
        selected = [str(x).strip() for x in value if str(x).strip()]
    else:
        selected = [part.strip() for part in raw.replace(';', ',').split(',') if part.strip()]
    unknown = [b for b in selected if b not in constant.BACKEND_LIST]
    if unknown:
        raise ValueError(
            f'{INTERFACE_BACKENDS_ENV} unknown backend(s) {unknown}; '
            f'expected subset of {constant.BACKEND_LIST}',
        )
    # Keep stable order from BACKEND_LIST
    return [b for b in constant.BACKEND_LIST if b in selected]



def _entry_engine_config(entry: dict[str, Any]) -> dict[str, Any]:
    """Per-model yaml engine / parallel block (``engine_config``; legacy
    ``parallel``)."""
    return entry.get(ENGINE_CONFIG_KEY) or entry.get('parallel') or {}


def _entry_has_prefix_cache_accuracy_tuning(entry: dict[str, Any]) -> bool:
    """True for yaml slices with explicit prefix-cache tuning knobs
    (evaluate)."""
    engine_extra = (_entry_engine_config(entry).get('extra') or {})
    return (
        'prefix-cache-decode-state-interval' in engine_extra
        and 'prefix-cache-state-budget' in engine_extra
    )


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
# Autotest-only keys in engine_config.extra (not forwarded to lmdeploy CLI).
CLI_SKIP_EXTRA_KEYS = frozenset()


def get_model_path_from_config(config: dict[str, Any], model_id: str) -> str:
    """Resolve ``model_id`` for lmdeploy / transformers."""
    if config['model_path_layout'] == 'hf_hub':
        return model_id
    return os.path.join(config['model_path'], model_id)


def get_model_work_path(config: dict[str, Any]) -> str:
    """Base directory for join-layout artifacts (e.g. quantized model
    output)."""
    if config['model_path_layout'] == 'hf_hub':
        return config['model_work_path']
    return config['model_path']


def resolve_extra_params(extra_params: dict[str, Any], config: dict[str, Any]) -> None:
    """Resolve relative model paths in extra_params."""
    model_path_keys = ['speculative-draft-model']

    for key in model_path_keys:
        if key in extra_params:
            value = extra_params[key]
            if value and isinstance(value, str) and not os.path.isabs(value):
                extra_params[key] = get_model_path_from_config(config, value)

    spec_cfg = extra_params.get('speculative_config')
    if isinstance(spec_cfg, dict) and 'model' in spec_cfg:
        model = spec_cfg['model']
        if model and isinstance(model, str) and not os.path.isabs(model):
            spec_cfg['model'] = get_model_path_from_config(config, model)


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
    if extra.get('enable-prefix-caching') is not None:
        if 'prefix_cache' not in funcs:
            return False
        # evaluate/infer accuracy: only dedicated yaml rows with tuned prefix-cache params
        if func_type == 'evaluate':
            return _entry_has_prefix_cache_accuracy_tuning(entry)
        return True
    if func_type == 'benchmark' and funcs == {'prefix_cache'}:
        return False
    if func_type == 'func':
        return 'func' in funcs
    return func_type in funcs


def _normalize_interface_suites(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, list):
        items = [str(x) for x in raw]
    else:
        raise TypeError(f'interface suites must be a list or str, got {type(raw).__name__}')
    unknown = [s for s in items if s not in INTERFACE_SUITES]
    if unknown:
        raise ValueError(f'unknown interface suite(s): {unknown}; expected {sorted(INTERFACE_SUITES)}')
    return [s for s in INTERFACE_SUITE_ORDER if s in items]


def _normalize_suites_extra_profile(raw: Any, *, default_suites: list[str] | None = None) -> dict[str, Any]:
    """Normalize one ``{suites, extra}`` launch profile."""
    if not isinstance(raw, dict):
        raise TypeError(f'interface profile must be {{suites, extra}}, got {type(raw).__name__}')
    extra = raw.get('extra') or {}
    if not isinstance(extra, dict):
        raise TypeError(f'interface.extra must be a dict, got {type(extra).__name__}')
    suites_raw = raw.get('suites')
    if suites_raw is None and default_suites is not None:
        suites = list(default_suites)
    else:
        suites = _normalize_interface_suites(suites_raw or [])
    return {'suites': suites, 'extra': copy.deepcopy(extra)}


def _interface_extra_key(extra: dict[str, Any] | None) -> tuple:
    """Stable key for comparing launch ``extra`` dicts (merge identical
    phases)."""
    return tuple(sorted((extra or {}).items(), key=lambda item: item[0]))


def _profiles_compatible_for_merge(
    suites_a: list[str],
    suites_b: list[str],
) -> bool:
    """Whether two same-``extra`` profiles can share one api_server phase.

    ``anthropic`` must not share a phase with ``toolcall``/``reasoning``: those
    suites need ``tool-call-parser`` / ``reasoning-parser`` in yaml ``extra``,
    which break anthropic Messages when present on the same api_server.
    """
    combined = set(suites_a) | set(suites_b)
    if 'anthropic' in combined and combined & {'toolcall', 'reasoning'}:
        return False
    return True


def _merge_profiles_same_extra(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse profiles that share identical ``extra`` into one launch
    phase."""
    buckets: list[dict[str, Any]] = []
    for profile in profiles:
        if not (profile.get('suites') or profile.get('extra')):
            continue
        ek = _interface_extra_key(profile.get('extra'))
        suites = list(profile.get('suites') or [])
        found = None
        for bucket in buckets:
            if bucket['ek'] != ek:
                continue
            if not _profiles_compatible_for_merge(bucket['suites'], suites):
                continue
            found = bucket
            break
        if found is None:
            buckets.append({
                'ek': ek,
                'extra': copy.deepcopy(profile.get('extra') or {}),
                'suites': suites,
            })
            continue
        for suite in suites:
            if suite not in found['suites']:
                found['suites'].append(suite)
    merged: list[dict[str, Any]] = []
    for bucket in buckets:
        ordered = [s for s in INTERFACE_SUITE_ORDER if s in bucket['suites']]
        ordered += [s for s in bucket['suites'] if s not in INTERFACE_SUITE_ORDER]
        merged.append({'suites': ordered, 'extra': bucket['extra']})
    return merged


def _backend_cfg_from_profiles(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    """Build backend cfg from ordered launch profiles.

    Each distinct ``extra`` is one api_server phase. Profiles with identical
    ``extra`` are merged (same launch command → one restart). ``suites`` on the
    cfg is the ordered union across profiles; ``extra`` is the first profile's
    extra (convenience for single-phase callers).
    """
    cleaned = _merge_profiles_same_extra(profiles)
    suites_union: list[str] = []
    for profile in cleaned:
        for suite in profile.get('suites') or []:
            if suite not in suites_union:
                suites_union.append(suite)
    suites_union = [s for s in INTERFACE_SUITE_ORDER if s in suites_union]
    return {
        'profiles': cleaned,
        'suites': suites_union,
        'extra': copy.deepcopy(cleaned[0]['extra']) if cleaned else {},
    }


def _normalize_interface_backend_cfg(raw: Any) -> dict[str, Any]:
    """Normalize one backend's interface block to ``{profiles, suites,
    extra}``.

    Preferred form — list of launch profiles (each ``{suites, extra}``):

    .. code-block:: yaml

        pytorch:
        - suites: [base, logprob, experts, toolcall, reasoning]
          extra:
            tool-call-parser: qwen3coder
            reasoning-parser: default
        - suites: [anthropic]
          extra:
            logprobs-mode: raw_logprobs

    Also accepted:
    - suite list shorthand: ``pytorch: [base, logprob]``
    - single dict: ``{suites, extra}``
    - legacy nested ``anthropic: {extra: {...}}`` on a single dict
    """
    # List of profiles: [{suites, extra}, ...]
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and (
            'suites' in raw[0] or 'extra' in raw[0]):
        profiles = [_normalize_suites_extra_profile(item) for item in raw]
        return _backend_cfg_from_profiles(profiles)

    if isinstance(raw, (list, str)):
        suites = _normalize_interface_suites(raw)
        profiles: list[dict[str, Any]] = []
        # Shorthand has no per-suite extra: split anthropic away from
        # toolcall/reasoning so a later yaml-less merge cannot share a phase
        # that must carry parsers (set explicitly in profile ``extra``).
        # Otherwise keep one shared phase.
        needs_split = (
            'anthropic' in suites
            and bool(set(suites) & {'toolcall', 'reasoning'})
        )
        if needs_split:
            other = [s for s in suites if s != 'anthropic']
            if other:
                profiles.append({'suites': other, 'extra': {}})
            profiles.append({'suites': ['anthropic'], 'extra': {}})
        elif suites:
            profiles.append({'suites': suites, 'extra': {}})
        return _backend_cfg_from_profiles(profiles)

    if not isinstance(raw, dict):
        raise TypeError(
            f'interface backend config must be a suite list, profile list, or '
            f'{{suites, extra}} dict, got {type(raw).__name__}',
        )

    # Legacy nested anthropic key on a single profile dict.
    if 'anthropic' in raw and ('suites' in raw or 'extra' in raw):
        main = _normalize_suites_extra_profile(
            {'suites': raw.get('suites') or [], 'extra': raw.get('extra') or {}},
        )
        anth_raw = raw.get('anthropic')
        if anth_raw is True or anth_raw == {} or anth_raw is None:
            anth = {'suites': ['anthropic'], 'extra': {}}
        elif isinstance(anth_raw, dict):
            anth = _normalize_suites_extra_profile(anth_raw, default_suites=['anthropic'])
            anth['suites'] = ['anthropic']
        else:
            raise TypeError('legacy interface.anthropic must be a dict or true')
        main['suites'] = [s for s in main['suites'] if s != 'anthropic']
        profiles = []
        if main['suites'] or main['extra']:
            profiles.append(main)
        profiles.append(anth)
        return _backend_cfg_from_profiles(profiles)

    if 'suites' in raw or 'extra' in raw:
        # Explicit {suites, extra}: keep as written (including anthropic in the
        # same suites list when launch extras match). Split only via a profile
        # list or legacy nested anthropic when extras differ.
        main = _normalize_suites_extra_profile(raw)
        return _backend_cfg_from_profiles([main] if (main['suites'] or main['extra']) else [])

    raise TypeError(
        'interface backend config must be a suite list, a list of '
        '{suites, extra} profiles, or a single {suites, extra} dict',
    )


def _normalize_interface_map(raw: Any) -> dict[str, dict[str, Any]]:
    """Normalize sibling ``interface`` to ``{backend: {profiles, suites,
    extra}}``.

    Supported forms:
    - ``interface: [base, logprob]``
    - ``interface: {pytorch: [base, logprob], turbomind: [base]}``
    - ``interface: {pytorch: {suites: [...], extra: {...}}}``
    - ``interface: {pytorch: [{suites, extra}, {suites: [anthropic], extra}]}``
    """
    if not raw:
        return {}
    if isinstance(raw, list):
        cfg = _normalize_interface_backend_cfg(raw)
        if cfg['profiles'] or cfg['suites'] or cfg['extra']:
            return {'*': cfg}
        return {}
    if not isinstance(raw, dict):
        raise TypeError(f'interface must be a dict or list, got {type(raw).__name__}')
    out: dict[str, dict[str, Any]] = {}
    for backend, value in raw.items():
        cfg = _normalize_interface_backend_cfg(value)
        if cfg['profiles'] or cfg['suites'] or cfg['extra']:
            out[str(backend)] = cfg
    return out


def get_interface_backend_config(entry: dict[str, Any], backend: str) -> dict[str, Any]:
    """Return ``{profiles, suites, extra}`` for ``backend`` (empty if
    unset)."""
    mapping = _normalize_interface_map(entry.get(INTERFACE_KEY))
    empty = {'profiles': [], 'suites': [], 'extra': {}}
    if not mapping:
        return empty
    if backend in mapping:
        return copy.deepcopy(mapping[backend])
    if '*' in mapping:
        return copy.deepcopy(mapping['*'])
    return empty


def get_interface_suites(entry: dict[str, Any], backend: str) -> list[str]:
    """Return union of interface suites enabled for ``backend``."""
    return list(get_interface_backend_config(entry, backend).get('suites') or [])


def get_interface_profiles(entry: dict[str, Any], backend: str) -> list[dict[str, Any]]:
    """Return ordered ``[{suites, extra}, ...]`` launch profiles for
    ``backend``."""
    cfg = get_interface_backend_config(entry, backend)
    profiles = cfg.get('profiles')
    if profiles:
        return copy.deepcopy(profiles)
    suites = list(cfg.get('suites') or [])
    extra = copy.deepcopy(cfg.get('extra') or {})
    if suites or extra:
        return [{'suites': suites, 'extra': extra}]
    return []


def _suite_launch_extra_defaults(suites: list[str] | set[str]) -> dict[str, Any]:
    """Suite → recommended launch keys (filled only when configs omit them).

    ``tool-call-parser`` / ``reasoning-parser`` are never defaulted — set them in
    each interface profile's ``extra`` in yaml.
    """
    suite_set = set(suites)
    defaults: dict[str, Any] = {}
    if suite_set & {'logprob', 'experts'}:
        defaults['logprobs-mode'] = 'raw_logprobs'
    if suite_set & {'experts'}:
        defaults['enable-return-routed-experts'] = True
    return defaults


def build_interface_launch_extra(
    entry: dict[str, Any],
    backend: str,
    suites: list[str] | set[str] | None = None,
    model_path: str | None = None,
    *,
    interface_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build launch ``extra`` dict for interface ``api_server``.

    Same structure as tools/pipeline ``engine_config.extra``. Merge order
    (later wins):

    1. suite defaults (``logprobs-mode``, ``enable-return-routed-experts`` only)
    2. ``engine_config.extra`` (shared row launch)
    3. ``interface_extra`` if provided, else ``interface.<backend>.extra``

    ``tool-call-parser`` / ``reasoning-parser`` must come from yaml (or
    ``engine_config.extra``); they are not inferred from suites / model path.
    """
    iface = get_interface_backend_config(entry, backend)
    suite_list = list(suites) if suites is not None else list(iface.get('suites') or [])
    extra_src = iface.get('extra') or {} if interface_extra is None else interface_extra
    merged: dict[str, Any] = {}
    merged.update(_suite_launch_extra_defaults(suite_list))
    merged.update(_parallel_launch_extra(_entry_engine_config(entry)))
    merged.update(copy.deepcopy(extra_src or {}))
    return merged


ROUTED_EXPERTS_UNSUPPORTED_SKIP = (
    'return_routed_experts not enabled in model interface config '
    '(add experts suite or enable-return-routed-experts: true in yaml)')


def iter_model_yaml_entries(model_id: str) -> list[dict[str, Any]]:
    """All matrix rows for *model_id* under the active ``TEST_ENV``."""
    env_key = _resolve_paths_env_key(os.environ.get('TEST_ENV'))
    return [entry for mid, entry in _iter_per_model_entries(env_key) if mid == model_id]


def model_enables_return_routed_experts(
    model_id: str,
    backend: str,
    *,
    required_suites: set[str] | frozenset[str] | None = None,
) -> bool:
    """True when yaml interface launch extra enables ``return_routed_experts``.

    When *required_suites* is set (e.g. ``{'toolcall'}`` or ``{'experts'}``),
    only matching interface profiles are considered.
    """
    if backend == 'turbomind':
        return False
    for entry in iter_model_yaml_entries(model_id):
        for prof in get_interface_profiles(entry, backend):
            suites = set(prof.get('suites') or [])
            if required_suites and not (required_suites & suites):
                continue
            extra = build_interface_launch_extra(
                entry,
                backend,
                suites=prof['suites'],
                interface_extra=prof.get('extra'),
            )
            if extra.get('enable-return-routed-experts'):
                return True
    return False


def derive_interface_server_extra(
    suites: list[str] | set[str],
    model_path: str | None = None,
    entry: dict[str, Any] | None = None,
    backend: str | None = None,
    *,
    interface_extra: dict[str, Any] | None = None,
) -> str:
    """Format interface launch flags via ``get_cli_str`` (same as tools)."""
    entry = entry or {}
    backend = backend or 'pytorch'
    extra = build_interface_launch_extra(
        entry,
        backend,
        suites=suites,
        model_path=model_path,
        interface_extra=interface_extra,
    )
    return get_cli_str(extra).strip()


def derive_interface_case_info(profiles: list[str], suites: list[str] | set[str]) -> list[str]:
    """Derive REST case groups from model profiles + interface suites.

    Directory-based suites (toolcall / reasoning) and anthropic protocol files are selected by path in CI; generate
    logprob/experts stay in one file and are filtered by pytest marks.
    """
    suite_set = set(suites)
    case_info: list[str] = []
    is_base_only = 'base' in profiles and not ({'chat', 'vl'} & set(profiles))
    if is_base_only:
        if suite_set & GENERATE_SUITES:
            case_info.append('completions_v1')
    else:
        if suite_set & GENERATE_SUITES:
            case_info.append('chat_completions_v1')
            case_info.append('generate')
    if 'anthropic' in suite_set:
        case_info.append('anthropic_v1')
        case_info.append('anthropic_sdk')
    if 'toolcall' in suite_set:
        case_info.append('toolcall')
    if 'reasoning' in suite_set:
        case_info.append('reasoning')
    return case_info


def derive_generate_marker(suites: list[str] | set[str], backend: str) -> str:
    """Pytest ``-m`` expression for ``test_restful_generate.py``.

    Prefer marks over splitting generate into multiple files: only a few tests
    are logprob/experts-specific, and one server process still covers the union.
    """
    suite_set = set(suites)
    parts = [f'not not_{backend}']
    if 'experts' in suite_set:
        return ' and '.join(parts)
    if 'logprob' in suite_set:
        parts.append('not experts')
        return ' and '.join(parts)
    # base-only generate coverage
    parts.extend(['not logprob', 'not experts'])
    return ' and '.join(parts)


def get_interface_matrix(
    env_key: str | None = None,
    backends: list[str] | None = None,
    deps_profile: DepsProfileSelector | None = 'all',
) -> list[dict[str, Any]]:
    """Build flat REST interface matrix rows from per-model ``interface``
    config.

    Each row: model, model_path, tp, backend, suites, case_info, extra,
    generate_marker. ``extra`` is CLI text from the same ``extra`` dict shape
    used by tools (``get_cli_str``).

    Dedup key includes parallel layout so the same model/backend can appear
    once per ``engine_config`` (e.g. tp16 vs dp/ep16).
    """
    config = get_config()
    matrix_env = env_key or _model_matrix_env_key(config)
    backend_filter = get_interface_backend_list(backends)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, tuple[tuple[str, int], ...]]] = set()

    for model_id, entry in _iter_per_model_entries(matrix_env, deps_profile=deps_profile):
        iface_map = _normalize_interface_map(entry.get(INTERFACE_KEY))
        if not iface_map:
            continue
        profiles = _normalize_profiles(entry.get('model_type', 'chat'))
        layout = _parallel_layout(_entry_engine_config(entry))
        tp = int(layout.get('tp', 1))
        layout_key = tuple(sorted((k, int(layout[k])) for k in PARALLEL_LAYOUT_KEYS if k in layout))
        model_name = model_id.split('/')[-1]

        target_backends: list[str]
        if '*' in iface_map:
            target_backends = list(backend_filter)
        else:
            target_backends = [b for b in backend_filter if b in iface_map]

        for backend in target_backends:
            iface_cfg = get_interface_backend_config(entry, backend)
            launch_profiles = get_interface_profiles(entry, backend)
            suites = list(iface_cfg.get('suites') or [])
            if not launch_profiles:
                continue
            key = (model_id, backend, layout_key)
            if key in seen:
                continue
            seen.add(key)
            case_info: list[str] = []
            phase_extras: list[str] = []
            for prof in launch_profiles:
                case_info.extend(derive_interface_case_info(profiles, prof['suites']))
                phase_extras.append(
                    derive_interface_server_extra(
                        prof['suites'],
                        model_path=model_id,
                        entry=entry,
                        backend=backend,
                        interface_extra=prof.get('extra') or {},
                    ))
            rows.append({
                'model': model_name,
                'model_path': model_id,
                'tp': tp,
                'backend': backend,
                'suites': suites,
                'case_info': case_info,
                'extra': phase_extras[0] if phase_extras else '',
                'phase_extras': phase_extras,
                'generate_marker': derive_generate_marker(suites, backend),
            })

    rows.sort(key=lambda r: (r['model_path'], r['backend'], r['tp']))
    return rows


def get_interface_run_config_list(
    backend: str,
    parallel_config: dict[str, int],
    model_types: tuple[str, ...] | list[str] = ('chat', 'base'),
    deps_profile: DepsProfileSelector | None = None,
) -> list[dict[str, Any]]:
    """Build ``run_config`` rows for interface REST tests (tools-style).

    Only models with an ``interface`` block for ``backend`` are included.
    Launch extras use the same dict shape as ``engine_config.extra`` via
    :func:`build_interface_launch_extra`.
    """
    config = get_config()
    matrix_env = _model_matrix_env_key(config)
    wanted = {t.replace('_model', '') for t in model_types}
    rows: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    profile = deps_profile if deps_profile is not None else 'all'

    for model_id, entry in _iter_per_model_entries(matrix_env, deps_profile=profile):
        launch_profiles = get_interface_profiles(entry, backend)
        if not launch_profiles:
            continue
        model_profiles = set(_normalize_profiles(entry.get('model_type', 'chat')))
        if not (model_profiles & wanted):
            continue
        layout = _parallel_layout(_entry_engine_config(entry))
        if not _parallel_dicts_equal(layout, parallel_config):
            continue
        backend_map = _normalize_entry_backends(entry, config, layout)
        communicators = backend_map.get(backend) or ['nccl']
        communicator = communicators[0]
        suites = get_interface_suites(entry, backend)
        phase_key = tuple(
            (tuple(p.get('suites') or []), tuple(sorted((p.get('extra') or {}).items())))
            for p in launch_profiles
        )
        key = (model_id, backend, communicator, tuple(sorted(layout.items())), phase_key)
        if key in seen:
            continue
        seen.add(key)

        interface_phases: list[dict[str, Any]] = []
        all_case_info: list[str] = []
        for prof in launch_profiles:
            case_info = derive_interface_case_info(list(model_profiles), prof['suites'])
            if not case_info:
                continue
            extra_params = build_interface_launch_extra(
                entry,
                backend,
                suites=prof['suites'],
                model_path=model_id,
                interface_extra=prof.get('extra') or {},
            )
            interface_phases.append({
                'suites': list(prof['suites']),
                'case_info': case_info,
                'extra_params': extra_params,
            })
            all_case_info.extend(case_info)

        if not interface_phases:
            continue

        rows.append({
            'model': model_id,
            'backend': backend,
            'communicator': communicator,
            'quant_policy': 0,
            'parallel_config': copy.deepcopy(layout),
            'extra_params': copy.deepcopy(interface_phases[0]['extra_params']),
            'interface_suites': list(suites),
            'case_info': all_case_info,
            'interface_phases': interface_phases,
            'generate_marker': derive_generate_marker(suites, backend),
        })

    return rows


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


def _is_fp8_enabled_in_entry(
    backend: str,
    quant_cfg: dict[str, list[str]],
) -> bool:
    """True when per-model ``quantization.<backend>`` includes runtime
    ``fp8``."""
    enabled = set(quant_cfg.get(backend) or [])
    return 'fp8' in enabled


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
    """Expand run configs from autotest/configs/<org>/<model>.yml entries.

    Honors ``DEPS_PROFILE`` via :func:`_iter_per_model_entries`. Do not gate
    models with :func:`get_model_list` (that helper always ignores deps-pinned
    rows); intersecting them would empty ``tp1``/``tp2`` cases under pinned
    profiles and show pytest ``[NOTSET]``.
    """
    extra = extra or {}
    env_key = _model_matrix_env_key(config)
    deps_profile = get_deps_profile_selector()
    run_configs: list[dict[str, Any]] = []
    seen: set[tuple] = set()

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
        launch_extra_sig = _entry_launch_extra_sig(entry)

        for model in models_for_quant:
            qcfg = quant_cfg
            for quant_policy in [0, 4, 8, 42]:
                if not _is_kvint_enabled_in_entry(backend, _base_model_name(model), quant_policy, qcfg):
                    continue
                for communicator in backend_map[backend]:
                    sig = (model, communicator, quant_policy, launch_extra_sig, '')
                    if sig in seen:
                        continue
                    seen.add(sig)
                    run_config = _build_run_config_entry(
                        model,
                        entry,
                        backend,
                        communicator,
                        parallel_config,
                        quant_policy,
                        config,
                        func_type,
                        extra,
                    )
                    run_configs.append(run_config)

                    # Runtime fp8 (--model-format fp8) is a separate case at
                    # quant_policy 0, mirroring legacy flat-yaml fp8_model_list.
                    if (
                        quant_policy == 0
                        and _is_fp8_enabled_in_entry(backend, qcfg)
                        and 'fp8' not in model.lower()
                        and run_config.get('extra_params', {}).get('model-format') is None
                    ):
                        fp8_sig = (model, communicator, quant_policy, launch_extra_sig, 'fp8')
                        if fp8_sig not in seen:
                            seen.add(fp8_sig)
                            fp8_config = copy.deepcopy(run_config)
                            fp8_config['extra_params']['model-format'] = 'fp8'
                            run_configs.append(fp8_config)
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
        norm_key = key.replace('_', '-')
        if norm_key in CLI_SKIP_EXTRA_KEYS:
            continue
        key = norm_key
        # ``null`` / ``true`` → bare ``--flag`` (argparse store_true).
        if value is None or value is True:
            cli_str.append(f'--{key}')
        elif value is False:
            continue
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
    if extra and extra.get('enable-prefix-caching') is not None:
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


def _apply_hf_hub_env(config: dict[str, Any]) -> None:
    """Point Hugging Face hub at the H-card cache (offline)."""
    if config['model_path_layout'] != 'hf_hub':
        return
    os.environ['HF_HUB_CACHE'] = config['model_path']
    os.environ['HF_HUB_OFFLINE'] = '1'


def get_config() -> dict[str, Any]:
    """Load global paths from ``autotest/env_paths.yml``; model matrices from
    ``configs/**``."""
    if not _per_model_configs_available():
        raise FileNotFoundError(
            f'Per-model autotest configs required: missing {PATHS_YML} or {CONFIGS_DIR}',
        )
    paths_key = _resolve_paths_env_key(os.environ.get('TEST_ENV'))
    config_copy = _load_paths_for_env(paths_key)
    _apply_hf_hub_env(config_copy)
    _apply_run_id_paths(config_copy)
    return config_copy


def get_gpus_per_instance(parallel_config: dict[str, int] | None) -> int:
    """GPU count for one api_server instance (align with launch_server dp
    layout)."""
    parallel_config = parallel_config or {}
    dp = parallel_config.get('dp', 1)
    tp = parallel_config.get('tp', 1)
    ep = parallel_config.get('ep', 1)
    return max(dp, tp, ep)


def get_cuda_prefix_by_workerid(worker_id: str | None, parallel_config: dict[str, int] | None = None) -> str | None:
    """Get cuda/ascend visible devices env prefix by worker id & parallel
    config."""
    para_conf = parallel_config or {}
    device_type = os.environ.get('DEVICE', 'cuda')

    gpus_per_instance = get_gpus_per_instance(para_conf)
    if gpus_per_instance <= 0:
        return ''

    cuda_id = get_cuda_id_by_workerid(worker_id, gpus_per_instance)
    if not cuda_id:
        return ''

    return f'ASCEND_RT_VISIBLE_DEVICES={cuda_id}' if device_type == 'ascend' else f'CUDA_VISIBLE_DEVICES={cuda_id}'


def get_cuda_id_by_workerid(worker_id: str | None, gpus_per_instance: int = 1) -> str | None:
    """Get cuda id str by worker id and GPUs per instance, return None if
    invalid worker id."""
    if worker_id is None or 'gw' not in worker_id:
        return None

    base_id = int(worker_id.replace('gw', ''))
    cuda_num = base_id * gpus_per_instance
    return ','.join([str(cuda_num + i) for i in range(gpus_per_instance)])


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

    gpus_per_instance = 1
    if parallel_config is not None:
        if isinstance(parallel_config, int):
            gpus_per_instance = parallel_config
        elif isinstance(parallel_config, dict):
            gpus_per_instance = get_gpus_per_instance(parallel_config)

    if device == 'ascend':
        device_id = get_cuda_id_by_workerid(worker_id, gpus_per_instance)
        if device_id is not None:
            os.environ['ASCEND_RT_VISIBLE_DEVICES'] = device_id
    else:
        cuda_id = get_cuda_id_by_workerid(worker_id, gpus_per_instance)
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
    'top_p',
    'top_k',
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


_VLMEVALKIT_GEN_KEYS = frozenset({
    'temperature',
    'top-k',
    'top-p',
    'repetition-penalty',
})


def _gen_config_to_vlmevalkit_kwargs(gen: dict[str, Any]) -> dict[str, Any]:
    """Map per-model yaml ``gen_config`` to VLMEvalKit ``run.py`` CLI keys."""
    return {
        key: value
        for key, value in gen.items()
        if key.replace('_', '-') in _VLMEVALKIT_GEN_KEYS
    }


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
            merged.update(_gen_config_to_vlmevalkit_kwargs(run_config['gen_config']))
        return merged

    # Base TurboMindAPIModel: keep scalar sampling fields, skip OpenAISDK mapping.
    if name == 'base' or name.startswith('base-'):
        return _eval_table_scalar_params(preset) or copy.deepcopy(preset)

    if run_config.get('gen_config'):
        result = _eval_table_scalar_params(preset)
        result.update(_gen_config_to_opencompass_kwargs(run_config['gen_config']))
        return result

    return copy.deepcopy(preset)


def _is_prefix_cache_run(extra_params: dict[str, Any]) -> bool:
    """True when run config enables prefix caching (distinct case / result
    dir).

    Value may be ``None`` (CLI flag without argument) after
    :func:`_build_run_config_entry` normalizes ``True`` → ``None``.
    """
    return 'enable-prefix-caching' in extra_params


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
    spec_algo = extra_params.get('speculative-algorithm')
    if spec_algo:
        extra_params_case += f'_{spec_algo}'.replace('_', '-')
    model_format = extra_params.get('model-format')
    if model_format:
        extra_params_case += f'_{model_format}'
    if _is_prefix_cache_run(extra_params):
        extra_params_case += '_prefix-cache'
    if not is_simple:
        for k, v in extra_params.items():
            if len(v) > 10:
                extra_params_case += f'_{k}'.replace('_', '-').replace('/', '-').replace('.', '-')
            else:
                extra_params_case += f'_{k}{v}'.replace('_', '-').replace('/', '-').replace('.', '-')

    return f'{backend_type}_{pure_model_name}_{communicator}_{parallel_str}_{quant_policy}{extra_params_case}'


def _format_case_variant_label(variant_suffix: str) -> str:
    """Human-readable variant tags (MTP / fp8 / prefix-cache) from case
    suffix."""
    if not variant_suffix:
        return '-'
    label = variant_suffix.lstrip('-_').replace('_', ' ').strip()
    return label or '-'


def parse_config_by_case(case_str: str) -> dict[str, Any]:
    """Parse run config dict from case name string."""
    case_parts = case_str.split('_')
    if len(case_parts) < 4:
        raise ValueError(f'Invalid case string: {case_str}')

    backend = case_parts[0]
    model = case_parts[1]
    communicator = case_parts[2]

    quant_idx = None
    quant_policy = 0
    variant_suffix = ''
    for i in range(len(case_parts) - 1, 2, -1):
        match = re.match(r'^(\d+)(.*)$', case_parts[i])
        if match:
            quant_idx = i
            quant_policy = int(match.group(1))
            variant_suffix = match.group(2)
            break
    if quant_idx is None:
        raise ValueError(f'No numeric quant policy found in case string: {case_str}')

    if quant_idx + 1 < len(case_parts):
        tail = '_'.join(case_parts[quant_idx + 1:])
        variant_suffix = f'{variant_suffix}_{tail}' if variant_suffix else tail

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
        'variant': _format_case_variant_label(variant_suffix),
    }
