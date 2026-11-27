# Copyright (c) Shanghai AI Laboratory. All rights reserved.
import contextlib
import os

from lmdeploy.utils import get_logger


def env_to_bool(
    env_var: str,
    default: bool = False,
    *,
    true_values: set | list = {'true', '1', 'yes', 'on'},
    false_values: set | list = {'false', '0', 'no', 'off'},
):
    """Env to bool."""
    value = os.getenv(env_var)
    if value is None:
        return default
    value = value.lower().strip()
    if value in true_values:
        return True
    elif value in false_values:
        return False
    else:
        raise ValueError(f"Cannot convert environment variable '{env_var}={value}' to boolean. "
                         f'Allowed true values: {true_values}, false values: {false_values}')


def env_to_int(
    env_var: str,
    default: int = 0,
):
    """Env to int."""
    value = os.getenv(env_var)
    if value is None:
        return default
    try:
        value = int(value)
    except Exception:
        value = default
    return value


def env_to_list_int(
    env_var: str,
    default: list[int] = None,
):
    """Env to list of int."""
    default_ = default if default is not None else []
    value = os.getenv(env_var)
    if value is None:
        return default_
    try:
        value = [int(x) for x in value.split(',')]
    except Exception:
        value = default_
    return value


def env_to_float(
    env_var: str,
    default: float = 0,
):
    """Env to float."""
    value = os.getenv(env_var)
    if value is None:
        return default
    try:
        value = float(value)
    except Exception:
        value = default
    return value


def env_to_choice(
    env_var: str,
    default: str,
    choices: set | list,
):
    """Env to selected string."""
    value = os.getenv(env_var)
    if value is None:
        return default
    value = value.lower().strip()
    if value not in choices:
        raise ValueError(f"Invalid environment variable '{env_var}={value}'. Allowed values: {choices}")
    return value


_ENVS = dict()


@contextlib.contextmanager
def set_envs():
    _origin_get_env = os.getenv

    def _patched_get_env(
        env_var: str,
        default: str | None = None,
    ):
        """Patched get_env."""
        if env_var in os.environ:
            _ENVS[env_var] = os.environ[env_var]

        return _origin_get_env(env_var, default)

    os.getenv = _patched_get_env
    try:
        yield
    finally:
        os.getenv = _origin_get_env


# ---------------------------------------------------------------------------
# Unified TurboMind kernel switch
#
#   LMDEPLOY_TM_KERNELS = off | auto | all | <class>=<state>[,...]
#
# One variable carries every grafted kernel class. A global token sets all
# classes; a class entry overrides the global value for that class. States:
# auto probes the capability and falls back, on forces the route and fails
# loudly, off keeps the native implementation. When the big env leaves a
# class open, the class legacy env decides (backward compat); a conflict
# warns once and the big env wins.
# ---------------------------------------------------------------------------
_TM_KERNEL_CLASS_LEGACY_ENVS = {
    'w4a16': 'LMDEPLOY_W4A16_GEMM_BACKEND',
    'moe_grouped': 'LMDEPLOY_W4A16_GEMM_BACKEND',
    # NVFP4 W4A16 lane (ct nvfp4-pack-quantized checkpoints; SM80 family).
    # No legacy env: the class defaults to auto (capability-based).
    'nvfp4': None,
}
_TM_KERNEL_CLASSES = tuple(_TM_KERNEL_CLASS_LEGACY_ENVS)
_TM_KERNEL_STATES = ('auto', 'on', 'off')
# Global tokens and the per-class state each one means.
_TM_GLOBAL_TOKENS = {'off': 'off', 'auto': 'auto', 'all': 'on'}
# Legacy provider names of LMDEPLOY_W4A16_GEMM_BACKEND map onto the same
# states; any other value kept the native route historically and maps to off.
_TM_LEGACY_PROVIDER_STATES = {'auto': 'auto', 'turbomind': 'on'}
# Resolved state -> legacy provider name. Readers that still consume the
# provider attribute (nn/linear gating, tm_gemm_modules) follow the unified
# decision through this mapping without code changes.
_TM_STATE_PROVIDERS = {'auto': 'auto', 'on': 'turbomind', 'off': 'triton'}

logger = get_logger('lmdeploy')

# One warning per (class, big-env state, legacy value) conflict.
_tm_kernel_conflict_warned = set()


def _tm_legacy_env_state(legacy_env, raw):
    """Map one raw legacy env value onto the three states."""
    if legacy_env == 'LMDEPLOY_W4A16_GEMM_BACKEND':
        return _TM_LEGACY_PROVIDER_STATES.get(raw.strip().lower(), 'off')
    state = raw.strip().lower()
    return state if state in _TM_KERNEL_STATES else 'auto'


def parse_tm_kernels(env_value=None):
    """Parse LMDEPLOY_TM_KERNELS into per-class explicit states.

    Returns one entry per known class: the state the big env sets for it
    (global token filled in, class entry overrides), or None when the
    class stays open for its legacy env. Parsing: ';' separates entries
    too, class names match
    exactly, the first entry that names a class decides it, unknown
    classes stay silent for forward compatibility, and an invalid state
    for a known class warns and defers that class to the global token (or
    the legacy env when no global token is set).
    """
    if env_value is None:
        env_value = os.getenv('LMDEPLOY_TM_KERNELS', '')
    global_state = None
    explicit = dict.fromkeys(_TM_KERNEL_CLASSES)
    decided = set()
    for item in env_value.replace(';', ',').split(','):
        item = item.strip()
        if not item:
            continue
        name, sep, state = item.partition('=')
        if not sep:
            token = item.lower()
            if token not in _TM_GLOBAL_TOKENS:
                logger.warning('Ignoring invalid LMDEPLOY_TM_KERNELS value %s', item)
                continue
            if global_state is None:
                global_state = _TM_GLOBAL_TOKENS[token]
            continue
        name = name.strip()
        if name not in explicit or name in decided:
            continue
        state = state.strip().lower()
        if state not in _TM_KERNEL_STATES:
            logger.warning('Ignoring invalid LMDEPLOY_TM_KERNELS entry %s=%s', name, state)
            decided.add(name)
            continue
        explicit[name] = state
        decided.add(name)
    if global_state is not None:
        for name, state in explicit.items():
            if state is None:
                explicit[name] = global_state
    return explicit


def _tm_kernel_legacy_state(cls):
    """State the legacy env of a class asks for.

    w4a16 and moe_grouped share LMDEPLOY_W4A16_GEMM_BACKEND. The lookup goes through the module attribute, the value the
    consumers and the routing tests read. Classes without a legacy env default to auto.
    """
    legacy_env = _TM_KERNEL_CLASS_LEGACY_ENVS[cls]
    if legacy_env is None:
        return 'auto'
    if legacy_env == 'LMDEPLOY_W4A16_GEMM_BACKEND':
        return _tm_legacy_env_state(legacy_env, w4a16_gemm_backend)
    return _tm_legacy_env_state(legacy_env, os.getenv(legacy_env, 'auto'))


def _warn_tm_kernel_conflict(cls, state, legacy_env, legacy_raw):
    """Warn once when the big env and the legacy env disagree on a class."""
    key = (cls, state, legacy_raw.strip())
    if key in _tm_kernel_conflict_warned:
        return
    _tm_kernel_conflict_warned.add(key)
    logger.warning('LMDEPLOY_TM_KERNELS sets %s=%s; legacy %s=%s is overridden.',
                   cls, state, legacy_env, legacy_raw)


def tm_kernel_state(cls):
    """Return the resolved state of one kernel class: auto, on, or off.

    Priority: the LMDEPLOY_TM_KERNELS class entry, the global big-env
    token, the class legacy env, then auto. An explicit big-env state
    that disagrees with an explicitly set legacy env warns once; the big
    env wins.
    """
    if cls not in _TM_KERNEL_CLASS_LEGACY_ENVS:
        raise ValueError(f'Unknown TM kernel class {cls!r}. '
                         f'Known classes: {list(_TM_KERNEL_CLASS_LEGACY_ENVS)}')
    state = parse_tm_kernels()[cls]
    if state is None:
        return _tm_kernel_legacy_state(cls)
    legacy_env = _TM_KERNEL_CLASS_LEGACY_ENVS[cls]
    legacy_raw = os.getenv(legacy_env) if legacy_env is not None else None
    if legacy_raw is not None:
        legacy_state = _tm_legacy_env_state(legacy_env, legacy_raw)
        if legacy_state != state:
            _warn_tm_kernel_conflict(cls, state, legacy_env, legacy_raw.strip())
    return state


with set_envs():
    # loader
    random_load_weight = env_to_bool('LMDEPLOY_RANDOM_LOAD_WEIGHT', True)

    # profile
    ray_nsys_enable = env_to_bool('LMDEPLOY_RAY_NSYS_ENABLE', False)
    ray_nsys_output_prefix = os.getenv('LMDEPLOY_RAY_NSYS_OUT_PREFIX', None)

    # ascend
    ascend_set_rt_visable_devices_by_ray = env_to_bool('ASCEND_SET_RT_VISIBLE_DEVICES_BY_RAY', False)

    # dp
    dp_master_addr = os.getenv('LMDEPLOY_DP_MASTER_ADDR', None)
    dp_master_port = os.getenv('LMDEPLOY_DP_MASTER_PORT', None)

    # executor
    executor_backend = os.getenv('LMDEPLOY_EXECUTOR_BACKEND', None)

    # torch profiler
    torch_profile_cpu = env_to_bool('LMDEPLOY_PROFILE_CPU', False)
    torch_profile_cuda = env_to_bool('LMDEPLOY_PROFILE_CUDA', False)
    torch_profile_delay = env_to_int('LMDEPLOY_PROFILE_DELAY', 0)
    torch_profile_duration = env_to_int('LMDEPLOY_PROFILE_DURATION', -1)
    torch_profile_output_prefix = os.getenv('LMDEPLOY_PROFILE_OUT_PREFIX', 'lmdeploy_profile_')
    torch_profile_use_gzip = env_to_bool('LMDEPLOY_PROFILE_USE_GZIP', True)

    # ray timeline
    ray_timeline_enable = env_to_bool('LMDEPLOY_RAY_TIMELINE_ENABLE', False)
    ray_timeline_output_path = os.getenv('LMDEPLOY_RAY_TIMELINE_OUT_PATH', 'ray_timeline.json')

    # ray external placement group bundles
    # only used when lmdeploy is initialized inside a Ray Actor with pg allocated
    ray_external_pg_bundles = env_to_list_int('LMDEPLOY_RAY_EXTERNAL_PG_BUNDLES', [])

    # enable ray zero-copy tensors
    os.getenv('RAY_ENABLE_ZERO_COPY_TORCH_TENSORS', '1')

    # dist
    dist_master_addr = os.getenv('LMDEPLOY_DIST_MASTER_ADDR', None)
    dist_master_port = os.getenv('LMDEPLOY_DIST_MASTER_PORT', None)

    # logging
    log_file = os.getenv('LMDEPLOY_LOG_FILE', None)
    os.getenv('LMDEPLOY_LOG_PID', '0')
    enable_request_cache_usage_metric = env_to_bool('LMDEPLOY_ENABLE_REQUEST_CACHE_USAGE_METRIC', True)

    # check env
    enable_check_env = env_to_bool('LMDEPLOY_ENABLE_CHECK_ENV', True)

    # hccl / ascend - passed to ray workers
    os.getenv('HCCL_BUFFSIZE', None)
    os.getenv('HCCL_CONNECT_TIMEOUT', None)
    os.getenv('HCCL_OP_EXPANSION_MODE', None)
    os.getenv('HCCL_IF_IP', None)

    # deepep
    os.getenv('DEEPEP_ENABLE_MNNVL', None)
    os.getenv('DEEPEP_MODE', 'auto')
    deep_ep_buffer_num_sms = env_to_int('DEEPEP_BUFFER_NUM_SMS', 20)

    # eplb
    eplb_num_groups = env_to_int('LMDEPLOY_EPLB_NUM_GROUPS', 4)
    eplb_experts_statistic_file = os.getenv('LMDEPLOY_EPLB_EXPERTS_STATISTIC_FILE', None)
    eplb_ranks_per_node = env_to_int('LMDEPLOY_EPLB_RANKS_PER_NODE', 8)
    eplb_num_redundant_experts = env_to_int('LMDEPLOY_EPLB_NUM_REDUNDANT_EXPERTS', 32)

    # deepgemm
    os.getenv('DG_JIT_DEBUG', '0')
    os.getenv('DG_JIT_PRINT_COMPILER_COMMAND', '0')

    # blocked FP8 GEMM
    blocked_fp8_gemm_backend = env_to_choice('LMDEPLOY_BLOCKED_FP8_GEMM_BACKEND', 'auto',
                                             {'auto', 'deepgemm', 'gluon', 'triton'})

    # W4A16 GEMM: the legacy env stays the base value; LMDEPLOY_TM_KERNELS
    # overrides the w4a16 class when it names it or sets a global state.
    # The first assignment is the provisional legacy value that the
    # resolution reads back; the second maps the resolved state onto the
    # provider names, so readers of the attribute (nn/linear gating,
    # tm_gemm_modules) follow the unified decision without code changes.
    w4a16_gemm_backend = env_to_choice('LMDEPLOY_W4A16_GEMM_BACKEND', 'auto',
                                       {'auto', 'triton', 'turbomind'})
    w4a16_gemm_backend = _TM_STATE_PROVIDERS[tm_kernel_state('w4a16')]

    # model agent
    skip_warmup = env_to_bool('LMDEPLOY_SKIP_WARMUP', False)

    # kernel optimizations
    router_single_group_fused = env_to_bool(
        'LMDEPLOY_ROUTER_SINGLE_GROUP_FUSED',
        False,
    )
    static_fp8_use_scaled_mm = env_to_bool(
        'LMDEPLOY_STATIC_FP8_USE_SCALED_MM',
        False,
    )
    static_fp8_use_compiled_quant = env_to_bool(
        'LMDEPLOY_STATIC_FP8_USE_COMPILED_QUANT',
        False,
    )
    static_fp8_compiled_quant_token_counts = env_to_list_int(
        'LMDEPLOY_STATIC_FP8_COMPILED_QUANT_TOKEN_COUNTS',
        [1],
    )
    moe_static_fp8_use_compiled_quant = env_to_bool(
        'LMDEPLOY_MOE_STATIC_FP8_USE_COMPILED_QUANT',
        False,
    )

    # Hy3
    hy3_shared_expert_overlap = env_to_bool(
        'LMDEPLOY_HY3_SHARED_EXPERT_OVERLAP',
        False,
    )

    # multimodal memory
    image_cache_max_bytes = env_to_int('LMDEPLOY_IMAGE_CACHE_MAX_BYTES', 4 * 1024**3)
    multimodal_session_trim_count = env_to_int('LMDEPLOY_MULTIMODAL_SESSION_TRIM_COUNT', 128)

    # model format
    scale_fmt = os.getenv('LMDEPLOY_SCALE_FMT', None)
    fp8_moe_only = env_to_bool('LMDEPLOY_FP8_MOE_ONLY', False)

    # repetition check
    repetition_window_size = env_to_int('LMDEPLOY_REPETITION_WINDOW_SIZE', 1024)

    # qwen3.5 recurrent_state dtype
    fp32_mamba_ssm_dtype = env_to_bool('LMDEPLOY_FP32_MAMBA_SSM_DTYPE', False)

    # DSA indexer fusion
    disable_dsa_indexer_fusion = env_to_bool('LMDEPLOY_DISABLE_DSA_INDEXER_FUSION', False)

    # DSA indexer score memory
    dsa_indexer_max_logits_mb = max(1, env_to_int('LMDEPLOY_DSA_INDEXER_MAX_LOGITS_MB', 512))

    # sparse MLA attention
    sparse_mla_backend = env_to_choice('LMDEPLOY_SPARSE_MLA_BACKEND', 'flashmla',
                                       {'flashmla', 'tilelang'})

    # Disable dense prefill for sparse MLA; set before startup.
    mla_disable_dense_prefill = env_to_bool('LMDEPLOY_MLA_DISABLE_DENSE_PREFILL', False)

    # Set before startup: use unrotated BF16 KPool queries/cache instead of FP8.
    glm53_kpool_bf16 = env_to_bool('LMDEPLOY_GLM53_KPOOL_BF16', False)

    # cudagraph
    # fake capture flag for debug cudagraph padding behavior
    fake_capture = env_to_bool('LMDEPLOY_FAKE_CUDA_GRAPH_CAPTURE', False)
    enable_decode_torch_compile = env_to_bool('LMDEPLOY_ENABLE_DECODE_TORCH_COMPILE', False)

    # cuda communicator
    enable_flashinfer_allreduce = env_to_bool('LMDEPLOY_ENABLE_FLASHINFER_ALLREDUCE', False)
    enable_symm_mem_allreduce = env_to_bool('LMDEPLOY_ENABLE_SYMM_MEM_ALLREDUCE', False)
    # NCCL remains the default; launch and small-token policies are internal.
    enable_symm_mem_lmhead = env_to_bool('LMDEPLOY_ENABLE_SYMM_MEM_LMHEAD', False)
    symm_mem_lmhead_max_mb = max(1, env_to_int('LMDEPLOY_SYMM_MEM_LMHEAD_MAX_MB', 64))

    # opt-ttft
    opt_ttft_policy = env_to_choice('LMDEPLOY_PT_TTFT_POLICY', 'size', {'fifo', 'size'})
    opt_ttft_short_turns = max(1, env_to_int('LMDEPLOY_PT_TTFT_SHORT_TURNS', 3))
    opt_ttft_aging_sec = env_to_float('LMDEPLOY_PT_TTFT_AGING_SEC', 2.0)


def get_all_envs():
    """Get all environment variables."""
    return _ENVS
