# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import os


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
    yield
    os.getenv = _origin_get_env


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

    # W4A16 GEMM
    w4a16_gemm_backend = env_to_choice('LMDEPLOY_W4A16_GEMM_BACKEND', 'auto',
                                       {'auto', 'triton', 'turbomind'})

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

    # memory trim
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

    # cudagraph
    # fake capture flag for debug cudagraph padding behavior
    fake_capture = env_to_bool('LMDEPLOY_FAKE_CUDA_GRAPH_CAPTURE', False)
    enable_decode_torch_compile = env_to_bool('LMDEPLOY_ENABLE_DECODE_TORCH_COMPILE', False)

    # cuda communicator
    enable_flashinfer_allreduce = env_to_bool('LMDEPLOY_ENABLE_FLASHINFER_ALLREDUCE', False)
    enable_symm_mem_allreduce = env_to_bool('LMDEPLOY_ENABLE_SYMM_MEM_ALLREDUCE', False)

    # opt-ttft
    opt_ttft_policy = env_to_choice('LMDEPLOY_PT_TTFT_POLICY', 'size', {'fifo', 'size'})
    opt_ttft_short_turns = max(1, env_to_int('LMDEPLOY_PT_TTFT_SHORT_TURNS', 3))
    opt_ttft_aging_sec = env_to_float('LMDEPLOY_PT_TTFT_AGING_SEC', 2.0)


def get_all_envs():
    """Get all environment variables."""
    return _ENVS
