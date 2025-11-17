# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import os
from typing import Union


def env_to_bool(
    env_var: str,
    default: bool = False,
    *,
    true_values: Union[set, list] = {'true', '1', 'yes', 'on'},
    false_values: Union[set, list] = {'false', '0', 'no', 'off'},
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


_ENVS = dict()


@contextlib.contextmanager
def set_envs():
    _origin_get_env = os.getenv

    def _patched_get_env(
        env_var: str,
        default: Union[str, None] = None,
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
    ascend_rank_table_file = os.getenv('ASCEND_RANK_TABLE_FILE_PATH')

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

    # ray timeline
    ray_timeline_enable = env_to_bool('LMDEPLOY_RAY_TIMELINE_ENABLE', False)
    ray_timeline_output_path = os.getenv('LMDEPLOY_RAY_TIMELINE_OUT_PATH', 'ray_timeline.json')

    # ray external placement group bundles
    # only used when lmdeploy is initialized inside a Ray Actor with pg allocated
    ray_external_pg_bundles = env_to_list_int('LMDEPLOY_RAY_EXTERNAL_PG_BUNDLES', [])

    # dist
    dist_master_addr = os.getenv('LMDEPLOY_DIST_MASTER_ADDR', None)
    dist_master_port = os.getenv('LMDEPLOY_DIST_MASTER_PORT', None)

    # logging
    log_file = os.getenv('LMDEPLOY_LOG_FILE', None)

    # check env
    enable_check_env = env_to_bool('LMDEPLOY_ENABLE_CHECK_ENV', True)

    # dlblas
    # we don't need to read this, it would be passed to ray workers
    # If Ray is launched from outside, it may fail to access the environment variables.
    os.getenv('DEEPEP_MAX_BATCH_SIZE', None)

    # deepgemm
    os.getenv('DG_JIT_DEBUG', '0')
    os.getenv('DG_JIT_PRINT_COMPILER_COMMAND', '0')

    # model agent
    skip_warmup = env_to_bool('LMD_SKIP_WARMUP', False)


def get_all_envs():
    """Get all environment variables."""
    return _ENVS
