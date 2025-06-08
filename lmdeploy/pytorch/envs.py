# Copyright (c) OpenMMLab. All rights reserved.
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


# profile
ray_nsys_enable = env_to_bool('LMDEPLOY_RAY_NSYS_ENABLE', False)
ray_nsys_output_prefix = os.getenv('LMDEPLOY_RAY_NSYS_OUT_PREFIX', None)

# ascend
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

# dist
dist_master_addr = os.getenv('LMDEPLOY_DIST_MASTER_ADDR', None)
dist_master_port = os.getenv('LMDEPLOY_DIST_MASTER_PORT', None)
