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
    """env to bool."""
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
