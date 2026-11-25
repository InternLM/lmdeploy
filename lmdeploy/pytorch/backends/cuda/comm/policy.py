# Copyright (c) OpenMMLab. All rights reserved.
from typing import Literal

from lmdeploy.pytorch import envs


def get_cuda_communication_policy(dist_config, group_name: Literal['tp', 'dcp'] = 'tp'):
    """Resolve opt-ins shared by Ray device binding and communicator admission.

    Return FlashInfer reduction, explicit symmetric reduction and symmetric gather flags, or None for an unsupported
    distribution. Provider conflicts are checked collectively by the communicator before allocating any workspace.
    """
    if dist_config.dp != 1 or dist_config.ep != 1 or dist_config.enable_microbatch:
        return None
    is_tp = group_name == 'tp'
    return (is_tp and envs.enable_flashinfer_allreduce,
            is_tp and envs.enable_symm_mem_allreduce,
            dist_config.communication_backend == 'auto' or (is_tp and envs.enable_symm_mem_lmhead))
