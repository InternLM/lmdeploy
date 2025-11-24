# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm
import inspect
from inspect import Parameter, Signature
from typing import Dict, Sequence

import psutil


def get_gpu_memory(device_id: int = None) -> int:
    """Returns the free and total physical memory of the GPU in bytes."""
    import torch
    if device_id is None:
        device_id = torch.cuda.current_device()
    return torch.cuda.mem_get_info(device_id)


def get_cpu_memory() -> int:
    """Returns the total CPU memory of the node in bytes."""
    return psutil.virtual_memory().total


def bind_sigature(input_names: str, args: Sequence, kwargs: Dict):
    """Bind args and kwargs to given input names."""
    kind = inspect._ParameterKind.POSITIONAL_OR_KEYWORD

    sig = Signature([Parameter(name, kind) for name in input_names])
    bind = sig.bind(*args, **kwargs)
    return bind.arguments


def singleton(cls):
    """Singleton decorator."""
    import multiprocessing as mp

    from lmdeploy.utils import get_logger
    logger = get_logger('lmdeploy')
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            pid = mp.current_process().pid
            logger.debug(f'pid:{pid} - Creating instance of singleton class {cls.__name__}')
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance
