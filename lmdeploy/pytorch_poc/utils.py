# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm
import inspect
from inspect import Parameter, Signature
from typing import Dict, Sequence

import psutil
import pycuda.driver as drv


def get_gpu_memory(id: int = 0) -> int:
    """Returns the free and total physical memory of the GPU in bytes."""
    drv.init()
    dev = drv.Device(id)
    cxt = dev.make_context()
    free, total = drv.mem_get_info()
    cxt.pop()
    return free, total


def get_cpu_memory() -> int:
    """Returns the total CPU memory of the node in bytes."""
    return psutil.virtual_memory().total


def bind_sigature(input_names: str, args: Sequence, kwargs: Dict):
    """Bind args and kwargs to given input names."""
    kind = inspect._ParameterKind.POSITIONAL_OR_KEYWORD

    sig = Signature([Parameter(name, kind) for name in input_names])
    bind = sig.bind(*args, **kwargs)
    return bind.arguments
